"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import time
import typing as t

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup

import wandb
from algos.ppo import AutoRegressivePPOTrainer
from evaluation.ni_metrics import compute_all_ni_metrics
from evaluation.reward_functions import RewardFunction
from evaluation.sentiment_metrics import compute_all_sentiment_metrics
from evaluation.summarisation_metrics import compute_all_summarisation_metrics


class Experiment:
    def __init__(
        self,
        dataloaders: t.Dict[str, DataLoader],
        ppo_trainer: AutoRegressivePPOTrainer,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        fixed_model: PreTrainedModel,
        reward_function: RewardFunction,
        device: torch.device,
        args,
    ) -> None:
        self.dataloaders = dataloaders
        self.train_dataloader = self.dataloaders["train"]
        self.validation_dataloader = self.dataloaders["validation"]
        self.test_dataloader = self.dataloaders["test"]
        self.device = device
        self.ppo_trainer = ppo_trainer
        self.model = model
        self.tokenizer = tokenizer
        self.fixed_model = fixed_model
        self.reward_function = reward_function
        self.args = args

        generation_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            pad_token_id=self.model.config.eos_token_id,
            top_p=1.0,
            top_k=0,
            do_sample=True,
            temperature=1,
        )

        for k, v in generation_kwargs.items():
            if getattr(args, k, None) is not None:
                generation_kwargs[k] = getattr(args, k)

        self.rollout_generation_kwargs = generation_kwargs.copy()
        self.evaluation_generation_kwargs = generation_kwargs.copy()
        self.evaluation_generation_kwargs.update({"temperature": 0, "do_sample": False})

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        num_training_steps = self.args.epochs * len(self.train_dataloader)

        if args.max_train_steps > 0:
            print("Max train steps passed, ignoring epoch argument")
            num_training_steps = args.max_train_steps

        self.lr_schedule = get_linear_schedule_with_warmup(
            optimizer=ppo_trainer.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        self.num_batches = len(self.train_dataloader)
        self.epoch = 0
        self.batch_num = 0
        self.num_steps_trained = 0
        if args.resumed:
            if self.load_checkpoint():
                self.epoch = self.args.checkpoint_epoch
                self.batch_num = self.args.checkpoint_batch_num
                self.num_steps_trained = self.args.checkpoint_num_steps

    def prepare_model_inputs(
        self, batch: t.Dict[str, t.Any], return_text_inputs: bool = False
    ) -> t.Tuple[t.Dict[str, torch.Tensor], t.Optional[t.List]]:
        text_inputs = None

        if self.args.text_only:
            # For RL, we want to pad to the left so that the batch of inputs are
            # right-aligned, and hence generation can be batched reasonably well
            prev_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            model_inputs = self.tokenizer(
                batch["inputs"],
                truncation=True,
                return_tensors="pt",
                padding=True,
                max_length=self.args.max_source_length,
            ).to(self.device)
            self.tokenizer.padding_side = prev_padding_side
            if return_text_inputs:
                text_inputs = batch["inputs"]
        else:
            model_inputs = {
                k: v.to(self.device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}
            }
            if return_text_inputs:
                text_inputs = self.tokenizer.batch_decode(model_inputs["input_ids"], skip_special_tokens=True)

        return model_inputs, text_inputs

    def run_training(self):
        print("Starting training...")
        if self.args.max_train_steps > 0:
            num_steps_trained = self.num_steps_trained
            epoch = self.epoch
            while num_steps_trained < self.args.max_train_steps:
                print("Starting epoch {}".format(epoch))
                for batch_num, batch in tqdm(enumerate(iter(self.train_dataloader))):
                    self.inner_training_loop(batch_num, epoch, batch, num_steps_trained)
                    num_steps_trained += 1
                    if num_steps_trained > self.args.max_train_steps:
                        print("Max train steps reached, exiting")
                        return
                epoch += 1
        else:
            for epoch in range(self.epoch, self.args.epochs):
                print("Starting epoch {}".format(epoch))
                for batch_num, batch in tqdm(
                    zip(range(self.batch_num, self.num_batches), (iter(self.train_dataloader)))
                ):
                    self.inner_training_loop(batch_num, epoch, batch, epoch * self.num_batches + batch_num)

    def inner_training_loop(self, batch_num: int, epoch: int, batch: t.Dict, num_steps_trained: int):
        stats = {}
        if (batch_num + 1) % self.args.log_interval == 0:
            # E.g. for log_interval = 10, log every 10 batches at batch 9, 19, 29, ...
            stats.update(
                self.run_evaluation(
                    splits=["validation"],
                    log_to_wandb=False,
                    num_evaluation_batches=self.args.validation_evaluation_batches,
                )
            )

        if (
            self.args.checkpoint_steps > 0
            and ((epoch * self.num_batches) + batch_num) % self.args.checkpoint_steps == 0
            and (batch_num > 0 or epoch > 0)
        ):
            checkpoint_time = self.save_checkpoint(epoch, batch_num, num_steps_trained)
            stats.update({"time/checkpoint_time": checkpoint_time})

        timing = dict()
        t0 = time.time()
        tm = time.time()

        log_text_generations = batch_num % self.args.log_interval == 0

        # Generation
        batch, text_inputs = self.add_generations_to_batch(
            batch,
            return_text_inputs=True,
            batch_num=batch_num,
            epoch=epoch,
            generation_kwargs=self.rollout_generation_kwargs,
        )
        timing["time/get_response"] = time.time() - tm
        batch["queries"] = text_inputs

        # Reward Calculation
        rewards = self.reward_function(batch, return_tensor=True)

        if log_text_generations and text_inputs is not None:
            stats.update({"train/text_table": self.prepare_text_table(text_inputs, batch, rewards)})

        response_lengths = np.array([len(r) for r in self.tokenizer(batch["responses"])["input_ids"]])

        # Training
        tm = time.time()
        stats.update(
            self.ppo_trainer.step(batch["input_ids"], batch["response_ids"], rewards, batch["attention_mask"])
        )
        self.lr_schedule.step()
        timing["time/optimization"] = time.time() - tm

        rewards = np.array(rewards.cpu())

        # Logging
        timing["time/batch"] = time.time() - t0
        stats.update(timing)
        stats.update(
            {
                "epoch": epoch,
                "batch": batch_num,
                "lr": self.lr_schedule.get_last_lr()[0],
                "train/average_reward": np.mean(rewards),
                "train/median_reward": np.median(rewards),
                "train/reward_distribution": rewards,
                "train/average_response_length": np.mean(response_lengths),
                "train/median_response_length": np.median(response_lengths),
                "train/response_length_distribution": response_lengths,
            }
        )
        wandb.log(stats, step=(epoch * self.num_batches) + batch_num)

    def _get_checkpoint_paths(self):
        checkpoint_path = os.path.join(self.args.model_dir, wandb.run.id)
        model_path = os.path.join(checkpoint_path, "model.pt")
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        arg_path = os.path.join(checkpoint_path, "args.bin")
        lr_scheduler_path = os.path.join(checkpoint_path, "lr_schedule.pt")
        return model_path, optimizer_path, arg_path, lr_scheduler_path

    def save_checkpoint(self, epoch: int, batch_num: int, num_steps_trained: int) -> float:
        print("Checkpointing...")
        tm = time.time()
        checkpoint_path = os.path.join(self.args.model_dir, wandb.run.id)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        model_path, optimizer_path, arg_path, lr_scheduler_path = self._get_checkpoint_paths()

        self.args.checkpoint_epoch = epoch
        self.args.checkpoint_batch_num = batch_num
        self.args.checkpoint_num_steps = num_steps_trained

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.args, arg_path)
        torch.save(self.ppo_trainer.optimizer.state_dict(), optimizer_path)
        torch.save(self.lr_schedule.state_dict(), lr_scheduler_path)

        checkpoint_time = time.time() - tm
        return checkpoint_time

    def load_checkpoint(self) -> bool:
        print("Loading checkpoint...")
        if not os.path.exists(os.path.join(self.args.model_dir, wandb.run.id, "model.pt")):
            print("No checkpoint found")
            return False

        model_path, optimizer_path, arg_path, lr_scheduler_path = self._get_checkpoint_paths()

        self.model.load_state_dict(torch.load(model_path))
        self.ppo_trainer.optimizer.load_state_dict(torch.load(optimizer_path))
        self.args = torch.load(arg_path)
        self.ppo_trainer.init_ppo_params(self.args)
        self.lr_schedule.load_state_dict(torch.load(lr_scheduler_path))

        return True

    def add_generations_to_batch(
        self,
        batch: t.Dict,
        return_text_inputs: bool,
        generation_kwargs: t.Dict[str, t.Any],
        batch_num: int = 0,
        epoch: int = 0,
    ) -> t.Tuple[t.Dict, t.Optional[t.List[str]]]:
        model_inputs, text_inputs = self.prepare_model_inputs(batch, return_text_inputs=return_text_inputs)
        batch.update(model_inputs)
        gen_start = model_inputs["input_ids"].shape[1]
        # Split batch input chunks of size args.generation_batch_size
        chunks = [
            {k: v[i : i + self.args.generation_batch_size] for k, v in model_inputs.items()}
            for i in range(0, self.args.batch_size, self.args.generation_batch_size)
        ]
        outputs = []
        max_gen_len = 0
        for chunk in chunks:
            gen_kwargs = dict(**chunk, **generation_kwargs)
            with torch.no_grad():
                try:
                    response_ids = self.model.generate(**gen_kwargs)[:, gen_start:]
                except RuntimeError as e:
                    print(f"Got error when generating on batch {batch_num} on epoch {epoch}: {e}")
                    print("Generating with top_k=100 to see if that fixes the issue...")
                    gen_kwargs["top_k"] = 100
                    response_ids = self.model.generate(**gen_kwargs)[:, gen_start:]
                max_gen_len = max(max_gen_len, response_ids.shape[-1])
                if response_ids.shape[-1] != gen_kwargs["max_new_tokens"]:
                    # Pad response_ids to have shape[-1] == max_new_tokens
                    response_ids = torch.cat(
                        [
                            response_ids,
                            torch.full(
                                size=(
                                    response_ids.shape[0],
                                    gen_kwargs["max_new_tokens"] - response_ids.shape[-1],
                                ),
                                fill_value=self.tokenizer.eos_token_id,
                                device=response_ids.device,
                            ),
                        ],
                        dim=-1,
                    )
                outputs.append(response_ids)
        # Concatenate outputs, removing unncesary padding if all generations didn't reach max length
        batch["response_ids"] = torch.cat(outputs, dim=0)[:, :max_gen_len]
        batch["responses"] = self.tokenizer.batch_decode(batch["response_ids"], skip_special_tokens=True)
        batch["attention_mask"] = torch.cat(
            [model_inputs["attention_mask"], torch.ones_like(batch["response_ids"])], dim=1
        )
        return batch, text_inputs

    def run_evaluation(
        self,
        splits: t.List[str] = ["validation", "test"],
        log_to_wandb=True,
        num_evaluation_batches: t.Optional[int] = None,
    ):
        raise NotImplementedError()

    def prepare_text_table(
        self, text_inputs: t.List[str], batch: t.Dict, rewards: torch.Tensor
    ) -> wandb.Table:
        raise NotImplementedError()


class NIExperiment(Experiment):
    def run_evaluation(
        self,
        splits: t.List[str] = ["validation", "test"],
        log_to_wandb=True,
        num_evaluation_batches: t.Optional[int] = None,
    ):
        # iterate through data*set* and dataloader, generate (in batches) outputs from the model.
        # Then pass them to compute_metrics function along with the groups from the dataset, and
        # then log the computed metrics to wandb
        datasets = [(dset, dload) for dset, dload in self.dataloaders.items() if dset in splits]
        all_metrics = {}
        print("Starting evaluation...")
        for data_set, dataloader in datasets:
            print("Starting evaluation on {}".format(data_set))
            all_labels, all_outputs, all_categories, all_domains, all_tasks = [], [], [], [], []
            all_inputs, all_rewards, all_response_ids = [], [], []  # type: ignore
            if num_evaluation_batches:
                # Zip dataloader with range to that we limit number of batches from the dataloader
                dataloader = zip(dataloader, range(num_evaluation_batches))
            else:
                dataloader = zip(dataloader, range(len(dataloader)))
            # Note that as the dataloader is shuffled, and we create a new iter each time, it will
            # select different batches from the dataloader in each evaluation run.
            # This only matters if num_evaluation_batches is not None, as then we subsample batches.

            for batch, _ in tqdm(iter(dataloader)):
                all_labels.extend(batch["labels"])
                all_categories.extend(batch["categories"])
                all_domains.extend(batch["domains"])
                all_tasks.extend(batch["tasks"])

                batch, text_inputs = self.add_generations_to_batch(
                    batch, return_text_inputs=True, generation_kwargs=self.evaluation_generation_kwargs
                )

                all_outputs.extend(batch["responses"])
                all_response_ids.extend(batch["response_ids"].cpu().numpy())
                all_inputs.extend(text_inputs)  # type: ignore

                rewards = self.reward_function(batch, return_tensor=False)
                all_rewards.extend(rewards)

            metrics = compute_all_ni_metrics(
                predictions=all_outputs,
                token_predictions=all_response_ids,
                references=all_labels,
                categories=all_categories,
                tasks=all_tasks,
                domains=all_domains,
                inputs=all_inputs,
                rewards=all_rewards,
                tokenizer=self.tokenizer,
            )

            metrics = {f"{data_set}/{k}": v for k, v in metrics.items()}

            all_metrics.update(metrics)

        if log_to_wandb:
            wandb.log(all_metrics)
        return all_metrics

    def prepare_text_table(
        self, text_inputs: t.List[str], batch: t.Dict, rewards: torch.Tensor
    ) -> wandb.Table:
        text_table = wandb.Table(columns=["input", "response", "label", "reward"])
        for input, response, label, reward in zip(
            text_inputs, batch["responses"], [label[0] for label in batch["labels"]], rewards
        ):
            text_table.add_data(input, response, label, reward)
        return text_table


class IMDBExperiment(Experiment):
    def run_evaluation(
        self,
        splits: t.List[str] = ["validation"],
        log_to_wandb=True,
        num_evaluation_batches: t.Optional[int] = None,
    ):
        # iterate through data*set* and dataloader, generate (in batches) outputs from the model.
        # Then pass them to compute_metrics function along with the groups from the dataset, and
        # then log the computed metrics to wandb
        datasets = [(dset, dload) for dset, dload in self.dataloaders.items() if dset in splits]
        all_metrics = {}
        print("Starting evaluation...")
        for data_set, dataloader in datasets:
            print("Starting evaluation on {}".format(data_set))
            sentiments, references, inputs, rewards, response_ids, outputs = [], [], [], [], [], []
            if num_evaluation_batches:
                # Zip dataloader with range to that we limit number of batches from the dataloader
                dataloader = zip(dataloader, range(num_evaluation_batches))
            else:
                dataloader = zip(dataloader, range(len(dataloader)))
            # Note that as the dataloader is shuffled, and we create a new iter each time, it will
            # select different batches from the dataloader in each evaluation run.
            # This only matters if num_evaluation_batches is not None, as then we subsample batches.

            for batch, _ in tqdm(iter(dataloader)):
                batch, text_inputs = self.add_generations_to_batch(
                    batch, return_text_inputs=True, generation_kwargs=self.evaluation_generation_kwargs
                )
                batch_rewards = self.reward_function(batch, return_tensor=False)

                inputs.extend(text_inputs or [])
                rewards.extend(batch_rewards)
                outputs.extend(batch["responses"])
                response_ids.extend(batch["response_ids"].cpu().numpy())
                references.extend(batch["labels"])
                sentiments.extend(batch["sentiment"])

            metrics = compute_all_sentiment_metrics(
                predictions=outputs,
                references=references,
                token_predictions=response_ids,
                inputs=inputs,
                rewards=rewards,
                sentiments=sentiments,
                tokenizer=self.tokenizer,
            )

            metrics = {f"{data_set}/{k}": v for k, v in metrics.items()}

            all_metrics.update(metrics)

        if log_to_wandb:
            wandb.log(all_metrics)
        return all_metrics

    def prepare_text_table(
        self, text_inputs: t.List[str], batch: t.Dict, rewards: torch.Tensor
    ) -> wandb.Table:
        text_table = wandb.Table(columns=["input", "response", "label", "reward"])
        for input, response, label, reward in zip(
            text_inputs, batch["responses"], batch["sentiment"], rewards
        ):
            text_table.add_data(input, response, label, reward)
        return text_table


class SummarisationExperiment(Experiment):
    def run_evaluation(
        self,
        splits: t.List[str] = ["validation"],
        log_to_wandb=True,
        num_evaluation_batches: t.Optional[int] = None,
    ):
        # iterate through data*set* and dataloader, generate (in batches) outputs from the model.
        # Then pass them to compute_metrics function along with the groups from the dataset, and
        # then log the computed metrics to wandb
        datasets = [(dset, dload) for dset, dload in self.dataloaders.items() if dset in splits]
        all_metrics = {}
        print("Starting evaluation...")
        for data_set, dataloader in datasets:
            print("Starting evaluation on {}".format(data_set))
            references, inputs, rewards, response_ids, outputs = [], [], [], [], []
            if num_evaluation_batches:
                # Zip dataloader with range to that we limit number of batches from the dataloader
                dataloader = zip(dataloader, range(num_evaluation_batches))
            else:
                dataloader = zip(dataloader, range(len(dataloader)))
            # Note that as the dataloader is shuffled, and we create a new iter each time, it will
            # select different batches from the dataloader in each evaluation run.
            # This only matters if num_evaluation_batches is not None, as then we subsample batches.

            for batch, _ in tqdm(iter(dataloader)):
                batch, text_inputs = self.add_generations_to_batch(
                    batch, return_text_inputs=True, generation_kwargs=self.evaluation_generation_kwargs
                )
                batch_rewards = self.reward_function(batch, return_tensor=False)

                inputs.extend(text_inputs or [])
                rewards.extend(batch_rewards)
                outputs.extend(batch["responses"])
                response_ids.extend(batch["response_ids"].cpu().numpy())
                references.extend(batch["labels"])

            metrics = compute_all_summarisation_metrics(
                predictions=outputs,
                references=references,
                token_predictions=response_ids,
                inputs=inputs,
                rewards=rewards,
                tokenizer=self.tokenizer,
            )

            metrics = {f"{data_set}/{k}": v for k, v in metrics.items()}

            all_metrics.update(metrics)

        if log_to_wandb:
            wandb.log(all_metrics)
        return all_metrics

    def prepare_text_table(
        self, text_inputs: t.List[str], batch: t.Dict, rewards: torch.Tensor
    ) -> wandb.Table:
        text_table = wandb.Table(columns=["input", "response", "label", "reward"])
        for input, response, label, reward in zip(text_inputs, batch["responses"], batch["labels"], rewards):
            text_table.add_data(input, response, label, reward)
        return text_table
