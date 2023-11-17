"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from core import vtrace
from utils.core import logprobs_from_logits


def looped(dataloader):
    while True:
        for i in dataloader:
            yield i


def calculate_kl_control_reward(actor_logits, frozen_logits, actions, kl_controller, FLAGS):
    actor_logprobs = logprobs_from_logits(actor_logits, actions)
    frozen_logprobs = logprobs_from_logits(frozen_logits, actions.to(frozen_logits.device)).to(
        actor_logprobs.device
    )
    kl_divergence = vtrace.calculate_kl_divergence(actor_logprobs, frozen_logprobs, FLAGS)
    return kl_divergence * -kl_controller.value, kl_divergence


def compute_entropy(logits):
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
    return entropy_per_timestep


class MockEnv:
    def __init__(
        self, dataloaders, tokenizer, model, frozen_model, reward_fn, kl_controller, accelerator, flags
    ) -> None:
        self.dataloaders = dataloaders
        self.tokenizer = tokenizer
        self.model = model
        self.frozen_model = frozen_model
        self.reward_fn = reward_fn
        self.kl_controller = kl_controller
        self.flags = flags
        self.device = flags.device
        self.frozen_model_device = getattr(
            self.frozen_model.get_base_model_transformer(), "first_device", self.flags.device
        )

        self.train_gen = iter(looped(dataloaders["train"]))
        self.eval_dataloaders = dataloaders
        self.accelerator = accelerator
        frozen_model.requires_grad = False

    def generation_kwargs(self, model, eval=False):
        return dict(
            max_new_tokens=self.flags.max_new_tokens,
            top_p=self.flags.top_p,
            top_k=self.flags.top_k,
            do_sample=self.flags.do_sample if not eval else False,
            temperature=self.flags.temperature if not eval else 0,
        )

    @torch.no_grad()
    def calculate_generation_results(
        self, batch: Dict[str, Any], eval: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.flags.device)
        attention_mask = batch["attention_mask"].to(self.flags.device)

        gen_kwargs = self.generation_kwargs(self.model, eval)
        gen_kwargs["input_ids"] = input_ids
        gen_kwargs["attention_mask"] = attention_mask

        gen_start = input_ids.shape[1]

        try:
            response_ids = self.accelerator.unwrap_model(self.model).generate(**gen_kwargs)[:, gen_start:]
        except RuntimeError as e:
            logging.info(
                f"Got error when generating ({e})."
                " Generating with top_k=100 to see if that fixes the issue..."
            )
            gen_kwargs["top_k"] = 100
            response_ids = self.accelerator.unwrap_model(self.model).generate(**gen_kwargs)[:, gen_start:]

        # Clamp response ids to maximum tokenizer vocab, necessary for OPT
        # which can output invalid tokens
        response_ids = torch.clamp(response_ids, 0, self.tokenizer.vocab_size - 1)

        return response_ids, input_ids, attention_mask

    @torch.no_grad()
    def train_rollout(self):
        return self.make_rollout(next(self.train_gen), eval=False)

    @torch.no_grad()
    def make_rollout(self, batch, eval=False) -> Dict[str, Any]:
        response_ids, input_ids, attention_mask = self.calculate_generation_results(batch, eval=eval)

        # text_in = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        text_out = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        reward_args = {
            "queries": batch["queries"],
            "responses": text_out,
            "labels": batch["labels"],
            "response_ids": response_ids,
        }

        final_rewards = self.reward_fn(reward_args, return_tensor=True)

        auxiliary_rewards = torch.zeros_like(final_rewards).float()

        response_lengths = (response_ids != self.tokenizer.pad_token_id).sum(dim=1)

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            # We will miscount if pad token and eos token are different
            response_lengths = torch.clamp(response_lengths + 1, max=self.flags.max_new_tokens)

        if self.flags.length_penalty:
            # Calculate length of non-padding token responses
            scaled_response_lengths = response_lengths / self.flags.max_new_tokens

            auxiliary_rewards += -self.flags.length_penalty_coef * (
                scaled_response_lengths**self.flags.length_penalty_power
            )

        if self.flags.eos_reward:
            # Add reward for having an eos_token in generation
            auxiliary_rewards += (
                self.flags.eos_reward_coef
                * ((response_ids == self.tokenizer.eos_token_id).sum(dim=1) > 0).float()
            )

        final_rewards += auxiliary_rewards

        if self.flags.eos_required:
            # Zero out rewards for responses that don't have an eos_token
            final_rewards *= (response_ids == self.tokenizer.eos_token_id).sum(dim=1) > 0
            final_rewards -= (
                self.flags.eos_required_penalty
                * ((response_ids == self.tokenizer.eos_token_id).sum(dim=1) == 0).float()
            )

        # Instead of setting reward at end of generate, set it at the sequence length of the response
        rewards = torch.zeros_like(response_ids).float()
        rewards[torch.arange(rewards.shape[0]), response_lengths - 1] = final_rewards

        # Same adjustment for done
        done = torch.zeros_like(response_ids).float()
        done[torch.arange(rewards.shape[0]), response_lengths - 1] = 1.0

        # Mask out all tokens after the *response* length for all loss calculations
        mask = torch.zeros_like(response_ids).float()
        mask[torch.arange(mask.shape[0]), response_lengths - 1] = 1.0
        mask = 1.0 - (mask.cumsum(dim=1) - mask)

        # Now that generation is done, with beams etc, measure log probs
        full_sequence = torch.cat([input_ids, response_ids], dim=1)
        full_mask = torch.cat([attention_mask, torch.ones_like(response_ids)], dim=1)
        out = self.model(full_sequence, attention_mask=full_mask)
        f_out = self.frozen_model(
            full_sequence.to(self.frozen_model_device),
            attention_mask=full_mask.to(self.frozen_model_device),
        )
        gen_start = full_sequence.shape[1] - response_ids.shape[1]

        policy_logits = out.logits[:, gen_start - 1 : -1, :]
        frozen_policy_logits = f_out.logits[:, gen_start - 1 : -1, :]

        kl_control_rewards, kl_div = calculate_kl_control_reward(
            policy_logits,
            frozen_policy_logits,
            response_ids,
            self.kl_controller,
            self.flags,
        )

        if eval:
            result = dict(
                token_predictions=response_ids.cpu().numpy().tolist(),
                rewards=final_rewards.cpu().numpy().tolist(),
                kl_div=kl_div.sum(dim=1).cpu().numpy().tolist(),
                kl_rewards=kl_control_rewards.sum(dim=1).cpu().numpy().tolist(),
                auxiliary_rewards=auxiliary_rewards.cpu().numpy().tolist(),
                entropy=compute_entropy(policy_logits).sum(dim=1).cpu().numpy().tolist(),
                predictions=text_out,
                inputs=batch["queries"],
                references=batch["labels"],
            )
            if sentiment := batch.get("sentiment", False):
                result["sentiment"] = sentiment.cpu().numpy()

            return result

        logging_outputs = {
            "text_in": batch["queries"],
            "text_out": text_out,
            "labels": batch["labels"],
        }
        if batch.get("sentiment") is not None:
            logging_outputs["sentiment"] = batch["sentiment"]

        return {
            "env_outputs": {
                "input_ids": batch["input_ids"],
                "attention_mask": attention_mask,
                "full_sequence": full_sequence,
                "full_mask": full_mask,
                "reward": rewards,
                "kl_rewards": kl_control_rewards,
                "mask": mask,
                "kl_div": kl_div,
                "done": done,
            },
            "actor_outputs": {
                "action": response_ids,
                "policy_logits": policy_logits.clone(),
                "baseline": out.value[:, gen_start - 1 : -1].clone(),
                "frozen_policy_logits": frozen_policy_logits.clone(),
                # "full_logits": out.logits,
                # "full_baseline": out.value,
            },
            "logging_outputs": logging_outputs,
        }

    @torch.no_grad()
    def run_evaluation(
        self,
        splits: List[str] = ["validation"],
        num_evaluation_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        # iterate through data*set* and dataloader, generate (in batches) outputs from the model.
        # Then pass them to compute_metrics function along with the groups from the dataset, and
        # then log the computed metrics to wandb
        datasets = [(dset, dload) for dset, dload in self.eval_dataloaders.items() if dset in splits]
        all_results = {}
        logging.info("Starting evaluation...")
        for data_set, dataloader in datasets:
            logging.info("Starting evaluation on {}".format(data_set))
            results = []
            if num_evaluation_batches:
                # Zip dataloader with range to that we limit number of batches from the dataloader
                dataloader = zip(dataloader, range(num_evaluation_batches))
            else:
                dataloader = zip(dataloader, range(len(dataloader)))

            for batch, _ in iter(dataloader):
                results.append(self.make_rollout(batch, eval=True))
                torch.cuda.empty_cache()

            # convert list of dicts `results` to a dict of lists
            results_dict = {k: [v for d in results for v in d[k]] for k in results[0]}

            all_results[data_set] = results_dict
        return all_results
