"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from core.nest import find_nested, remove_nested
from evaluation.summarisation_metrics import TEXT_TABLE_KEY
from packaging import version
from requests import RequestException
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import Seq2SeqTrainingArguments, Trainer
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import (DistributedLengthGroupedSampler,
                                           DistributedSamplerWithLoop,
                                           IterableDatasetShard,
                                           LengthGroupedSampler,
                                           find_batch_size, nested_concat,
                                           nested_numpify, nested_truncate)
from transformers.trainer_utils import (EvalLoopOutput, denumpify_detensorize,
                                        has_length, seed_worker, speed_metrics)
from transformers.training_args import ParallelMode
from transformers.utils import is_datasets_available, is_torch_tpu_available

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

_is_torch_generator_available = False

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True


@dataclass
class EvalGenerationTrainingArguments(Seq2SeqTrainingArguments):
    generation_max_new_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `max_new_tokens` to use on each evaluation loop when `predict_with_generate=True`"
                "Will default to the `max_new_tokens` value of the model configuration."
            )
        },
    )
    generation_pad_token_id: Optional[int] = field(default=None)
    training_evaluation_batches: Optional[int] = field(default=None)
    validation_evaluation_batches: Optional[int] = field(default=None)
    num_return_sequences: Optional[int] = field(default=1)
    evaluation_splits: Optional[str] = field(default=None)
    temperature: Optional[float] = field(default=1.0)
    do_sample: Optional[bool] = field(default=True)
    override_output_dir: Optional[str] = field(default=None)


def pick_dataset_subset(dataset: Dataset, num_instances: int, random=True) -> Dataset:
    return Subset(
        dataset,
        list(
            map(
                int,
                list(np.random.choice(len(dataset), num_instances, replace=False)),
            )
        )
        if random
        else list(range(num_instances)),
    )


class EvalGenerationTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_batches: Optional[int] = None,
        text_table_extra_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to
        compute metrics, as they are task-dependent (pass it to the init
        `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If
                it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. It must
                implement the `__len__` method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a
                dictionary) that should be ignored when gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For
                example the metrics "bleu" will be named "eval_bleu" if the
                prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential
            metrics computed from the predictions. The dictionary also contains
            the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        metrics: Dict[str, Any] = {}

        descriptions, key_prefixs, dataloaders, eval_batchess = [], [], [], []
        if eval_dataset is not None:
            descriptions.append(f"{metric_key_prefix.capitalize()} Evaluation")
            key_prefixs.append(metric_key_prefix)
            num_instances = min(
                self.args.eval_batch_size * eval_batches,
                len(eval_dataset),
            )
            eval_dataset = pick_dataset_subset(eval_dataset, num_instances, random=False)
            dataloaders.append(self.get_eval_dataloader(eval_dataset))
            eval_batchess.append(eval_batches)
        else:
            train_dataset = self.train_dataset
            if self.args.training_evaluation_batches:
                train_num_instances = min(
                    self.args.eval_batch_size * self.args.training_evaluation_batches,
                    len(self.train_dataset),
                )
                train_dataset = pick_dataset_subset(train_dataset, train_num_instances, random=True)
            descriptions.append("Training Evaluation")
            key_prefixs.append("")
            dataloaders.append(self.get_eval_dataloader(train_dataset))
            eval_batchess.append(eval_batches or self.args.training_evaluation_batches)

            eval_dataset = self.eval_dataset
            if self.args.validation_evaluation_batches:
                eval_num_instances = min(
                    self.args.eval_batch_size * self.args.validation_evaluation_batches,
                    len(self.eval_dataset),
                )
                eval_dataset = pick_dataset_subset(eval_dataset, eval_num_instances, random=True)
            descriptions.append("Validation Evaluation")
            key_prefixs.append("eval")
            dataloaders.append(self.get_eval_dataloader(eval_dataset))
            eval_batchess.append(eval_batches or self.args.validation_evaluation_batches)

        for description, metric_key_prefix, dataloader, eval_batches in zip(
            descriptions, key_prefixs, dataloaders, eval_batchess
        ):
            start_time = time.time()

            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
            output = eval_loop(
                dataloader,
                description=description,
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                eval_steps=eval_batches,
                text_table_extra_id=text_table_extra_id,
            )

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            metrics.update(output.metrics)
            metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_steps: Optional[int] = None,
        text_table_extra_id: Optional[int] = None,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        print(f"***** Running {description} *****")
        if eval_steps:
            print(f"  Evaluation steps: {eval_steps}")
        if has_length(dataloader):
            print(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            print("  Num examples: Unknown")
        print(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        iterator: Iterable = enumerate(dataloader)
        # Not technically necessary as the dataloader has already been cut to
        # the correct size, but better safe than sorry
        if eval_steps is not None:
            iterator = itertools.islice(iterator, eval_steps)

        for step, inputs in iterator:
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels, inputs_decode = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = (
                    labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
                )
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = (
                    logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
                )
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (
                        losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                    )
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = (
                        logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                    )
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels
                        if all_labels is None
                        else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode
                if all_inputs is None
                else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
            )

        # Number of samples
        if has_length(dataloader):
            num_samples = self.num_examples(dataloader)
        else:
            # Just use this instead of complicated checks above, to handle eval_steps argument.
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the
        # number of samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(
                labels=all_labels, inputs=all_inputs, preds=all_preds, text_table_extra_id=text_table_extra_id
            )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        if metric_key_prefix != "":
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        num_beams: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets
                 under the argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        loss, _, _ = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=True,
            ignore_keys=ignore_keys,
        )

        has_labels = "labels" in inputs or "label" in inputs
        inputs = self._prepare_inputs(inputs)
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False

        gen_kwargs = {
            "num_beams": num_beams or self.args.generation_num_beams or self.model.config.num_beams,
            "max_new_tokens": max_new_tokens or self.args.generation_max_new_tokens,
            "pad_token_id": pad_token_id or self.args.generation_pad_token_id,
            "synced_gpus": default_synced_gpus,
            "temperature": self.args.temperature,
            "do_sample": self.args.do_sample,
            # "num_return_sequences": self.args.num_return_sequences,
        }

        model_inputs = self.prepare_generation_inputs(
            inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        )
        try:
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
        except AttributeError:
            # Model is DP so doesn't have .device, but that's ok.
            pass

        response_ids = self.model.generate(**model_inputs, **gen_kwargs)[
            :, model_inputs["input_ids"].shape[1] :
        ]
        # in case the batch is shorter than max length, the output should be padded
        if response_ids.shape[-1] < gen_kwargs["max_new_tokens"]:
            response_ids = self._pad_tensors_to_max_len(response_ids, gen_kwargs["max_new_tokens"])

        if has_labels:
            labels = inputs.get("labels", inputs.get("label"))
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])
            labels = labels.contiguous()
        else:
            labels = None

        return loss, response_ids.contiguous(), labels, model_inputs["input_ids"]

    def prepare_generation_inputs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return new dict of input_ids and attention_mask for testing
        generation by removing the label outputs."""
        output_sizes = torch.count_nonzero(labels + 100, dim=1)
        min_output_size = output_sizes.min().item()
        new_input_ids = torch.full(
            (input_ids.shape[0], input_ids.shape[1] - min_output_size),
            self.tokenizer.pad_token_id,
            device=input_ids.device,
        )
        new_attention_mask = torch.full(
            (input_ids.shape[0], input_ids.shape[1] - min_output_size), 0, device=input_ids.device
        )
        for i in range(input_ids.shape[0]):
            new_input_ids[i, output_sizes[i] - min_output_size :] = input_ids[i, : -output_sizes[i]]
            new_attention_mask[i, output_sizes[i] - min_output_size :] = attention_mask[i, : -output_sizes[i]]
        return {"input_ids": new_input_ids, "attention_mask": new_attention_mask}

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError(
                    "Pad_token_id must be set in the configuration of the model, in order to pad tensors"
                )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        # We don't need to save logs in the state, as we're getting them on
        # wandb, and there's various things we want to log that can't be
        # converted to json unfortunately (e.g. wandb tables, np arrays)
        # output = {**logs, **{"step": self.state.global_step}}
        # self.state.log_history.append(output)
        try:
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        except RequestException as e:
            text_tables = find_nested(lambda key: TEXT_TABLE_KEY in key, logs)
            print(
                "Failed to log to WANDB due to errors ",
                e,
                "with table json:\n%s",
                [t._to_table_json() for t in text_tables],
            )
            new_logs = remove_nested(lambda key: TEXT_TABLE_KEY in key, logs)
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, new_logs)

    def _nested_gather(self, tensors):
        if hasattr(self, "accelerator"):
            return self.accelerator.gather_for_metrics((tensors))
        else:
            return super()._nested_gather(tensors)

    def _pad_across_processes(self, tensors, **kwargs):
        if hasattr(self, "accelerator"):
            return self.accelerator.pad_across_processes(tensors, **kwargs)
        else:
            return super()._pad_across_processes(tensors)
