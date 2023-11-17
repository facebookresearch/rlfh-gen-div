"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
import torch
from packaging import version
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    IterableDatasetShard,
    LengthGroupedSampler,
)
from transformers.trainer_utils import has_length, seed_worker, speed_metrics
from transformers.training_args import ParallelMode
from transformers.utils import is_datasets_available

from lab.sl_trainer import pick_dataset_subset

_is_torch_generator_available = False

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True


@dataclass
class RewardFunctionTrainingArguments(TrainingArguments):
    training_evaluation_batches: Optional[int] = field(default=None)
    validation_evaluation_batches: Optional[int] = field(default=None)


class RewardFunctionTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        evaluation_batches: Optional[int] = None,
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

        descriptions, key_prefixs, dataloaders = [], [], []
        if eval_dataset is not None:
            descriptions.append(f"{metric_key_prefix.capitalize()} Evaluation")
            key_prefixs.append(metric_key_prefix)
            num_instances = min(
                self.args.eval_batch_size * evaluation_batches,
                len(eval_dataset),
            )
            eval_dataset = pick_dataset_subset(eval_dataset, num_instances, random=False)
            dataloaders.append(self.get_eval_dataloader(eval_dataset))
        else:
            train_dataset = self.train_dataset
            if self.args.training_evaluation_batches or evaluation_batches:
                batches = evaluation_batches or self.args.training_evaluation_batches
                train_num_instances = min(self.args.eval_batch_size * batches, len(self.train_dataset))
                train_dataset = pick_dataset_subset(train_dataset, train_num_instances)

            descriptions.append("Training Evaluation")
            key_prefixs.append("")
            dataloaders.append(self.get_train_dataloader(train_dataset, evaluation=True))

            eval_dataset = self.eval_dataset
            if self.args.validation_evaluation_batches or evaluation_batches:
                batches = evaluation_batches or self.args.validation_evaluation_batches
                eval_num_instances = min(self.args.eval_batch_size * batches, len(self.eval_dataset))
                eval_dataset = pick_dataset_subset(eval_dataset, eval_num_instances)

            descriptions.append("Validation Evaluation")
            key_prefixs.append("eval")
            dataloaders.append(self.get_eval_dataloader(eval_dataset))

        for description, mk_prefix, dataloader in zip(
            descriptions,
            key_prefixs,
            dataloaders,
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
                metric_key_prefix=mk_prefix,
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

        total_train_batch_size = (
            self._train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        )
        total_samples_seen = self.state.global_step * total_train_batch_size
        metrics["total_samples_seen"] = total_samples_seen
        metrics["percent_progress"] = int(total_samples_seen * 100 / (self.state.max_steps or 1))

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def get_train_dataloader(self, train_dataset=None, evaluation: bool = False) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a
        random sampler (adapted to distributed training if necessary)
        otherwise.

        Subclass and override this method if you want to inject some custom
        behavior.
        """
        if self.train_dataset is None and train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if train_dataset is None:
            train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler(train_dataset, evaluation)

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def _get_train_sampler(
        self, train_dataset=None, evaluation: bool = False
    ) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset

        if train_dataset is None or not has_length(train_dataset):
            return None

        if evaluation:
            return self._get_eval_sampler(train_dataset)

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is
            # sampled from a generator seeded with `args.seed`) if data_seed
            # isn't provided. Further on in this method, we default to
            # `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    return RandomSampler(train_dataset, generator=generator)
                return RandomSampler(train_dataset)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
