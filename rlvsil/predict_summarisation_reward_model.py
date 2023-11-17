"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from datasets import load_metric
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint

import wandb
from dataset.summarisation_dataloader import make_rf_prediction_dataset
from dataset.summarisation_feedback_dataloader import make_dataset
from lab.rf_trainer import RewardFunctionTrainer, RewardFunctionTrainingArguments


@dataclass
class Args:
    checkpoint_name: str = field()
    summarisation_dataset_queries: bool = field(default=False)
    dataset: str = field(default="sl")  # Literal["sl", "rl", "rf"]
    wandb_run_name: Optional[str] = field(default=None)
    wandb_tags: Optional[str] = field(default=None)
    wandb_group: Optional[str] = field(default=None)
    wandb_project: Optional[str] = field(default=None)
    wandb_entity: Optional[str] = field(default="robkirk")
    run_id: Optional[str] = field(default=None)
    model_name: Optional[str] = field(default="gpt2")
    eval_subsample_num: Optional[int] = field(default=None)


def predict_reward_model(args: Args, training_args: RewardFunctionTrainingArguments):
    job_id = int(os.environ.get("SLURM_ARRAY_JOB_ID", 0))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    config = dict(**asdict(args), **training_args.to_sanitized_dict())
    config["job_id"] = job_id
    config["task_id"] = task_id

    if not args.run_id:
        if job_id and task_id:
            run_id = f"{job_id}-{task_id}"
        else:
            run_id = wandb.util.generate_id()
    else:
        run_id = args.run_id

    if training_args.process_index == 0:
        wandb.init(
            id=run_id,
            resume="allow",
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=args.wandb_tags.split(",") if args.wandb_tags else None,
            group=args.wandb_group,
            config=config,
        )

        if wandb.run.resumed:
            print("Resuming run")
        else:
            print("Initializing run")

    training_args.output_dir = f"checkpoints/summarisation_rf/{run_id}"
    training_args.overwrite_output_dir = True
    training_args.evaluation_strategy = "steps"
    training_args.save_total_limit = 100
    training_args.report_to = ["wandb"]
    training_args.lr_scheduler_type = "cosine"

    if os.path.isdir(training_args.output_dir) and get_last_checkpoint(training_args.output_dir) is not None:
        training_args.resume_from_checkpoint = True

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.dataset == "sl":
        dss = make_rf_prediction_dataset(args, tokenizer)
    elif args.dataset == "rl":
        args.summarisation_dataset_queries = True
        dss = make_rf_prediction_dataset(args, tokenizer)
    elif args.dataset == "rf":
        dss = make_dataset(tokenizer)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    def compute_metrics(eval_preds):
        acc_metric = load_metric("accuracy")
        precision_metric = load_metric("precision")
        recall_metric = load_metric("recall")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return dict(
            **acc_metric.compute(predictions=predictions, references=labels),
            **precision_metric.compute(predictions=predictions, references=labels),
            **recall_metric.compute(predictions=predictions, references=labels),
        )

    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = RewardFunctionTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    for key, ds in dss.items():
        print("Generating predictions for ds, key", args.dataset, key)
        with torch.no_grad():
            output = trainer.predict(ds, metric_key_prefix=key)
        wandb.log(output.metrics)
        predictions_file_name = f"{training_args.output_dir}/{args.dataset}_{key}_predictions.npy"
        print("Saving predictions for ", key, " at ", predictions_file_name)
        np.save(predictions_file_name, output.predictions)


if __name__ == "__main__":
    parser = HfArgumentParser((Args, RewardFunctionTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    predict_reward_model(args, training_args)
