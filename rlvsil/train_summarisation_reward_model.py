"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_metric
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

import wandb
from dataset.summarisation_feedback_dataloader import (
    DataCollatorWithPaddingTwoInputs, make_dataset)
from dataset.summarisation_formatting import SUBREDDITS
from lab.rf_trainer import (RewardFunctionTrainer,
                            RewardFunctionTrainingArguments)
from models.model_creation import construct_model_from_class
from utils.core import normalise_seed


@dataclass
class Args:
    training: bool = field(default=True)
    evaluate: bool = field(default=True)
    push_to_hub_final: bool = field(default=False)
    calculate_normalisation: bool = field(default=True)
    normalisation_checkpoint_dir: Optional[str] = field(default="final-normalised")
    value_head_activation: bool = field(default=False)
    parallelize: bool = field(default=False)
    wandb_run_name: Optional[str] = field(default=None)
    wandb_tags: Optional[str] = field(default=None)
    wandb_group: Optional[str] = field(default=None)
    wandb_project: Optional[str] = field(default=None)
    wandb_entity: Optional[str] = field(default="robkirk")
    run_id: Optional[str] = field(default=None)
    model_name: Optional[str] = field(default="gpt2")
    eval_subsample_num: Optional[int] = field(default=None)
    freeze_layers: Optional[float] = field(default=0.8)
    freeze_lm_head: bool = field(default=True)
    value_normalisation: float = field(default=0.0)
    value_normalisation_std: float = field(default=1.0)
    dataset_random_subset: Optional[int] = field(default=None, metadata={"choices": [10, 50]})
    dataset_structured_subset: Optional[str] = field(
        default=None, metadata={"choices": ["length", "sentiment"] + list(SUBREDDITS)}
    )
    eval_dataset: Optional[str] = field(default=None)
    target_eval_datapoints: int = field(default=500)
    save_preds_to_wandb: bool = field(default=False)
    bettertransformer: bool = field(default=False)
    torchcompile: bool = field(default=False)


def train_reward_model(args: Args, training_args: RewardFunctionTrainingArguments):
    job_id = int(os.environ.get("SLURM_ARRAY_JOB_ID", 0))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    config = dict(**asdict(args), **training_args.to_sanitized_dict())
    config["job_id"] = job_id
    config["task_id"] = task_id

    training_args.seed = normalise_seed(training_args.seed)

    if not args.run_id:
        if job_id and task_id:
            run_id = f"{job_id}-{task_id}"
        else:
            run_id = wandb.util.generate_id()
    else:
        run_id = args.run_id

    resumed = False

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

    training_args.output_dir = f"/checkpoint/{os.environ['USER']}/rlvsil/{args.wandb_project}/{run_id}"
    training_args.overwrite_output_dir = True
    training_args.evaluation_strategy = "steps"
    training_args.save_total_limit = training_args.save_total_limit or 5
    training_args.report_to = ["wandb"]
    training_args.lr_scheduler_type = "cosine"
    training_args.remove_unused_columns = True

    if os.path.isdir(training_args.output_dir) and get_last_checkpoint(training_args.output_dir) is not None:
        resumed = True
        training_args.resume_from_checkpoint = True

    model, tokenizer, _ = construct_model_from_class(args)

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    ds = make_dataset(tokenizer, args)

    eval_result_file_name = "eval_results_train.csv"

    def compute_metrics(eval_preds):
        acc_metric = load_metric("accuracy")
        precision_metric = load_metric("precision")
        recall_metric = load_metric("recall")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        if args.save_preds_to_wandb:
            # Create file in output_dir with predictions, and save to wandb
            file_name = os.path.join(training_args.output_dir, eval_result_file_name)
            with open(file_name, "w") as writer:
                writer.write("Index,Pred,Label\n")
                for i, pred, label in zip(range(len(predictions)), predictions, labels):
                    writer.write(f"{i},{pred},{label}\n")
            wandb.save(file_name)

        return dict(
            logits_mean=np.mean(logits),
            logits_var=np.var(logits),
            logits_std=np.std(logits),
            logits_median=np.median(logits),
            logits_ptp=np.ptp(logits),
            **acc_metric.compute(predictions=predictions, references=labels),
            **precision_metric.compute(predictions=predictions, references=labels),
            **recall_metric.compute(predictions=predictions, references=labels),
        )

    data_collator = DataCollatorWithPaddingTwoInputs(tokenizer)
    trainer = RewardFunctionTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=(
            ds["validation"].select(range(args.eval_subsample_num))
            if args.eval_subsample_num
            else ds["validation"]
        ),
        compute_metrics=compute_metrics,
    )

    if args.training:
        trainer.train(resume_from_checkpoint=resumed)
        trainer.save_model(f"{training_args.output_dir}/final/")

    if args.evaluate or args.calculate_normalisation:
        target_batches = math.ceil(args.target_eval_datapoints / trainer.args.eval_batch_size)
        trainer.args.validation_evaluation_batches = target_batches
        trainer.args.training_evaluation_batches = target_batches
        metrics = trainer.evaluate()

    if args.calculate_normalisation:
        normalisation_factor = -(
            metrics.get(
                "_logits_mean",
                metrics.get("logits_mean", metrics.get("train/logits_mean", metrics.get("eval/logits_mean"))),
            )
        )
        normalisation_std = metrics.get(
            "_logits_std",
            metrics.get("logits_std", metrics.get("train/logits_std", metrics.get("eval/logits_std"))),
        )

        model.v_head.normalisation.data = torch.tensor(
            normalisation_factor, requires_grad=False, device=model.v_head.normalisation.device
        )
        model.v_head.normalisation_std.data = torch.tensor(
            normalisation_std, requires_grad=False, device=model.v_head.normalisation_std.device
        )

        if not args.evaluate:
            model.to("cpu")

        print("Normalisation factor", normalisation_factor)
        print("Normalisation std", normalisation_std)

        trainer.save_model(f"{training_args.output_dir}/{args.normalisation_checkpoint_dir}/")
        if args.push_to_hub_final:
            trainer.model.push_to_hub(f"UCL-DARK/{run_id}", private=True, use_auth_token=True)
            trainer.tokenizer.push_to_hub(f"UCL-DARK/{run_id}", private=True, use_auth_token=True)

    if args.evaluate:
        eval_result_file_name = "eval_results_test.csv"
        trainer.evaluate(ds["test"], metric_key_prefix="test", evaluation_batches=target_batches)
        if args.dataset_structured_subset:
            for key in {"full_validation", "ood_validation", "full_test", "ood_test"}:
                eval_result_file_name = f"eval_results_{key}.csv"
                trainer.evaluate(ds[key], metric_key_prefix=key, evaluation_batches=target_batches)


if __name__ == "__main__":
    parser = HfArgumentParser((Args, RewardFunctionTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    train_reward_model(args, training_args)
