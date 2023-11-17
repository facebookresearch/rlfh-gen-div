"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import os
from functools import partial

import torch
from datasets import disable_caching
from transformers import HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint

import wandb
from dataset.summarisation_dataloader import (make_summarisation_collate_fn,
                                              make_summarisation_dataset)
from evaluation.reward_functions import make_reward_function
from evaluation.summarisation_metrics import compute_summarisation_metrics
from lab.args import Args
from lab.sl_trainer import (EvalGenerationTrainer,
                            EvalGenerationTrainingArguments)
from models.model_creation import (construct_device_map,
                                   construct_model_from_class,
                                   tie_frozen_layers)
from utils.core import normalise_seed
from utils.hardware import get_device


def setup_wandb(args: Args, training_args: EvalGenerationTrainingArguments) -> str:
    # Get $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID from environment variables
    job_id = int(os.environ.get("SLURM_ARRAY_JOB_ID", 0))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    config = dict(**vars(args), **vars(training_args))
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
            resume=args.resume,
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=args.wandb_tags.split(",") if args.wandb_tags else None,
            group=args.wandb_group,
        )

        if wandb.run.resumed:
            print("Resuming run")
            for k, v in wandb.config.as_dict().items():
                if k not in {"do_train", "do_eval"}:
                    setattr(args, k, v)
        else:
            print("Initializing run")
            wandb.config.update(config)

    return run_id


def train(args: Args, training_args: EvalGenerationTrainingArguments):
    disable_caching()
    device = get_device(args.device_idx)

    training_args.seed = normalise_seed(training_args.seed)
    set_seed(training_args.seed)

    run_id = setup_wandb(args, training_args)

    args.device = device

    model, tokenizer, model_ref = construct_model_from_class(args)
    model.to(device)
    if args.eval_rl_model:
        rl_checkpoint = torch.load(args.eval_rl_model_checkpoint_dir)
        model.load_state_dict(rl_checkpoint["learner_state"]["model"])

    if model_ref:
        model_ref.to(device)

    if args.wandb_watch:
        wandb.watch(model)

    trainer_cls = EvalGenerationTrainer

    reward_function = None
    if args.eval_sl_reward_function:
        reward_function = make_reward_function(args.reward_function, args)
        device_map = construct_device_map(
            args.device_idx, args.rf_device_idx, reward_function.model, args.freeze_layers
        )
        reward_function.model.to(args.rf_device_idx)
        reward_function.parallelize(device_map)
        tie_frozen_layers(model, reward_function.model)
    raw_dataset = make_summarisation_dataset(args, tokenizer)
    collate_fn = make_summarisation_collate_fn(args, tokenizer)
    metric_fn = partial(compute_summarisation_metrics, tokenizer=tokenizer, reward_function=reward_function)

    if training_args.overwrite_output_dir:
        training_args.output_dir = training_args.overwrite_output_dir + f"/{run_id}"
    else:
        training_args.output_dir = f"/checkpoint/{os.environ['USER']}/rlvsil/{args.wandb_project}/{run_id}"
    training_args.overwrite_output_dir = True
    training_args.report_to = ["wandb"]
    training_args.save_total_limit = training_args.save_total_limit or 5
    training_args.lr_scheduler_type = "cosine"
    training_args.predict_with_generate = True
    training_args.generation_num_beams = 1
    training_args.generation_pad_token_id = tokenizer.pad_token_id
    training_args.remove_unused_columns = False
    training_args.evaluation_strategy = "steps"

    if os.path.isdir(training_args.output_dir) and get_last_checkpoint(training_args.output_dir) is not None:
        args.resumed = True
        training_args.resume_from_checkpoint = True
    else:
        args.resumed = False
        training_args.resume_from_checkpoint = False

    trainer = trainer_cls(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset["validation"],
        data_collator=collate_fn,
        compute_metrics=metric_fn,
    )
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=args.resumed)
        trainer.save_model(f"{training_args.output_dir}/final/")

    if training_args.do_eval:
        target_batches = math.ceil(args.target_eval_datapoints / trainer.args.eval_batch_size)
        trainer.args.validation_evaluation_batches = target_batches
        trainer.args.training_evaluation_batches = target_batches

        if training_args.evaluation_splits:
            splits = training_args.evaluation_splits.split(",")
        else:
            splits = ["train", "validation", "test"]
            if args.dataset_structured_subset:
                splits.extend(["full_validation", "ood_validation", "full_test", "ood_test"])

        for split in splits:
            trainer.evaluate(raw_dataset[split], metric_key_prefix=split, eval_batches=target_batches)


if __name__ == "__main__":
    parser = HfArgumentParser((Args, EvalGenerationTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    args.training = training_args.do_train
    train(args, training_args)
