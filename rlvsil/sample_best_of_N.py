"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%m-%d-%Y %H:%M:%S"
)


def get_run_id(args, job_id, task_id):
    if not args.run_id:
        if job_id and task_id:
            run_id = f"{job_id}-{task_id}"
        else:
            run_id = wandb.util.generate_id()
    else:
        run_id = args.run_id

    return run_id


def setup_wandb(args, run_id, config):
    if training_args.process_index == 0:
        wandb_run = wandb.init(
            id=run_id,
            resume=args.resume,
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=args.wandb_tags.split(",") if args.wandb_tags else None,
            group=args.wandb_group,
        )

        if wandb.run.resumed:
            for k, v in wandb.config.as_dict().items():
                if k not in {"do_train", "do_eval"}:
                    setattr(args, k, v)
        else:
            logging.info("Initializing run")
            wandb.config.update(config)
    return wandb_run


def setup_models(args):
    device_0 = args.device if torch.cuda.is_available() else torch.device("cpu")
    model, tokenizer, model_ref = construct_model_from_class(args)

    if not args.parallelize:
        model.to(device_0)

    if args.eval_rl_model:  # false for SL
        rl_checkpoint = torch.load(args.eval_rl_model_checkpoint_dir)
        model.load_state_dict(rl_checkpoint)

    reward_function = make_reward_function(args.reward_function, args)

    if args.tie_frozen_layers:
        tie_frozen_layers(model, reward_function.model)

    if args.parallelize:
        device_map = construct_device_map(
            torch.device(args.device).index,
            torch.device(args.rf_device).index,
            reward_function.model,
            args.rm_split_percentage,
        )
        reward_function.model.parallelize(device_map)
        logging.info("RM parallelized to %s", device_map)
        reward_function.device = args.device  # This is the device inputs are sent to
    else:
        reward_function.model = reward_function.model.to(device_0)

    return model, tokenizer, reward_function


def train(args: Args, training_args: EvalGenerationTrainingArguments):
    disable_caching()

    training_args.seed = normalise_seed(training_args.seed)
    set_seed(training_args.seed)

    # Get $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID from environment variables
    job_id = int(os.environ.get("SLURM_ARRAY_JOB_ID", 0))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    config = dict(**vars(args), **vars(training_args))
    config["job_id"] = job_id
    config["task_id"] = task_id
    run_id = get_run_id(args, job_id=job_id, task_id=task_id)

    model, tokenizer, reward_function = setup_models(args)
    raw_dataset = make_summarisation_dataset(args, tokenizer)
    collate_fn = make_summarisation_collate_fn(args, tokenizer)
    metric_fn = partial(compute_summarisation_metrics, tokenizer=tokenizer, reward_function=reward_function)

    wandb_run = setup_wandb(args, run_id, config=config)
    if args.wandb_watch:
        wandb.watch(model)
    logging.info("Wandb is initialized.")
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
    training_args.do_sample = True

    training_args._setup_devices
    training_args._n_gpu = 1

    if os.path.isdir(training_args.output_dir) and get_last_checkpoint(training_args.output_dir) is not None:
        args.resumed = True
        training_args.resume_from_checkpoint = True
        logging.info("SL Model Checkpoint Loaded!")

    trainer_cls = EvalGenerationTrainer

    # set CUDA_VISIBLE_DEVICES to only first device, and then reset after training initialisation
    # this is to avoid OOM errors when initialising the trainer
    prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if prev_cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices.split(",")[0]

    trainer = trainer_cls(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset["validation"],
        data_collator=collate_fn,
        compute_metrics=metric_fn,
    )

    if prev_cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible_devices

    target_batches = math.ceil(args.target_eval_datapoints / trainer.args.eval_batch_size)
    trainer.args.validation_evaluation_batches = target_batches
    trainer.args.training_evaluation_batches = target_batches

    def do_dataset_evaluation(metric_key_prefix, eval_dataset):
        """Do evaluation on one dataset and do BoN for that as well."""
        with torch.no_grad():
            results = []
            logging.info("Starting evaluation for dataset: %s", metric_key_prefix)
            for i in range(training_args.num_return_sequences):
                res = trainer.evaluate(
                    eval_dataset,
                    metric_key_prefix=metric_key_prefix,
                    eval_batches=target_batches,
                    text_table_extra_id=i,
                )

                results.append(res)

        tables = [
            results[i][f"{metric_key_prefix}_text_table_{i}"].data
            for i in range(training_args.num_return_sequences)
        ]
        bon = []
        for i in range(len(tables[0])):
            candidate_row = tables[0][i]
            for j in range(0, training_args.num_return_sequences):
                other_row = tables[j][i]
                if other_row[-1][0] > candidate_row[-1][0]:  # row[-1][0] corresponds to reward
                    candidate_row = other_row
            bon.append(candidate_row)

        text_table = wandb.Table(columns=["input", "response", "reference", "reward"], data=bon)
        if wandb_run:
            wandb_run.log({f"{metric_key_prefix}/best-of-N": text_table})

    datasets = [raw_dataset["test"], raw_dataset["validation"], raw_dataset["train"]]
    keys = ["test", "validation", "train"]
    if args.dataset_structured_subset:
        for key in ("full_validation", "ood_validation", "full_test", "ood_test"):
            if key in raw_dataset:
                datasets.append(raw_dataset[key])
                keys.append(key)

    evaluation_splits = (
        training_args.evaluation_splits.split(",") if training_args.evaluation_splits else None
    )
    for key, dataset in zip(keys, datasets):
        if (evaluation_splits and key in evaluation_splits) or (not evaluation_splits):
            do_dataset_evaluation(key, dataset)


if __name__ == "__main__":
    parser = HfArgumentParser((Args, EvalGenerationTrainingArguments))
    args, training_args, unk_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    args.training = training_args.do_train  # args.training is required to construct model from class.
    train(args, training_args)
