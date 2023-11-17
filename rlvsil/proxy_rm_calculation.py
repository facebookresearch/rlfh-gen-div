"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""Script to calculate rewards for generated text with a proxy RM."""
import torch
import json
import os
from dataclasses import asdict, field, dataclass

from models.model_creation import construct_device_map
import numpy as np
import pandas as pd
from evaluation.reward_functions import make_reward_function
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

import wandb
from dataset.summarisation_feedback_dataloader import (
    DataCollatorWithPaddingTwoInputs, make_dataset)
from dataset.summarisation_formatting import SUBREDDITS
from lab.rf_trainer import (RewardFunctionTrainer,
                            RewardFunctionTrainingArguments)
from train_summarisation_reward_model import Args as RMArgs
from utils.core import normalise_seed


@dataclass
class Args(RMArgs):
    analyse_run_id: str = field(default=None)
    infer_batch_size: int = field(default=8)
    table_names: str = field(default="testbestofN")
    rf_model_dir: str = field(default=None)
    device: str = field(default="cuda:0")
    rf_device: str = field(default="cuda:0")
    rm_split_percentage: float = field(default=0)
    parallelize: bool = field(default=False)


def get_wandb_run_dataset(run_id: str, table_name: str, path: str = "ucl-dark/rlvsil-main") -> pd.DataFrame:
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"

    api = wandb.Api()

    artifact = api.artifact(f"{path}/run-{run_id}-{table_name}:latest", type="run_table")

    # Check whether artiface.file() already exists
    if not os.path.exists(artifact.file()):
        artifact.download()
    else:
        print(f"File for run {run_id} already exists.")

    with open(artifact.file()) as f:
        res = json.load(f)

    df = pd.DataFrame(res["data"], columns=res["columns"])
    return df


def art_name(run_id, table_name):
    return f"run-{run_id}-{table_name}bestofN"


def main(args):
    # Initialise
    job_id = int(os.environ.get("SLURM_ARRAY_JOB_ID", 0))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    config = asdict(args)
    config["job_id"] = job_id
    config["task_id"] = task_id

    if not args.run_id:
        if job_id and task_id:
            run_id = f"{job_id}-{task_id}"
        else:
            run_id = wandb.util.generate_id()
    else:
        run_id = args.run_id

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

    print("Initializing run")

    # Make reward function
    reward_function = make_reward_function("summarisation", args)
    if args.parallelize:
        device_map = construct_device_map(
            torch.device(args.device).index,
            torch.device(args.rf_device).index,
            reward_function.model,
            args.rm_split_percentage,
        )
        reward_function.model.parallelize(device_map)
        print("RM parallelized to ", device_map)
        reward_function.device = args.device  # This is the device inputs are sent to
    else:
        # Llama doesn't support .parallelize()
        reward_function.device = args.device
        reward_function.model.to(reward_function.device)

    tokenizer = reward_function.tokenizer
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        reward_function.model.config.pad_token_id = tokenizer.pad_token_id

    # Create Dataset from wandb
    for table_name in args.table_names.split(","):
        df = get_wandb_run_dataset(args.analyse_run_id, table_name)
        df.rename(columns={"input": "queries", "response": "responses"}, inplace=True)

        # Do inference on dataset
        rewards = []
        for i in range(0, len(df), args.infer_batch_size):
            batch = df.iloc[i : i + args.infer_batch_size]
            rewards.append(reward_function(batch, return_tensor=False))
        rewards = np.concatenate(rewards)

        # Construct new table and log results
        df["proxy_rewards"] = rewards
        average_rewards = np.mean(rewards)
        var_rewards = np.var(rewards)
        std_rewards = np.std(rewards)
        table = wandb.Table(dataframe=df)
        wandb.log(
            {
                f"{table_name}/proxy_rewards": table,
                f"{table_name}/average_proxy_reward": average_rewards,
                f"{table_name}/var_proxy_reward": var_rewards,
                f"{table_name}/std_proxy_reward": std_rewards,
            }
        )


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
