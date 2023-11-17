"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from dataclasses import dataclass, field
from typing import Optional

from dataset.summarisation_formatting import SUBREDDITS


@dataclass
class Args:
    """The above argsparse parser, but as a dataclass with fields instead."""

    wandb_run_name: Optional[str] = field(default=None)
    wandb_project: Optional[str] = field(default=None)
    wandb_group: Optional[str] = field(default=None)
    wandb_entity: str = field(default="robkirk")
    wandb_watch: bool = field(default=False)
    wandb_tags: Optional[str] = field(default=None)
    resume: str = field(default="allow", metadata={"choices": ["allow", "must", "never"]})
    run_id: Optional[str] = field(default=None)
    device_idx: int = field(default=0)
    device: str = field(default="cuda:0")
    model_name: str = field(default="gpt2")
    base_model_name: str = field(default="")

    freeze_layers: float = field(default=0.8)
    freeze_lm_head: bool = field(default=True)
    tie_frozen_layers: bool = field(default=False)
    parallelize: bool = field(default=False)
    bettertransformer: bool = field(default=False)
    torchcompile: bool = field(default=False)
    value_head_activation: bool = field(default=False)
    value_normalisation: float = field(default=0.0)
    value_normalisation_std: float = field(default=1.0)
    rm_split_percentage: float = field(default=1.0)
    policy_split_percentage: float = field(default=1.0)
    max_new_tokens: int = field(default=100)
    rl_training: bool = field(default=False)
    summarisation_dataset_queries: bool = field(default=False)
    dataset_random_subset: Optional[int] = field(default=None, metadata={"choices": [10, 50]})
    dataset_structured_subset: Optional[str] = field(
        default=None, metadata={"choices": ["length", "sentiment"] + list(SUBREDDITS)}
    )
    eval_dataset: Optional[str] = field(
        default=None, metadata={"choices": ["cnndm"]}
    )

    max_source_length: int = field(default=1024)
    max_target_length: int = field(default=1024)
    dataset: str = field(default="summarisation")
    rf_model_dir: str = field(default="checkpoints/hf/sentiment/")
    reward_function: str = field(
        default="summarisation",
        metadata={"choices": ["rouge", "bleu", "token_id", "sentiment", "summarisation"]},
    )
    eval_sl_reward_function: bool = field(default=False)
    rf_device_idx: int = field(default=0)
    rf_device: str = field(default="cuda:0")
    policy_head_device: str = field(default="cuda:0")
    target_eval_datapoints: int = field(default=500)

    eval_rl_model: bool = field(default=False)
    eval_rl_model_checkpoint_dir: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.base_model_name == "":
            self.base_model_name = self.model_name
