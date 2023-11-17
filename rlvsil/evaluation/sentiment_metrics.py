"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from transformers import PreTrainedTokenizer

import wandb


def compute_all_sentiment_metrics(
    predictions: List[str],
    inputs: List[str],
    token_predictions: List[int],
    sentiments: List[int],
    references: List[str],
    tokenizer: PreTrainedTokenizer,
    rewards: Optional[List[float]] = None,
) -> Dict[str, Any]:
    result = {}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in token_predictions]
    response_data = list(
        map(
            list,  # type:ignore
            zip(
                inputs,
                predictions,
                sentiments,
                ([None] * len(predictions)) if rewards is not None else rewards,  # type:ignore
            ),
        )
    )
    text_table = wandb.Table(columns=["input", "response", "reference", "reward"], data=response_data)
    result.update({"text_table": text_table})
    result.update(
        dict(
            average_prediction_length=np.mean(prediction_lens),
            median_prediction_length=np.median(prediction_lens),
            prediction_length_distribution=prediction_lens,
        )
    )
    if rewards is not None:
        result.update(
            average_reward=np.mean(rewards),
            median_reward=np.median(rewards),
            reward_distribution=np.array(rewards),
        )
    return result
