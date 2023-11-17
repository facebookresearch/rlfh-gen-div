"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, Dict, List, Union, Optional

import numpy as np
import wandb
from transformers import PreTrainedTokenizer

from evaluation.ni_metrics import compute_metrics

TEXT_TABLE_KEY = "text_table"


def compute_all_summarisation_metrics(
    inputs: List[str],
    predictions: List[str],
    references: List[str],
    token_predictions: List[int],
    tokenizer: PreTrainedTokenizer,
    return_text_table: bool = True,
    return_distributions: bool = True,
    text_table_extra_id: Optional[int] = None,
    **extras: Union[List[float], List[int], np.ndarray],
) -> Dict[str, Any]:
    result = {}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in token_predictions]
    result.update(compute_metrics(predictions, [[r] for r in references]))
    result.update(
        dict(
            average_prediction_length=np.mean(prediction_lens),
            median_prediction_length=np.median(prediction_lens),
        )
    )
    if return_distributions:
        result.update(prediction_length_distribution=prediction_lens)

    if return_text_table:
        columns = ["input", "response", "reference"]
        input_data: list = [inputs, predictions, references]
        for k, v in extras.items():
            columns.append(k)
            input_data.append(v)

        response_data = list(map(list, zip(*input_data)))
        text_table = wandb.Table(columns=columns, data=response_data)
        key = TEXT_TABLE_KEY if text_table_extra_id is None else TEXT_TABLE_KEY + f"_{text_table_extra_id}"
        result.update({key: text_table})

    for k, v in extras.items():
        result.update({f"average_{k}": np.mean(v)})
        result.update({f"median_{k}": np.median(v)})
        if return_distributions:
            result.update({f"{k}_distribution": np.array(v)})
    return result


def compute_summarisation_metrics(
    inputs,
    labels,
    preds,
    tokenizer,
    reward_function,
    text_table_extra_id=None,
    return_text_table=True,
    return_distributions=True,
):
    inputs = inputs.copy()
    np.place(inputs, inputs == -100, tokenizer.pad_token_id)
    labels = labels.copy()
    np.place(labels, labels == -100, tokenizer.pad_token_id)

    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    kwargs = {}
    if reward_function is not None:
        rewards = []
        for decoded_input, decoded_pred in zip(decoded_inputs, decoded_preds):
            reward = reward_function(
                {"queries": [decoded_input], "responses": [decoded_pred]}, return_tensor=False
            )
            rewards.append(reward)
        kwargs["rewards"] = rewards

    return compute_all_summarisation_metrics(
        inputs=decoded_inputs,
        predictions=decoded_preds,
        references=decoded_labels,
        token_predictions=preds,
        tokenizer=tokenizer,
        return_text_table=return_text_table,
        return_distributions=return_distributions,
        text_table_extra_id=text_table_extra_id,
        **kwargs,
    )
