"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import os
from abc import ABC, abstractmethod

import hydra
import torch
from models.model_creation import construct_model_from_class
from rouge import Rouge
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation.ni_metrics import (bleu_score, metric_max_over_ground_truths,
                                   rouge1_score)


class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, batch: dict, return_tensor: bool) -> Tensor:
        """Returns the batch of rewards for a batch of input-output pairs.

        Inputs will come with prompt as well as information for calculating the reward.
        """
        pass


class BLEURewardFunction(RewardFunction):
    def __call__(self, batch: dict, return_tensor: bool) -> Tensor:
        """Returns the batch of rewards for a batch of input-output pairs."""
        score = [
            metric_max_over_ground_truths(bleu_score, response, labels)
            for response, labels in zip(batch["responses"], batch["labels"])
        ]
        if return_tensor:
            return torch.tensor(score, device=batch["response_ids"].device)
        return score


class RougeRewardFunction(RewardFunction):
    def __call__(self, batch: dict, return_tensor: bool) -> Tensor:
        """Returns the batch of rewards for a batch of input-output pairs."""
        score = [
            metric_max_over_ground_truths(rouge1_score, response, labels)
            for response, labels in zip(batch["responses"], batch["labels"])
        ]
        if return_tensor:
            return torch.tensor(score, device=batch["response_ids"].device)
        return score


class TokenIDRewardFunction(RewardFunction):
    def __init__(self, threshold: int, eos_token_id: int = 50256) -> None:
        self.threshold = threshold
        self.eos_token_id = eos_token_id

    def __call__(self, batch: dict, return_tensor: bool) -> Tensor:
        """Calculate percentage of tokens in batch["response_ids"] are over a certain threshold."""
        res = (
            torch.count_nonzero(
                (self.eos_token_id != batch["response_ids"]) * (batch["response_ids"] > self.threshold), dim=1
            )
            / batch["response_ids"].shape[1]
        )
        if return_tensor:
            return res
        return res.cpu().numpy()


class SentimentRewardFunction(RewardFunction):
    def __init__(
        self, sentiment_model_dir: str, device: torch.device, positive_sentiment: bool = True
    ) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_dir)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(sentiment_model_dir)
        self.tokenizer.truncation_side = "left"
        self.sentiment_index = 1 if positive_sentiment else 0
        self.model.eval()

    def __call__(self, batch: dict, return_tensor: bool) -> Tensor:
        """Calculate sentiment of the responses in the batch using self.sentiment_model."""
        with torch.no_grad():
            batch_sentiment = self.model(
                **self.tokenizer(
                    [i + r for i, r in zip(batch["queries"], batch["responses"])],
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                ).to(self.model.device)
            )
            batch_sentiment = torch.nn.functional.softmax(batch_sentiment.logits, dim=-1)[
                :, self.sentiment_index
            ]
        if return_tensor:
            return batch_sentiment
        return batch_sentiment.cpu().numpy()


class SummarisationRewardFunction(RewardFunction):
    def __init__(self, rf_model_dir: str, device: torch.device, args) -> None:
        self.args = args
        self.model, self.tokenizer, _ = construct_model_from_class(args)
        self.device = device
        self.tokenizer.truncation_side = "left"
        self.model.eval()
        self.eos_token = self.tokenizer.eos_token

    def __call__(self, batch: dict, return_tensor: bool) -> Tensor:
        """Calculate sentiment of the responses in the batch using self.sentiment_model."""
        with torch.no_grad():
            rewards = self.model.calculate_reward(
                **self.tokenizer(
                    [i + r + self.eos_token for i, r in zip(batch["queries"], batch["responses"])],
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                    return_token_type_ids=False,
                ).to(self.device)
            )
        if return_tensor:
            return rewards
        return rewards.cpu().numpy()


def make_reward_function(reward_function_str: str, args) -> RewardFunction:
    try:
        original_cwd = hydra.utils.get_original_cwd()
    except ValueError:
        original_cwd = os.getcwd()
    rf_model_dir = os.path.join(original_cwd, args.rf_model_dir)
    if not os.path.exists(rf_model_dir):
        # If we're passing path to model on HF hub
        rf_model_dir = args.rf_model_dir
    device = getattr(args, "rf_device", getattr(args, "device"))
    if reward_function_str == "bleu":
        return BLEURewardFunction()
    elif reward_function_str == "rouge":
        return RougeRewardFunction()
    elif reward_function_str == "token_id":
        return TokenIDRewardFunction(threshold=args.token_id_rf_threshold)
    elif reward_function_str == "sentiment":
        return SentimentRewardFunction(rf_model_dir, device, args.desired_sentiment)
    elif reward_function_str == "summarisation":
        new_args = copy.deepcopy(args)
        new_args.freeze_layers = 1
        new_args.value_head_activation = False
        new_args.parallelize = False
        new_args.model_name = rf_model_dir
        new_args.rl_training = None
        new_args.model_name = args.rf_model_dir
        new_args.policy_head_device = device
        new_args.policy_split_percentage = args.rm_split_percentage
        return SummarisationRewardFunction(rf_model_dir, device, new_args)
    else:
        raise ValueError(f"Unknown reward function: {reward_function_str}")
