"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

# Code from: https://github.com/lvwerra/trl/blob/4fe9988eb8adf0227c26432f8eb3e57a66556350/trl/ppo.py#L108
# Licence https://github.com/huggingface/trl/blob/4fe9988eb8adf0227c26432f8eb3e57a66556350/LICENSE

import math
import random
import time
import typing as t
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from utils.core import (
    WANDB_PADDING,
    add_suffix,
    average_torch_dicts,
    clip_by_value,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    stack_dicts,
    stats_to_np,
    whiten,
)


class KLController(ABC):
    def __init__(self, value):
        self.value = value

    @abstractmethod
    def update(self, current, n_steps):
        pass


class AdaptiveKLController(KLController):
    """
    Adaptive KL controller described in the paper:
    Fine-Tuning Language Models from Human Preferences (OpenAI)
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

    def state_dict(self):
        return {"value": self.value, "target": self.target, "horizon": self.horizon}

    def load_state_dict(self, state_dict):
        self.value = state_dict["value"]
        self.target = state_dict["target"]
        self.horizon = state_dict["horizon"]


class FixedKLController(KLController):
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state_dict):
        self.value = state_dict["value"]


class AutoRegressivePPOTrainer:
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """

    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef": 0.2,
        "target": 6,
        "horizon": 10000,
        "gamma": 1,
        "lam": 0.95,
        "cliprange": 0.2,
        "cliprange_value": 0.2,
        "vf_coef": 0.1,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
        "minibatch_size": 32,
        "temperature": 1.0,
        "kl_approx": 0,
    }

    def __init__(self, model, ref_model, tokenizer, args):
        """
        Initialize PPOTrainer.
        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            tokenizer (tokenizer): Hugging Face tokenizer
            args (dict or None): PPO parameters for training. Can include following keys:
              'lr' (float): Adam learning rate, default: 1.41e-5
              'batch_size' (int): Number of samples per optimisation step, default: 256
              'forward_batch_size' (int): Number of samples forward passed through model at a time,
                     default: 16
              'minibatch_size' (int): Number of samples per minibatch, default: 32
              'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
              'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
              'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
              'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
              'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
              'vf_coef' (float): Scaling factor for value loss, default: 0.1
              'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
              'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control),
                     default: 0.2
              'target' (float): Target KL value for adaptive KL control, default: 6.0
              'horizon' (float): Horizon for adaptive KL control, default: 10000
        """
        self.init_ppo_params(args)

        self.ref_model = ref_model
        self.model = model
        self.tokenizer = tokenizer

        self.optimizer = Adam(model.parameters(), lr=self.ppo_params["lr"])

    def init_ppo_params(self, args) -> None:
        self.ppo_params: dict = self.default_params
        for k, v in self.ppo_params.items():
            if getattr(args, k, None) is not None:
                self.ppo_params[k] = getattr(args, k)

        if self.ppo_params["adap_kl_ctrl"]:
            self.kl_ctl: KLController = AdaptiveKLController(
                self.ppo_params["init_kl_coef"], self.ppo_params["target"], self.ppo_params["horizon"]
            )
        else:
            self.kl_ctl = FixedKLController(self.ppo_params["init_kl_coef"])

        if not (
            self.ppo_params["forward_batch_size"] <= self.ppo_params["batch_size"]
            and self.ppo_params["batch_size"] % self.ppo_params["forward_batch_size"] == 0
        ):
            print("forward_batch_size must be <= batch_size and batch_size must be divisible by FBS")
            self.ppo_params["forward_batch_size"] = self.ppo_params["batch_size"]
            print("Adjusting forward_batch_size to be equal to batch_size")

        if not (
            self.ppo_params["minibatch_size"] <= self.ppo_params["batch_size"]
            and self.ppo_params["batch_size"] % self.ppo_params["minibatch_size"] == 0
        ):
            print("minibatch_size must be < batch_size and batch_size must be divisible by MBS")
            # maximum divisor of batch_size to make minibatch_size
            bs_divisor = 2 ** min(math.floor(np.log2(self.ppo_params["batch_size"])), 3)
            self.ppo_params["minibatch_size"] = int(self.ppo_params["batch_size"] / bs_divisor)
            print(
                f"Adjusting minibatch_size to be equal to {self.ppo_params['minibatch_size']}"
                f" = {self.ppo_params['batch_size']} / {bs_divisor}"
            )

    def step(self, queries: Tensor, responses: Tensor, scores: Tensor, attention_mask: Tensor) -> dict:
        """
        Run a PPO optimisation step.
        args:
            queries: batch of tensors containing the encoded queries, shape [bs, query_length]
            responses: batch of tensors containing the encoded responses, shape [bs, response_length]
            scores: batch containing the scores, shape [bs]
        returns:
            train_stats (dict): a summary of the training statistics
        """

        bs = self.ppo_params["batch_size"]
        mbs = self.ppo_params["minibatch_size"]
        assert bs == len(queries), f"Batch size ({bs}) does not match number of examples ({len(queries)})"

        timing = dict()
        t0 = time.time()

        t = time.time()
        logprobs, ref_logprobs, values = self.proximal_forward_pass(queries, responses, attention_mask)
        timing["time/ppo/forward_pass"] = time.time() - t

        t = time.time()
        rewards, non_score_reward = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing["time/ppo/compute_rewards"] = time.time() - t

        t = time.time()
        all_stats = []
        idxs = list(range(bs))

        for _ in range(self.ppo_params["ppo_epochs"]):
            random.shuffle(idxs)
            for i in range(int(bs / mbs)):
                idx = idxs[i : i + mbs]
                train_stats = self.train_minibatch(
                    logprobs[idx],
                    values[idx],
                    rewards[idx],
                    queries[idx],
                    responses[idx],
                    torch.cat([queries[idx], responses[idx]], dim=1),
                    attention_mask[idx],
                )
                all_stats.append(train_stats)
        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        for key_to_flatten in ("policy/advantages", "policy/ratio", "returns/dist"):
            train_stats[key_to_flatten] = torch.nan_to_num(
                torch.flatten(train_stats[key_to_flatten]).unsqueeze(0), WANDB_PADDING
            )

        stats = self.record_step_stats(
            scores=scores,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
        )
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t

        self.kl_ctl.update(stats["objective/kl"], self.ppo_params["batch_size"])

        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)
        return stats

    def proximal_forward_pass(
        self, queries: Tensor, responses: Tensor, attention_mask: Tensor
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate model outputs for proximal policy."""
        input_ids = torch.cat([queries, responses], dim=1)
        logprobs_acc, ref_logprobs_acc, values_acc = [], [], []
        gen_len = responses.shape[1]
        bs, fbs = self.ppo_params["batch_size"], self.ppo_params["forward_batch_size"]
        with torch.no_grad():
            for i in range(0, bs, fbs):
                logprobs, v, ref_logprobs, _ = self.get_logprobs_vpreds(
                    input_ids[i : i + fbs], attention_mask[i : i + fbs], gen_len, get_ref_logprobs=True
                )
                logprobs_acc.append(logprobs)
                ref_logprobs_acc.append(ref_logprobs)
                values_acc.append(v)
        return (
            torch.cat(logprobs_acc, dim=0),
            torch.cat(ref_logprobs_acc, dim=0),
            torch.cat(values_acc, dim=0),
        )

    def train_minibatch(self, logprobs, values, rewards, query, response, input_ids, attention_mask):
        """Train one PPO minibatch"""
        loss_p, loss_v, train_stats = self.loss(
            logprobs, values, rewards, query, response, input_ids, attention_mask
        )
        loss = loss_p + loss_v
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return train_stats

    def get_logprobs_vpreds(
        self, input_ids: torch.Tensor, attention_mask: Tensor, gen_len: int, get_ref_logprobs=False
    ) -> t.Tuple[torch.Tensor, torch.Tensor, t.Optional[torch.Tensor], torch.Tensor]:
        ref_logprobs = None
        model_outputs = self.model(input_ids, attention_mask=attention_mask)
        logits, v = model_outputs.logits, model_outputs.value
        logits /= self.ppo_params["temperature"]
        logprobs = logprobs_from_logits(logits[:, -gen_len - 1 : -1, :], input_ids[:, -gen_len:])
        value = v[:, -gen_len - 1 : -1]
        if get_ref_logprobs:
            ref_logits = self.ref_model(input_ids, attention_mask=attention_mask).logits
            ref_logits /= self.ppo_params["temperature"]
            ref_logprobs = logprobs_from_logits(ref_logits[:, -gen_len - 1 : -1, :], input_ids[:, -gen_len:])

        return logprobs, value, ref_logprobs, logits

    def compute_rewards(
        self, scores: torch.Tensor, logprobs: torch.Tensor, ref_logprobs: torch.Tensor
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """Compute per token rewards from scores and KL-penalty."""
        non_score_rewards = self.calculate_kl_divergence(logprobs, ref_logprobs) * -self.kl_ctl.value
        rewards = non_score_rewards.clone()
        rewards[:, -1] += scores
        return rewards, non_score_rewards

    def calculate_kl_divergence(self, logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
        """Calculate KL-divergence between logprobs and reference logprobs, depending on kl_approx."""
        logr = ref_logprobs - logprobs
        if self.ppo_params["kl_approx"] == 1:
            return -logr
        elif self.ppo_params["kl_approx"] == 2:
            return (logr**2) / 2
        elif self.ppo_params["kl_approx"] == 3:
            return logr.exp() - 1 - logr
        else:
            raise ValueError(f"Unknown kl_approx {self.ppo_params['kl_approx']}")

    def loss(self, old_logprobs, values, rewards, query, response, input_ids, attention_mask):
        """Calculate policy and value losses."""
        lastgaelam = 0
        advantages_reversed = []
        gen_len = response.shape[1]

        for i in reversed(range(gen_len)):
            nextvalues = values[:, i + 1] if i < gen_len - 1 else 0.0
            delta = rewards[:, i] + self.ppo_params["gamma"] * nextvalues - values[:, i]
            lastgaelam = delta + self.ppo_params["gamma"] * self.ppo_params["lam"] * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        logprob, vpred, _, logits = self.get_logprobs_vpreds(input_ids, attention_mask, gen_len)

        # Value loss
        vpredclipped = clip_by_value(
            vpred, values - self.ppo_params["cliprange_value"], values + self.ppo_params["cliprange_value"]
        )
        vf_losses1 = (vpred - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))

        # Policy loss
        ratio = torch.exp(logprob - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.ppo_params["cliprange"], 1.0 + self.ppo_params["cliprange"]
        )
        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))

        with torch.no_grad():
            # Statistics
            loss = pg_loss + self.ppo_params["vf_coef"] * vf_loss

            vf_clipfrac = torch.mean(torch.gt(vf_losses2, vf_losses1).double())
            pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

            entropy = torch.mean(-logprob)
            logr = old_logprobs - logprob
            policykl = torch.mean(self.calculate_kl_divergence(logprob, old_logprobs))
            approxkl1 = torch.mean(-logr)
            approxkl2 = torch.mean((logr**2) / 2)
            # approxkl3 = torch.mean((logr.exp() - 1) - logr)
            return_mean, return_var = torch.mean(returns), torch.var(returns)
            value_mean, value_var = torch.mean(values), torch.var(values)

            stats = dict(
                loss=dict(policy=pg_loss, value=vf_loss, total=loss),
                policy=dict(
                    entropy=entropy,
                    approxkl1=approxkl1,
                    approxkl2=approxkl2,
                    # approxkl3=approxkl3,
                    policykl=policykl,
                    clipfrac=pg_clipfrac,
                    advantages=advantages.cpu(),
                    advantages_mean=torch.mean(advantages),
                    advantages_var=torch.var(advantages),
                    ratio=ratio,
                ),
                returns=dict(dist=returns.cpu(), mean=return_mean, var=return_var),
                val=dict(
                    vpred=torch.mean(vpred),
                    error=torch.mean((vpred - returns) ** 2),
                    clipfrac=vf_clipfrac,
                    mean=value_mean,
                    var=value_var,
                ),
            )
        return pg_loss, self.ppo_params["vf_coef"] * vf_loss, flatten_dict(stats)

    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl_list = data["logprobs"] - data["ref_logprobs"]
        mean_kl = torch.mean(torch.sum(kl_list, dim=1))
        mean_entropy = torch.mean(torch.sum(-data["logprobs"], dim=1))
        mean_non_score_reward = torch.mean(torch.sum(data["non_score_reward"], dim=1))
        stats = {
            "objective/kl": mean_kl,
            "objective/kl_dist": kl_list.cpu().numpy(),
            "objective/logprobs": data["logprobs"].cpu().numpy(),
            "objective/ref_logprobs": data["ref_logprobs"].cpu().numpy(),
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
        }

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        stats["ppo/val/var_explained"] = 1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
        return stats
