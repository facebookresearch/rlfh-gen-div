"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

# This file taken from
#     https://github.com/deepmind/scalable_agent/blob/
#         cd66d00914d56c8ba2f0615d9cdeefcb169a8d70/vtrace.py
# and modified.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

import collections

import torch
import torch.nn.functional as F

VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, -2), dim=-1),
        torch.flatten(actions),
        reduction="none",
    ).view_as(actions)


def from_logits(
    behavior_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    lam,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    importance_sampling_correction=True,
    use_gae_lambda_advantages=False,
):
    """V-trace for softmax policies."""

    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        lam=lam,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        importance_sampling_correction=importance_sampling_correction,
        use_gae_lambda_advantages=use_gae_lambda_advantages,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    lam,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    importance_sampling_correction=True,
    use_gae_lambda_advantages=False,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None and importance_sampling_correction:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0) if importance_sampling_correction else torch.ones(discounts.shape[0])
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat([values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
        deltas = (clipped_rhos if importance_sampling_correction else 1) * (
            rewards + discounts * values_t_plus_1 - values
        )

        vs_acc = torch.zeros_like(bootstrap_value)
        adv_acc = torch.zeros_like(bootstrap_value)
        vs_result = []
        adv_result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            vs_acc = deltas[t] + discounts[t] * lam * cs[t] * vs_acc
            adv_acc = deltas[t] + discounts[t] * lam * adv_acc
            vs_result.append(vs_acc)
            adv_result.append(adv_acc)
        vs_result.reverse()
        vs_minus_v_xs = torch.stack(vs_result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        if use_gae_lambda_advantages:
            adv_result.reverse()
            advantages = torch.stack(adv_result)
        else:
            broadcasted_bootstrap_values = torch.ones_like(vs[0]) * bootstrap_value
            vs_t_plus_1 = torch.cat([vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0)
            advantages = rewards + discounts * vs_t_plus_1 - values

        if clip_pg_rho_threshold is not None and importance_sampling_correction:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos

        pg_advantages = (clipped_pg_rhos if importance_sampling_correction else 1) * advantages

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)


def calculate_kl_divergence(logprobs, ref_logprobs, FLAGS):
    logr = ref_logprobs - logprobs
    if FLAGS.kl_approx == 1:
        return -logr
    elif FLAGS.kl_approx == 2:
        return (logr**2) / 2
    elif FLAGS.kl_approx == 3:
        return logr.exp() - 1 - logr
    else:
        raise ValueError(f"Unknown kl_approx {FLAGS.kl_approx}")
