"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import dataclasses
import logging
import math

import torch


@dataclasses.dataclass
class StatSum:
    value: float = 0

    def result(self):
        return self.value

    def __sub__(self, other):
        assert isinstance(other, StatSum)
        return StatSum(self.value - other.value)

    def __iadd__(self, other):
        if isinstance(other, StatSum):
            self.value += other.value
        else:
            self.value += other
        return self

    def reset(self):
        pass

    def decay_cumulative(self):
        pass

    def __repr__(self):
        return repr(self.result())


@dataclasses.dataclass
class StatMean:
    # Compute using Welfords Online Algorithm
    # Algo: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    # Math: https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
    n: int = 0
    mu: float = 0
    m2: float = 0
    cumulative: bool = False

    def result(self):
        if self.n == 0:
            return None
        return self.mu

    def mean(self):
        return self.mu

    def std(self):
        if self.n < 1:
            return None
        return math.sqrt(self.m2 / self.n)

    def __sub__(self, other):
        assert isinstance(other, StatMean)
        n_new = self.n - other.n
        if n_new == 0:
            return StatMean(0, 0, 0)
        mu_new = (self.mu * self.n - other.mu * other.n) / n_new
        delta = other.mu - mu_new
        m2_new = self.m2 - other.m2 - (delta**2) * n_new * other.n / self.n
        return StatMean(n_new, mu_new, m2_new)

    def __iadd__(self, other):
        if isinstance(other, StatMean):
            other_n = other.n
            other_mu = other.mu
            other_m2 = other.m2
        elif isinstance(other, torch.Tensor):
            other_n = other.numel()
            other_mu = other.mean().item()
            other_m2 = ((other - other_mu) ** 2).sum().item()
        else:
            other_n = 1
            other_mu = other
            other_m2 = 0
        # See parallelized Welford in wiki
        new_n = other_n + self.n
        delta = other_mu - self.mu
        self.mu += delta * (other_n / max(new_n, 1))
        delta2 = other_mu - self.mu
        self.m2 += other_m2 + (delta2**2) * (self.n * other_n / max(new_n, 1))
        self.n = new_n
        return self

    def reset(self):
        if not self.cumulative:
            self.mu = 0
            self.n = 0

    def decay_cumulative(self, n=1e6):
        """Adjust sample size downwards to upweight recent samples"""
        if not self.cumulative:
            return
        if self.n > n:
            self.m2 *= n / self.n
            self.n = n

    def __repr__(self):
        return repr(self.result())


class GlobalStatsAccumulator:
    """Class for global accumulation state. add_stats gets reduced."""

    def __init__(self, rpc_group, global_stats):
        self.rpc_group = rpc_group
        self.global_stats = global_stats

        self.reduce_future = None
        self.queued_global_stats = None
        self.sent_global_stats = None
        self.prev_stats = None

    def add_stats(self, dst, src):
        for k, v in dst.items():
            v += src[k]
        return dst

    def enqueue_global_stats(self, stats):
        if self.queued_global_stats is None:
            self.queued_global_stats = copy.deepcopy(stats)
        else:
            # Sum pending data.
            self.add_stats(self.queued_global_stats, stats)

    def reduce(self, stats) -> bool:
        error = False
        if self.reduce_future is not None and self.reduce_future.done():
            if self.reduce_future.exception() is not None:
                logging.info(
                    "global stats accumulation error: %s",
                    self.reduce_future.exception(),
                )
                self.enqueue_global_stats(self.sent_global_stats)
                error = True
            else:
                self.add_stats(self.global_stats, self.reduce_future.result())
                for v in self.global_stats.values():
                    v.decay_cumulative()
            self.reduce_future = None

        for v in stats.values():
            v.decay_cumulative()

        stats_diff = stats
        if self.prev_stats is not None:
            stats_diff = {k: v - self.prev_stats[k] for k, v in stats.items()}

        self.enqueue_global_stats(stats_diff)
        self.prev_stats = copy.deepcopy(stats)

        if self.reduce_future is None:
            # Only reduce when not currently reducing.
            # Otherwise, we keep queued_global_stats for next time.
            self.sent_global_stats = self.queued_global_stats
            self.queued_global_stats = None
            # Additional copy to deal with potential partial reductions.
            self.reduce_future = self.rpc_group.all_reduce(
                "global stats", copy.deepcopy(self.sent_global_stats), op=self.add_stats
            )

        return not error

    def reset(self):
        if self.prev_stats is not None:
            for _, v in self.prev_stats.items():
                v.reset()
