"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""Calculate diversity metrics on model outputs, both for a given inputs and across inputs."""
import typing as t

import numpy as np

from diversity.diversity_metrics import (
    AveragedDistinctNgrams, AveragedExpectationAdjustedDistinctNgrams,
    CosineSimilarity2Diversity, NLIFromSim, NLISampleFromSim,
    OpenAiEmbeddingsFromSim, SentBertFromSim)

DEFAULT_CONFIGS = {
    "averaged_distinct_ngrams": AveragedDistinctNgrams.default_config,
    "ead_averaged_distinct_ngrams": AveragedExpectationAdjustedDistinctNgrams.default_config,
    "cosine_similarity_2d_diversity": CosineSimilarity2Diversity.default_config,
    "sent_bert_from_sim": SentBertFromSim.default_config,
    "nli_from_sim": NLIFromSim.default_config,
    "nli_sample_from_sim": NLISampleFromSim.default_config,
    "openai_from_sim": OpenAiEmbeddingsFromSim.default_config,
}


METRICS = [
    AveragedDistinctNgrams,
    AveragedExpectationAdjustedDistinctNgrams,
    CosineSimilarity2Diversity,
    SentBertFromSim,
    NLIFromSim,
    OpenAiEmbeddingsFromSim,
    NLISampleFromSim,
]


def initialise_metrics(metric_configs=DEFAULT_CONFIGS):
    """Initialise diversity metrics from config."""
    if metric_configs is None:
        metric_configs = DEFAULT_CONFIGS
    metrics = []
    for metric in METRICS:
        if metric.name in metric_configs:
            metrics.append(metric(metric_configs[metric.name]))
    return metrics


def calculate_output_diversity(outputs: t.List[str], metrics) -> t.Dict[str, float]:
    """Calculate diversity metrics on model outputs, both for a given inputs and across inputs."""
    results = {}
    for metric in metrics:
        try:
            results[metric.name] = metric(outputs)
        except Exception as e:
            print(f"Error calculating diversities for metric {metric}: {e}")
    return results


def calculate_diversity_metrics(
    outputss: t.List[t.List[str]],
    metric_configs=DEFAULT_CONFIGS,
    no_per_input=False,
    no_overall_input=False,
    sample_overall=False,
) -> t.Dict[str, float]:
    """Calculate diversity metrics on model outputs, both for a given inputs and across inputs."""

    metrics = initialise_metrics(metric_configs)
    results = {}

    if not no_per_input:
        print("calculating per-input diversities")
        per_input_diversities = []
        i = 1
        for outputs in outputss:
            per_input_diversities.append(calculate_output_diversity(outputs, metrics))
            if i % 10 == 0:
                print(f"{i}/{len(outputss)}")

        average_per_input_diversity = {
            f"mean_per_input_{k}": np.mean([d[k] for d in per_input_diversities])
            for k in per_input_diversities[0].keys()
        }
        std_per_input_diversity = {
            f"std_per_input_{k}": np.std([d[k] for d in per_input_diversities])
            for k in per_input_diversities[0].keys()
        }
        results.update(average_per_input_diversity)
        results.update(std_per_input_diversity)
        print("Average per-input diversities:")
        print(average_per_input_diversity)
        print("Std per-input diversities:")
        print(std_per_input_diversity)

    if not no_overall_input:
        print("calculating overall diversities")
        if not sample_overall:
            overall_diversities = calculate_output_diversity(
                [output for outputs in outputss for output in outputs], metrics
            )
        else:
            outputs = [output for outputs in outputss for output in outputs]
            outputs_sampled_subset = list(np.random.choice(outputs, replace=False, size=500))
            overall_diversities = calculate_output_diversity(outputs_sampled_subset, metrics)

        overall_diversities = {f"overall_{k}": v for k, v in overall_diversities.items()}

        print("Average overall diversities:")
        print(overall_diversities)

        results.update(overall_diversities)

    print("calculating overall single-input diversities")
    overall_single_output_diversities = calculate_output_diversity(
        [outputs[0] for outputs in outputss], metrics
    )
    overall_single_output_diversities = {
        f"overall_single_output_{k}": v for k, v in overall_single_output_diversities.items()
    }

    print("Average overall single-input diversities:")
    print(overall_single_output_diversities)

    results.update(overall_single_output_diversities)

    return results


if __name__ == "__main__":
    # Test functions on some example inputs
    outputss = [
        [
            "I like to eat apples.",
            "I like to eat bananas.",
            "I like to eat oranges.",
        ],
        [
            "I love to eat apples.",
            "I love to eat bananas.",
            "I love to eat oranges.",
        ],
        [
            "I love muching on apples.",
            "I love muching on bananas.",
            "I love muching on oranges.",
        ],
    ]
    config = DEFAULT_CONFIGS.copy()
    del config["sent_bert_from_sim"]
    print(calculate_diversity_metrics(outputss))
