"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, Dict, List

import click
import numpy as np
import scaleapi


def process_task(task: scaleapi.Task) -> float:
    """Takes the task and how much disagreement there was between user ratings and original ratings"""

    user_ranking = list(map(int, task.response["annotations"]["Rank Summaries"]))
    ref_index = task.metadata["ref_index"]
    ref_index_in_ranking = user_ranking.index(ref_index)
    agree = 0

    for i, el in enumerate(task.metadata["preferred"]):
        agree += int((user_ranking.index(i) < ref_index_in_ranking) == el)

    return agree / len(task.metadata["preferred"])


def calculate_disagreement(final_results: List[Dict[str, Any]], prior_results: List[Dict[str, Any]]) -> float:
    """Calculate the disagreement between the final and prior results."""
    final_results = sorted(final_results, key=lambda x: x["run_id"])
    prior_results = sorted(prior_results, key=lambda x: x["run_id"])

    final_scores = np.array([x["preference_score"] for x in final_results])
    prior_scores = np.array([x["preference_score"] for x in prior_results])

    return np.abs(final_scores - prior_scores).mean()


@click.command()
@click.argument("scale_api_key")
@click.argument("scale_project_name", default="rlvsil-human-eval-ranking")
@click.argument("scale_batch_names_str")
def analyse_scale_batch(
    scale_api_key: str,
    scale_project_name: str,
    scale_batch_names_str: str,
):
    """Take a scale batch name and details, scrape data from scale api, then process."""

    scale_batch_names = scale_batch_names_str.split(",")

    # Set up scale api
    client = scaleapi.ScaleClient(api_key=scale_api_key)

    click.echo(f"Getting tasks for {len(scale_batch_names)} batches from scale api")
    # Get the tasks
    tasks = []
    for scale_batch_name in scale_batch_names:
        scale_tasks = list(
            client.get_tasks(
                project_name=scale_project_name,
                batch_name=scale_batch_name,
                status=scaleapi.TaskStatus.Completed,
            )
        )
        tasks.extend(scale_tasks)
    click.echo(f"Retrieved {len(tasks)} tasks from scale.")

    agreements: List[float] = []
    for task in tasks:
        agreement = process_task(task)
        agreements.append(agreement)

    click.echo(f"Got {len(agreements)} results from scale tasks.")
    click.echo(f"Mean agreement: {np.mean(agreements)}")


if __name__ == "__main__":
    analyse_scale_batch()
