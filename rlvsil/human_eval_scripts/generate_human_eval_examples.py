"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""This script retrieves human evaluations from Scale for several models, and
then returns examples where both models were preferred."""
import itertools
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd
import scaleapi


def get_relevant_scale_tasks(
    scale_client: scaleapi.ScaleClient,
    wandb_run_ids: List[str],
    scale_project_name: str,
) -> List[scaleapi.Task]:
    """Queries scale_client for relevant batches (by tags), and then gets corresponding tasks."""
    return list(scale_client.get_tasks(scale_project_name, tags=wandb_run_ids))


def aggregate_scale_tasks(tasks: List[scaleapi.Task]) -> List[Dict[str, Any]]:
    """Given the list of scale tasks, aggregates those which are on the same post together.

    Returns lists of dict like
    {
        post: ..
        response: {
            <run_id>: {
                response: ...
                preference_score: <int>
                summaries: <str>
                task_url: <str>
            }
        }
    }
    """
    tasks_post_to_details: dict = defaultdict(list)
    for task in tasks:
        post = task.params["attachments"][0]["content"].split("## Summaries")[0]
        summaries = [
            attachment["content"].split("## Summaries")[1].split("|")[-3:-1]
            for attachment in task.params["attachments"]
        ]
        metadata = task.metadata
        response = task.response
        for ref, run_ids, annotation, summary_pair in zip(
            metadata["references"], metadata["run_ids"], response["annotations"].values(), summaries
        ):
            for run_id in run_ids:
                preference_score = -(ref - 0.5) * int(annotation)
                tasks_post_to_details[post].append(
                    dict(
                        run_id=run_id,
                        response=response,
                        summary=summary_pair[1 - ref],
                        reference=summary_pair[ref],
                        preference_score=preference_score,
                        task_url=f"https://dashboard.scale.com/audit?taskId={task.id}&benchmark=1",
                    )
                )

    aggregated_tasks = []
    for post, details in tasks_post_to_details.items():
        aggregated_tasks.append(
            dict(
                post=post,
                response={
                    detail["run_id"]: dict(
                        response=detail["response"],
                        preference_score=detail["preference_score"],
                        summary=detail["summary"],
                        reference=detail["reference"],
                        task_url=detail["task_url"],
                    )
                    for detail in details
                },
            )
        )
    return aggregated_tasks


def find_contrasting_examples(
    scale_tasks: List[Dict[str, Any]], wandb_run_ids: Tuple[str, str], n_examples: Optional[int], strict: bool
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Finds n_examples where wandb_run_ids[0] was preferred to wandb_run_ids[1], and vice versa.

    If strict, then the example needs to have the first run preferred to ref and the second not.
    If not strict, it just needs to have preference_score for the first run > than for the second run.
    """
    run_id_0, run_id_1 = wandb_run_ids
    results_0 = []
    results_1 = []
    for task in scale_tasks:
        if run_id_0 not in task["response"] or run_id_1 not in task["response"]:
            continue
        score_0 = task["response"][run_id_0]["preference_score"]
        score_1 = task["response"][run_id_1]["preference_score"]
        if strict:
            if score_0 > 0 and score_1 < 0:
                results_0.append(task)
            elif score_0 < 0 and score_1 > 0:
                results_1.append(task)
        else:
            if score_0 > score_1:
                results_0.append(task)
            elif score_0 < score_1:
                results_1.append(task)

        if n_examples is not None and len(results_0) >= n_examples and len(results_1) >= n_examples:
            break

    return results_0[:n_examples], results_1[:n_examples]


@click.command()
@click.argument("scale_api_key")
@click.argument("scale_project_name")
@click.argument("wandb_run_ids")
@click.option("--n_examples", default=None, type=int, help="Number of examples.")
@click.option(
    "--strict",
    default=False,
    type=bool,
    help="Whether to only return examples with preference_score either side of 0.",
)
@click.option("--save_results_file", default=None, help="If set, saves results to this file.")
@click.option("--print_results", default=0, type=int, help="How many results to print")
def generate_human_eval_examples(
    scale_api_key: str,
    scale_project_name: str,
    wandb_run_ids: str,
    n_examples: Optional[int],
    strict: bool,
    save_results_file: Optional[str],
    print_results: int,
):
    run_ids = wandb_run_ids.split(",")

    scale_client = scaleapi.ScaleClient(api_key=scale_api_key)

    click.echo("Getting relevant scale tasks...")
    scale_tasks = get_relevant_scale_tasks(scale_client, run_ids, scale_project_name)
    click.echo(f"Got {len(scale_tasks)} scale tasks. Aggregating...")
    aggregated_tasks = aggregate_scale_tasks(scale_tasks)
    click.echo(f"Aggregated down to {len(aggregated_tasks)} tasks.")

    contrasting_examples = {}
    for run_id_pair in itertools.combinations(run_ids, 2):
        click.echo(f"Finding contrasting examples for {run_id_pair}")
        results = find_contrasting_examples(aggregated_tasks, run_id_pair, n_examples, strict)
        contrasting_examples[" > ".join(run_id_pair)] = results[0]
        contrasting_examples[" > ".join(reversed(run_id_pair))] = results[1]

    for key, examples in contrasting_examples.items():
        click.echo(f"Got {len(examples)} examples for {key}")

    if print_results:
        click.echo("Results:")
        for key, examples in contrasting_examples.items():
            click.echo("========================================")
            click.echo(f"{key} ({len(examples)} examples)")
            click.echo("------------------------------------------")
            for example in examples[:print_results]:
                click.echo(example["post"].strip())
                click.echo()
                click.echo("## Summaries")
                click.echo()
                key_0, key_1 = key.split(" > ")
                res_0, res_1 = example["response"][key_0], example["response"][key_1]
                url_0, url_1 = res_0["task_url"], res_1["task_url"]
                if url_0 == url_1:
                    click.echo("Task url: " + res_0["task_url"])
                    url_0, url_1 = None, None
                click.echo("Reference: " + res_0["reference"])
                click.echo()
                click.echo(f"{key_0} (score: {res_0['preference_score']}) {res_0['summary']}".strip())
                click.echo(url_0)
                click.echo()
                click.echo(f"{key_1} (score: {res_1['preference_score']}) {res_1['summary']}".strip())
                click.echo(url_1)
                click.echo()
                click.echo("------------------------------------------")
    if save_results_file is not None:
        click.echo(f"Saving results to {save_results_file}")
        with open(save_results_file, "w") as f:
            json.dump(contrasting_examples, f)


if __name__ == "__main__":
    generate_human_eval_examples()
