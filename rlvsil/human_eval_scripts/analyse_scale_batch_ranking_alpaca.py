"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import scaleapi


def make_result_dict(
    reference_index: int,
    model_name,
    task: scaleapi.Task,
    audited: bool,
    preference_score: float,
) -> Dict[str, Any]:
    return {
        "preference_score": preference_score,
        "reference_rank": reference_index,
        "model_name": model_name,
        "task_id": task.id,
        "content": task.params["attachments"][0]["content"],
        "audited": audited,
        "batch_name": task.batch,
    }


def produce_result_dict(task: scaleapi.Task, ranking: np.ndarray, audited: bool) -> Dict[str, Any]:
    reference_idx = task.metadata["reference_idx"]
    model_name = task.metadata["model_name"]

    reference_preferred = int(ranking[0]) == int(reference_idx)

    # calculate which run ids are before "reference" and which are after in ordered_run_ids
    return make_result_dict(
        reference_idx,
        model_name,
        task,
        audited,
        preference_score=0.0 if reference_preferred else 1.0,
    )


def process_task(task: scaleapi.Task) -> Tuple[Dict[str, Any], Optional[float]]:
    """Takes the task and returns a dict from model_id to bool specifying
    whether the model's output was preferred."""
    audited = (
        len(getattr(task, "prior_responses", [])) > 0
        or getattr(task, "customer_review_status", None) is not None
    )

    ranking = np.array(task.response["annotations"]["Rank Responses"])
    result = produce_result_dict(task, ranking, audited)

    disagreement = None
    if len(prior_responses := getattr(task, "prior_responses", [])) > 0:
        prior_results = produce_result_dict(
            task, np.array(prior_responses[-1]["annotations"]["Rank Responses"]), audited
        )
        disagreement = int(result["preference_score"] != prior_results["preference_score"])
        # final_results = prior_results  #  when you want to analyse original results
    elif getattr(task, "customer_review_status", None) == "accepted":
        disagreement = 0

    return result, disagreement


def calculate_disagreement(final_results: List[Dict[str, Any]], prior_results: List[Dict[str, Any]]) -> float:
    """Calculate the disagreement between the final and prior results."""
    final_results = sorted(final_results, key=lambda x: x["run_id"])
    prior_results = sorted(prior_results, key=lambda x: x["run_id"])

    final_scores = np.array([x["preference_score"] for x in final_results])
    prior_scores = np.array([x["preference_score"] for x in prior_results])

    return np.abs(final_scores - prior_scores).mean()


def parse_instruction(task_text: str) -> str:
    """Parse the instruction from the task text."""
    return task_text.split("# Instruction\n\n")[1].split("\n\n# Responses")[0]


@click.command()
@click.argument("scale_api_key")
@click.argument("scale_project_name", default="rlvsil-human-eval-ranking")
@click.argument("scale_batch_names_str")
@click.option(
    "--df_save_file",
    default=None,
    help="Where (if anywhere) to save the resulting dataframe, for further analysis",
)
@click.option("--annotations_csv", default=None, help="Where to load GPT annotations to calculate agreement.")
def analyse_scale_batch(
    scale_api_key: str,
    scale_project_name: str,
    scale_batch_names_str: str,
    df_save_file: Optional[str],
    annotations_csv: Optional[str],
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

    result: List[Dict[str, Any]] = []
    disagreements: List[float] = []
    if annotations_csv is not None:
        annotations_df = pd.read_csv(annotations_csv)
        annotator_preference_scores = []
    for task in tasks:
        try:
            task_res, disagreement = process_task(task)
        except ValueError as e:
            print("Task Error: ", task, e)
            continue
        result.append(task_res)
        if annotations_csv is not None:
            # Find the corresponding row in the agreement csv
            task_row = annotations_df[annotations_df["instruction"] == parse_instruction(task_res["content"])]
            if len(task_row) == 0 or len(task_row) > 1:
                print(f"Got {len(task_row)} rows for task {task_res['task_id']}")
                continue
            else:
                task_row = task_row.iloc[0]
            # Calculate the disagreement
            if task_row["generator_1"] == task_res["model_name"]:
                annotation_preference_score = 1.0 - (task_row["preference"] - 1.0)
            elif task_row["generator_2"] == task_res["model_name"]:
                annotation_preference_score = task_row["preference"] - 1.0
            else:
                raise ValueError("Model name not found in annotations csv")
            annotator_preference_scores.append(annotation_preference_score)
            disagreement = float(np.abs(task_res["preference_score"] - annotation_preference_score))
            disagreements.append(disagreement)
        elif disagreement is not None:
            disagreements.append(disagreement)

    click.echo(f"Got {len(result)} results from scale tasks.")
    click.echo(f"Got {len(disagreements)} results for calculating annotator-researcher agreement")
    if len(disagreements) > 0:
        click.echo(f"Mean disagreement: {np.mean(disagreements)}")

    results_df = pd.DataFrame(result)

    # Add information about model_name and subset to results_df
    click.echo("Results:")
    click.echo(f"Preference Score: {results_df['preference_score'].mean()}")
    # Echo annotation preference score if present
    if annotations_csv is not None:
        click.echo(f"Preference Score (from annotations): {np.mean(annotator_preference_scores)}")

    if df_save_file:
        results_df.to_csv(df_save_file, index=False)
        click.echo(f"Saved results dataframe to {df_save_file}")


if __name__ == "__main__":
    analyse_scale_batch()
