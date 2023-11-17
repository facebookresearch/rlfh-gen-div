"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import click
import numpy as np
import pandas as pd
import scaleapi
import wandb

CORRUPTED_RUNS = {
    "227857-13": ("facebook/opt-2.7b", 50),
    "227857-31": ("facebook/opt-6.7b", 50),
    "1690885-1": ("facebook/opt-6.7b", 50),
    "2930376-1": ("facebook/opt-2.7b", 100),
    "2930376-2": ("facebook/opt-6.7b", 100),
    "2929534-2": ("facebook/opt-6.7b", 50),
}

MODEL_NAME_TO_REAL_MODEL_NAMES = {
    "/checkpoint/ishitamed/rlvsil/final_ckpts/66372915-8/final/": "facebook/opt-2.7b",
    "/checkpoint/ishitamed/rlvsil/final_ckpts/1690885-1/final": "facebook/opt-6.7b",
    "/checkpoint/ishitamed/rlvsil/final_ckpts/66372915-21/final": "facebook/opt-6.7b",
}


def art_name(run_id: str) -> str:
    return f"{run_id}_human_evaluations_df_ranking"


def make_result_dict(
    run_id: str,
    reference_index: int,
    task: scaleapi.Task,
    audited: bool,
    res_rank: int,
    preference_score: float,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "preference_score": preference_score,
        "rank": res_rank,
        "run_ids": task.metadata["run_ids"],
        "reference_rank": reference_index,
        "neutral": preference_score == 0.5,
        "task_id": task.id,
        "content": task.params["attachments"][0]["content"],
        "subreddit": task.params["attachments"][0]["content"].split("SUBREDDIT: r/")[-1].split("\n")[0],
        "post": task.params["attachments"][0]["content"].split("POST: ")[-1].split("# Summaries")[0],
        "audited": audited,
        "batch_name": task.batch,
    }


def produce_result_dicts(task: scaleapi.Task, ranking: np.ndarray, audited: bool) -> List[Dict[str, Any]]:
    run_ids_order = np.array(task.metadata["run_ids"], dtype=object)
    ordered_run_idss = run_ids_order[ranking.astype(int)]

    results = []

    # calculate which run ids are before "reference" and which are after in ordered_run_ids
    reference_index = [i for i, idss in enumerate(ordered_run_idss) if "reference" in idss][0]
    before_reference = ordered_run_idss[:reference_index]
    after_reference = ordered_run_idss[reference_index + 1 :]

    res_rank = 0
    # Add dict for each run id to results
    for run_ids in before_reference:
        for run_id in run_ids:
            results.append(
                make_result_dict(
                    run_id,
                    reference_index,
                    task,
                    audited,
                    res_rank,
                    preference_score=1,
                )
            )
        res_rank += 1

    if len(ordered_run_idss[reference_index]) > 1:
        for run_id in ordered_run_idss[reference_index]:
            if run_id == "reference":
                continue
            results.append(
                make_result_dict(
                    run_id,
                    reference_index,
                    task,
                    audited,
                    res_rank,
                    preference_score=0.5,
                )
            )

    res_rank += 1
    for run_ids in after_reference:
        for run_id in run_ids:
            results.append(
                make_result_dict(
                    run_id,
                    reference_index,
                    task,
                    audited,
                    res_rank,
                    preference_score=0,
                )
            )
        res_rank += 1

    return results


def process_task(task: scaleapi.Task) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """Takes the task and returns a dict from model_id to bool specifying
    whether the model's output was preferred."""
    audited = (
        len(getattr(task, "prior_responses", [])) > 0
        or getattr(task, "customer_review_status", None) is not None
    )

    ranking = task.response["annotations"]["Rank Summaries"]
    if ranking == "0":
        click.echo(f"Task {task.id} has no annotations")
        return [], None
    final_results = produce_result_dicts(task, np.array(ranking), audited)

    disagreement = None
    if len(prior_responses := getattr(task, "prior_responses", [])) > 0:
        prior_results = produce_result_dicts(
            task, np.array(prior_responses[-1]["annotations"]["Rank Summaries"]), audited
        )
        disagreement = calculate_disagreement(final_results, prior_results)
        # final_results = prior_results  #  when you want to analyse original results
    elif getattr(task, "customer_review_status", None) == "accepted":
        disagreement = 0

    return final_results, disagreement


def calculate_disagreement(final_results: List[Dict[str, Any]], prior_results: List[Dict[str, Any]]) -> float:
    """Calculate the disagreement between the final and prior results."""
    final_results = sorted(final_results, key=lambda x: x["run_id"])
    prior_results = sorted(prior_results, key=lambda x: x["run_id"])

    final_scores = np.array([x["preference_score"] for x in final_results])
    prior_scores = np.array([x["preference_score"] for x in prior_results])

    return np.abs(final_scores - prior_scores).mean()


def get_wandb_runs(wandb_run_ids: Set[str]) -> Dict[str, wandb.apis.public.Run]:
    """Retrieves the Runs from the wandb api."""
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
    api = wandb.Api()

    return {run_id: api.run(f"ucl-dark/rlvsil-main/{run_id}") for run_id in wandb_run_ids}


def get_model_name_and_dataset(run) -> Tuple[str, Union[str, int]]:
    if CORRUPTED_RUNS.get(run.id):
        return CORRUPTED_RUNS[run.id]

    model_name = None
    if run.config.get("model_name"):
        model_name = run.config["model_name"]
    else:
        # Get model name from command used to start run
        args = json.load(run.file("wandb-metadata.json").download(replace=True))["args"]
        if args:
            for i, v in enumerate(args):
                if v == "--model_name":
                    model_name = args[i + 1]
                    break
        if not model_name:
            try:
                model_name = CORRUPTED_RUNS[run.id][0]
            except KeyError:
                model_name = run.id.split("_")[1]

    subset = None

    if run.config.get("dataset_random_subset"):
        subset = run.config["dataset_random_subset"]
    else:
        # Get subset from command used to start run
        args = json.load(run.file("wandb-metadata.json").download(replace=True))["args"]
        if args:
            for i, v in enumerate(args):
                if v == "--dataset_random_subset":
                    subset = args[i + 1]
                    break

    if not subset:
        if run.config.get("dataset_structured_subset"):
            subset = run.config["dataset_structured_subset"]
        else:
            # Get subset from command used to start run
            args = json.load(run.file("wandb-metadata.json").download(replace=True))["args"]
            if args:
                for i, v in enumerate(args):
                    if v == "--dataset_structured_subset":
                        subset = args[i + 1]
                        break

    if not subset:
        subset = run.id.split("_")[2]

    training = run.summary.get("train/epoch") is not None
    if not training:
        subset = 0

    return model_name, subset


def get_wandb_run_dataframes(
    wandb_runs: Dict[str, wandb.apis.public.Run],
) -> Dict[str, Optional[pd.DataFrame]]:
    """Check whether each wandb run has the artifact 'human_evaluations_df', and if so return it."""
    results = {}

    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
    api = wandb.Api()

    for run_id, run in wandb_runs.items():
        try:
            artifact = api.artifact(f"ucl-dark/rlvsil-main/{art_name(run_id)}:latest")
            if not os.path.exists(artifact.file()):
                artifact.download()
            results[run_id] = pd.read_csv(artifact.file())
        except wandb.errors.CommError:
            results[run_id] = None
    return results


def get_results_from_df(
    df: pd.DataFrame,
    wandb_runs: Dict[str, wandb.apis.public.Run],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    run_id_to_preference_score = defaultdict(list)
    run_id_to_comparison_count: dict = defaultdict(int)
    for run_id in wandb_runs.keys():
        df_filter = df["run_id"] == run_id
        run_id_subdf = df[df_filter]
        run_id_to_preference_score[run_id] = run_id_subdf["preference_score"].mean()
        run_id_to_comparison_count[run_id] = len(run_id_subdf)

    return run_id_to_preference_score, run_id_to_comparison_count


def update_wandb_artifacts(
    wandb_run_dfs: Dict[str, pd.DataFrame],
):
    """Update wandb artifacts with new human evaluations."""
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
    api = wandb.Api()
    for run_id, df in wandb_run_dfs.items():
        file_name = f"save_data/{art_name(run_id)}.csv"
        df.to_csv(file_name, index=False)  # type: ignore
        artifact = wandb.Artifact(
            name=art_name(run_id),
            type="human_eval",
        )
        artifact.add_file(file_name)
        try:
            # Check whether artifact already exists
            api.artifact(f"ucl-dark/rlvsil-main/{art_name(run_id)}:latest")
            artifact.save()
        except wandb.errors.CommError:
            # If not, create it, which we have to do by initialising a run
            wandb.init(project="rlvsil-main", entity="ucl-dark", id=run_id, reinit=True)
            wandb.log_artifact(artifact)
            wandb.finish()


def update_results_on_wandb(
    wandb_run_dfs: Dict[str, pd.DataFrame],
    wandb_runs: Dict[str, wandb.apis.public.Run],
    wandb_update_strategy: str,
    results_df: pd.DataFrame,
    run_id_to_model_name_and_dataset: Dict[str, Tuple[str, Union[str, int]]],
) -> Dict[str, pd.DataFrame]:
    click.echo("Updating wandb runs with results.")
    score_name = "human_evaluation_score_ranking"
    count_name = "human_evaluation_count_ranking"
    if wandb_update_strategy == "average":
        # Update local dfs with newly calculated dfs
        for run_id in wandb_run_dfs.keys():
            if wandb_run_dfs[run_id] is not None:
                wandb_run_dfs[run_id] = pd.concat(
                    [wandb_run_dfs[run_id], results_df[results_df["run_id"] == run_id]]
                )
            else:
                wandb_run_dfs[run_id] = pd.concat(
                    [wandb_run_dfs[run_id], results_df[results_df["run_id"] == run_id]]
                )
    elif wandb_update_strategy == "override":
        for run_id in wandb_run_dfs.keys():
            wandb_run_dfs[run_id] = results_df[results_df["run_id"] == run_id]

    if wandb_update_strategy != "skip":
        # Upload updated dfs to wandb
        click.echo("Uploading updated dataframes to wandb.")
        update_wandb_artifacts(wandb_run_dfs)

    for run_id, run in wandb_runs.items():
        run_id_to_preference_score, run_id_to_comparison_count = get_results_from_df(
            pd.concat(wandb_run_dfs.values()), wandb_runs
        )
        eval_score = run_id_to_preference_score[run_id]
        comparison_count = run_id_to_comparison_count[run_id]
        if run.config.get(score_name) is not None and wandb_update_strategy == "skip":
            click.echo(f"Skipping run {run_id} as it already has a score.")

        run.config.update(
            {
                score_name: eval_score,
                count_name: comparison_count,
            }
        )
        run.update()

    click.echo("Updated wandb runs, new values: ")
    for run_id, run in wandb_runs.items():
        model_name, subset = run_id_to_model_name_and_dataset[run_id]
        click.echo(
            f"{run_id}: {model_name} - {subset}" f": {run.config[score_name]} ({run.config[count_name]})"
        )

    return wandb_run_dfs


def parse_task(task_res: Dict[str, Any]) -> Tuple[str, ...]:
    """Parse the input from the task text."""
    model_outputs = tuple((el[4:] for el in task_res["content"].split("\n\n## Summary")[1:]))
    if task_res["run_ids"][0][0] == "reference":
        return model_outputs[1], model_outputs[0]
    else:
        return model_outputs[0], model_outputs[1]


@click.command()
@click.argument("scale_api_key")
@click.argument("scale_project_name", default="rlvsil-human-eval-ranking")
@click.argument("scale_batch_names_str")
@click.option("--update_on_wandb", default=False, help="Update wandb with calculated statistics.")
@click.option(
    "--wandb_update_strategy",
    default="average",
    help="How to update wandb, either average, skip or override.",
)
@click.option(
    "--df_save_file",
    default=None,
    help="Where (if anywhere) to save the resulting dataframe, for further analysis",
)
@click.option(
    "--wandb_df_save_file",
    default=None,
    help="Where (if anywhere) to save the wandb dataframe, for further analysis",
)
@click.option(
    "--annotations_json", default=None, help="Where to load GPT annotations to calculate agreement."
)
def analyse_scale_batch(
    scale_api_key: str,
    scale_project_name: str,
    scale_batch_names_str: str,
    update_on_wandb: bool,
    wandb_update_strategy: str,
    df_save_file: Optional[str],
    wandb_df_save_file: Optional[str],
    annotations_json: Optional[str],
):
    """Take a scale batch name and details, scrape data from scale api, then process."""

    scale_batch_names = scale_batch_names_str.split(",")

    # Set up scale api
    client = scaleapi.ScaleClient(api_key=scale_api_key)

    # click.echo(f"Getting tasks for {len(scale_batch_names)} batches from scale api")
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
    # click.echo(f"Retrieved {len(tasks)} tasks from scale.")

    result: List[Dict[str, Any]] = []
    disagreements: List[float] = []
    if annotations_json is not None:
        annotations_df = pd.read_json(annotations_json)
        annotator_preference_scores = []
    for task in tasks:
        try:
            task_ress, disagreement = process_task(task)
        except ValueError as e:
            print("Task Error: ", task, e)
            continue
        result.extend(task_ress)
        if annotations_json is not None:
            # Find the corresponding row in the agreement csv
            for task_res in task_ress:
                output_model, output_reference = parse_task(task_res)
                task_row = annotations_df[
                    (
                        (annotations_df["output_1"] == output_model)
                        & (annotations_df["output_2"] == output_reference)
                    )
                    | (
                        (annotations_df["output_1"] == output_reference)
                        & (annotations_df["output_2"] == output_model)
                    )
                ]
                if len(task_row) == 0 or len(task_row) > 1:
                    print(f"Got {len(task_row)} rows for task {task_res['task_id']}")
                    continue
                else:
                    task_row = task_row.iloc[0]
                # Calculate the disagreement
                if task_row["output_1"] == output_model:
                    annotation_preference_score = 1.0 - (task_row["preference"] - 1.0)
                elif task_row["output_2"] == output_model:
                    annotation_preference_score = task_row["preference"] - 1.0
                else:
                    raise ValueError("Model name not found in annotations csv")
                annotator_preference_scores.append(annotation_preference_score)
                disagreement = float(np.abs(task_res["preference_score"] - annotation_preference_score))
                disagreements.append(disagreement)
        elif disagreement is not None:
            disagreements.append(disagreement)

    # click.echo(f"Got {len(result)} results from scale tasks.")
    # click.echo(f"Got {len(disagreements)} results for calculating annotator-researcher agreement")
    if len(disagreements) > 0:
        click.echo(f"Mean disagreement: {np.mean(disagreements)}")
    if annotations_json is not None and len(annotator_preference_scores) > 0:
        click.echo(f"Mean annotator preference score: {np.mean(annotator_preference_scores)}")

    results_df = pd.DataFrame(result)
    wandb_runs = get_wandb_runs(set(results_df["run_id"]))

    run_id_to_model_name_and_dataset = {
        run_id: get_model_name_and_dataset(wandb_run) for run_id, wandb_run in wandb_runs.items()
    }
    # Add information about model_name and subset to results_df
    model_name_and_dataset = [run_id_to_model_name_and_dataset[run_id] for run_id in results_df["run_id"]]
    results_df["model_name"] = [x[0] for x in model_name_and_dataset]
    results_df["dataset"] = [x[1] for x in model_name_and_dataset]

    run_id_to_preference_score, run_id_to_comparison_count = get_results_from_df(results_df, wandb_runs)

    click.echo("Results:")
    for run_id in wandb_runs.keys():
        model_name, subset = run_id_to_model_name_and_dataset[run_id]
        click.echo(
            f"{run_id}: {model_name} - {subset}"
            f": {run_id_to_preference_score[run_id]} ({run_id_to_comparison_count[run_id]})"
        )
    click.echo()
    wandb_run_dfs = get_wandb_run_dataframes(wandb_runs)
    if update_on_wandb:
        wandb_run_dfs = update_results_on_wandb(
            wandb_run_dfs,
            wandb_runs,
            wandb_update_strategy,
            results_df,
            run_id_to_model_name_and_dataset,
        )

    if df_save_file:
        results_df.to_csv(df_save_file, index=False)
        click.echo(f"Saved results dataframe to {df_save_file}")

    if wandb_df_save_file:
        wandb_df = pd.concat(wandb_run_dfs.values())
        wandb_df.to_csv(wandb_df_save_file, index=False)
        click.echo(f"Saved wandb dataframe to {wandb_df_save_file}")


if __name__ == "__main__":
    analyse_scale_batch()
