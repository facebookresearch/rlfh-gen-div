"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import click
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


def art_name(run_id: str, weighted_preference: bool) -> str:
    return (
        f"{run_id}_human_evaluations_df_weighted" if weighted_preference else f"{run_id}_human_evaluations_df"
    )


def process_task(
    task: scaleapi.Task, weighted_preference: bool = False, include_zeros: bool = False
) -> Tuple[List[Dict[str, Any]], List[scaleapi.Task]]:
    """Takes the task and returns a dict from model_id to bool specifying
    whether the model's output was preferred."""
    response = task.response
    metadata = task.metadata
    results = []
    corrupted_results = []

    for ref, run_ids, annotation, attachments in zip(
        metadata["references"],
        metadata["run_ids"],
        response["annotations"].values(),
        task.params["attachments"],
    ):
        if "\n" in attachments["content"].split("|\n|")[-1]:
            corrupted_results.append(task)
        else:
            for run_id in run_ids:
                if not include_zeros and annotation == "0":
                    preference_score = None
                else:
                    if weighted_preference:
                        # Weighted preference, where we return the annotation scaled by 0.5
                        preference_score = -(ref - 0.5) * int(annotation)
                    else:
                        # ref = 1 implies Summary 1 is model, and negative
                        # annotation numbers signify model preferered
                        preference_score = (ref - 0.5) * int(annotation) < 0
                results.append(
                    dict(
                        run_id=run_id,
                        preference_score=preference_score,
                        neutral=annotation == "0",
                    )
                )

    return results, corrupted_results


def get_wandb_runs(wandb_run_ids: Set[str]) -> Dict[str, wandb.apis.public.Run]:
    """Retrieves the Runs from the wandb api."""
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
    api = wandb.Api()

    return {run_id: api.run(f"ucl-dark/rlvsil-main/{run_id}") for run_id in wandb_run_ids}


def get_model_name_and_dataset(run) -> Tuple[str, int]:
    if CORRUPTED_RUNS.get(run.id):
        return CORRUPTED_RUNS[run.id]

    if run.config.get("model_name"):
        model_name = run.config["model_name"]
    else:
        # Get model name from command used to start run
        args = json.load(run.file("wandb-metadata.json").download(replace=True))["args"]
        for i, v in enumerate(args):
            if v == "--model_name":
                model_name = args[i + 1]
                break
        model_name = CORRUPTED_RUNS[run.id][0]

    if run.config.get("dataset_random_subset"):
        subset = run.config["dataset_random_subset"]
    else:
        # Get subset from command used to start run
        subset = None
        args = json.load(run.file("wandb-metadata.json").download(replace=True))["args"]
        for i, v in enumerate(args):
            if v == "--dataset_random_subset":
                subset = args[i + 1]
                break
        subset = subset or 100

    training = run.summary.get("train/epoch") is not None
    if not training:
        subset = 0

    return model_name, int(subset)


def get_wandb_run_dataframes(
    wandb_runs: Dict[str, wandb.apis.public.Run],
    weighted_preference: bool,
) -> Dict[str, Optional[pd.DataFrame]]:
    """Check whether each wandb run has the artifact 'human_evaluations_df', and if so return it."""
    results = {}

    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
    api = wandb.Api()

    for run_id, run in wandb_runs.items():
        try:
            artifact = api.artifact(f"ucl-dark/rlvsil-main/{art_name(run_id, weighted_preference)}:latest")
            if not os.path.exists(artifact.file()):
                artifact.download()
            results[run_id] = pd.read_csv(artifact.file())
        except wandb.errors.CommError:
            results[run_id] = None
    return results


def get_results_from_df(
    df: pd.DataFrame,
    wandb_runs: Dict[str, wandb.apis.public.Run],
    include_zeros: bool = False,
) -> Tuple[Dict[str, float], Dict[str, int], int]:
    none_count = df["neutral"].sum()

    run_id_to_preference_score = defaultdict(list)
    run_id_to_comparison_count: dict = defaultdict(int)
    for run_id in wandb_runs.keys():
        df_filter = df["run_id"] == run_id
        if not include_zeros:
            df_filter = df_filter & ~df["neutral"]
        run_id_subdf = df[df_filter]
        run_id_to_preference_score[run_id] = run_id_subdf["preference_score"].mean()
        run_id_to_comparison_count[run_id] = len(run_id_subdf)

    return run_id_to_preference_score, run_id_to_comparison_count, none_count


@click.command()
@click.argument("scale_api_key")
@click.argument("scale_project_name")
@click.argument("scale_batch_names_str")
@click.option("--update_on_wandb", default=False, help="Update wandb with calculated statistics.")
@click.option(
    "--weighted_preference", default=False, help="Use weighted preferences in calculating preferences."
)
@click.option(
    "--include_zeros", default=False, help="Include neutral evaluations in calculation of preference score."
)
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
def analyse_scale_batch(
    scale_api_key: str,
    scale_project_name: str,
    scale_batch_names_str: str,
    update_on_wandb: bool,
    wandb_update_strategy: str,
    weighted_preference: bool,
    include_zeros: bool,
    df_save_file: Optional[str],
    wandb_df_save_file: Optional[str],
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
    corrupted_results: List[scaleapi.Task] = []
    none_count = 0
    for task in tasks:
        try:
            task_res, corrupted_res = process_task(
                task, weighted_preference=weighted_preference, include_zeros=include_zeros
            )
        except Exception as e:
            print(e)
            print("")
        result.extend(task_res)
        corrupted_results.extend(corrupted_res)

    click.echo(f"Got {len(result)} results from scale tasks.")
    click.echo(f"Got {len(corrupted_results)} corrupted results from scale tasks, ignoring.")

    results_df = pd.DataFrame(result)
    wandb_runs = get_wandb_runs(set(results_df["run_id"]))

    run_id_to_model_name_and_dataset = {
        run_id: get_model_name_and_dataset(wandb_run) for run_id, wandb_run in wandb_runs.items()
    }
    # Add information about model_name and subset to results_df
    model_name_and_dataset = [run_id_to_model_name_and_dataset[run_id] for run_id in results_df["run_id"]]
    results_df["model_name"] = [x[0] for x in model_name_and_dataset]
    results_df["dataset"] = [x[1] for x in model_name_and_dataset]

    run_id_to_preference_score, run_id_to_comparison_count, none_count = get_results_from_df(
        results_df, wandb_runs, include_zeros
    )

    click.echo("Results:")
    for run_id in wandb_runs.keys():
        model_name, subset = run_id_to_model_name_and_dataset[run_id]
        click.echo(
            f"{run_id}: {model_name} - {subset}"
            f": {run_id_to_preference_score[run_id]} ({run_id_to_comparison_count[run_id]})"
        )
    click.echo(f"Number of comparisons with no preference: {none_count}")
    click.echo()
    wandb_run_dfs = get_wandb_run_dataframes(wandb_runs, weighted_preference)
    if update_on_wandb:
        wandb_run_dfs = update_results_on_wandb(
            wandb_run_dfs,
            wandb_runs,
            wandb_update_strategy,
            results_df,
            run_id_to_model_name_and_dataset,
            weighted_preference,
        )

    if df_save_file:
        results_df.to_csv(df_save_file, index=False)
        click.echo(f"Saved results dataframe to {df_save_file}")

    if wandb_df_save_file:
        wandb_df = pd.concat(wandb_run_dfs.values())
        wandb_df.to_csv(wandb_df_save_file, index=False)
        click.echo(f"Saved wandb dataframe to {wandb_df_save_file}")


def update_wandb_artifacts(
    wandb_run_dfs: Dict[str, pd.DataFrame],
    wandb_runs: Dict[str, wandb.apis.public.Run],
    weighted_preference: bool,
):
    """Update wandb artifacts with new human evaluations."""
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
    api = wandb.Api()
    for run_id, df in wandb_run_dfs.items():
        file_name = f"data/{art_name(run_id, weighted_preference)}.csv"
        df.to_csv(file_name, index=False)  # type: ignore
        artifact = wandb.Artifact(
            name=art_name(run_id, weighted_preference),
            type="human_eval",
        )
        artifact.add_file(file_name)
        try:
            # Check whether artifact already exists
            api.artifact(f"ucl-dark/rlvsil-main/{art_name(run_id, weighted_preference)}:latest")
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
    run_id_to_model_name_and_dataset: Dict[str, Tuple[str, int]],
    weighted_preference: bool,
) -> Dict[str, pd.DataFrame]:
    click.echo("Updating wandb runs with results.")
    score_name = "human_evaluation_score" if not weighted_preference else "weighted_human_evaluation_score"
    count_name = "human_evaluation_count" if not weighted_preference else "weighted_human_evaluation_count"
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
        update_wandb_artifacts(wandb_run_dfs, wandb_runs, weighted_preference)

    for run_id, run in wandb_runs.items():
        run_id_to_preference_score, run_id_to_comparison_count, none_count = get_results_from_df(
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


if __name__ == "__main__":
    analyse_scale_batch()
