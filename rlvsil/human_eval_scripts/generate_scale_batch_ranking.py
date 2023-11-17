"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import html
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import click
import pandas as pd
import scaleapi
import wandb


def make_fields(num_items_to_rank: int) -> list:
    return [
        {
            "type": "ranking_order",
            "field_id": "Rank Summaries",
            "title": "Rank the following summaries from best to worst.",
            "num_items_to_rank": num_items_to_rank,
            "first_label": "Best",
            "last_label": "Worst",
            "hint": (
                "Remember to consider whether the summary captures the essence of the post or article,"
                " is clear, accurate, serves the same purpose as the post or article and is concise."
            ),
            "options_flexible": False,
            "num_options": num_items_to_rank,
            "positions_flexible": False,
            "required": True,
            "option_label": "Summary",
        }
    ]


def get_wandb_run_dataset(run_id: str, table_name: str, path: str = "ucl-dark/rlvsil-main") -> pd.DataFrame:
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"

    api = wandb.Api()

    artifact = api.artifact(f"{path}/run-{run_id}-{table_name}:latest", type="run_table")

    # Check whether artiface.file() already exists
    if not os.path.exists(artifact.file()):
        artifact.download()
    else:
        click.echo(f"File for run {run_id} already exists.")

    with open(artifact.file()) as f:
        res = json.load(f)

    df = pd.DataFrame(res["data"], columns=res["columns"])
    df.rename(columns={"input": "article"}, inplace=True)
    return df


def merge_wandb_run_dataframes(datasets: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple dataframes into one."""
    ds0 = datasets[0]
    merged = False
    for i, ds in enumerate(datasets[1:]):
        ds0 = ds0.merge(
            ds[["article", "response"]],
            on="article",
            how="inner",
            suffixes=[f" {i}", ""] if (i + 2 < len(datasets)) else [f" {i}", f" {i + 1}"],
        )
        merged = True
    if not merged:
        ds0 = ds0.rename(columns={"response": "response 0"})
    return ds0


def format_entry(post: str, summaries: List[str]) -> str:
    tldr = "TL;DR:"
    if post[-len(tldr) :] == tldr:
        post = post[: -len(tldr)]
    post_title = "Post" if "subreddit" in post.lower() else "Article"
    return (
        f"# {post_title}\n\n{html.escape(post.replace('</s>', ''), quote=False)}\n\n"
        + "# Summaries\n\n"
        + "\n\n".join(
            [
                f"## Summary {i + 1}\n\n{html.escape(summaries[i].replace('</s>', ''), quote=False)}"
                for i in range(len(summaries))
            ]
        )
    )


def make_test_data(row: pd.Series, n_responses: int, run_ids: List[str], duplicate_rows: List) -> dict:
    indices = list(range(n_responses))
    summary_to_run_ids: Dict[str, List[str]] = defaultdict(list)

    # TODO: how to do this properly
    reference = row["reference"]
    summary_to_run_ids[reference].append("reference")

    for i, summary in enumerate(row[[f"response {i}" for i in indices]]):
        summary_to_run_ids[summary].append(run_ids[i])

    if len(summary_to_run_ids) < len(run_ids):
        # click.echo(f"Found matches in generations from run ids: {queried_run_ids}")
        # click.echo(queried_summaries)
        duplicate_rows.append(row)

    summaries = set(summary_to_run_ids.keys())
    summaries.add(reference)

    # randomly shuffle summaries
    random_summaries = list(summaries)
    random.shuffle(random_summaries)

    # Record run ids of each option
    shuffled_run_ids = [summary_to_run_ids[summary] for summary in random_summaries]

    return {
        "attachments": [{"type": "text", "content": format_entry(row["article"], random_summaries)}],
        "metadata": {
            "run_ids": shuffled_run_ids,
            "created": datetime.now().isoformat(),
        },
        "fields": make_fields(len(random_summaries)),
    }


def transform_df_into_scale_data(df: pd.DataFrame, n_responses: int, run_ids: List[str]) -> List[dict]:
    """Takes a dataframe w/ post and summaries and transforms it into a list of dicts for scale."""
    duplicate_rows: list = []
    result = [make_test_data(row, n_responses, run_ids, duplicate_rows) for _, row in df.iterrows()]
    click.echo(f"Found {len(duplicate_rows)} duplicate rows")
    return result


@click.command()
@click.argument("wandb_run_ids")
@click.argument("scale_api_key")
@click.argument("scale_project_name")
@click.argument("scale_batch_name")
@click.option("--task_limit", default=5)
@click.option("--task_index_start", default=0)
@click.option("--task_log_interval", default=100)
@click.option("--self_label_batch", default=True)
@click.option("--calibration_batch", default=False)
@click.option("--task_tags", default="rlvsil", help="Comma separated list of tags")
@click.option("--output_data_to_csv", default=False, help="Output data to csv rather than uploading to scale")
@click.option("--allow_existing_batch", default=True, help="Allow existing batch")
@click.option("--finalize", default=False, help="Finalize the batch after uploading")
@click.option(
    "--data_table_names", default="testtext_table", help="Which wandb table to use (train, eval, test)"
)
def generate_scale_batch(
    wandb_run_ids: str,
    scale_api_key: str,
    scale_project_name: str,
    scale_batch_name: str,
    task_limit: int,
    task_index_start: int,
    task_log_interval: int,
    self_label_batch: bool,
    calibration_batch: bool,
    task_tags: str,
    output_data_to_csv: bool,
    allow_existing_batch: bool,
    finalize: bool,
    data_table_names: str,
):
    """Gets data from wandb run, formats it into a batch to send to Scale AI data labelling service"""

    run_ids = wandb_run_ids.split(",")
    table_names = data_table_names.split(",")

    if len(table_names) == 1:
        table_names = table_names * len(run_ids)

    click.echo(f"Getting data from wandb runs {run_ids}...")
    datasets = [
        get_wandb_run_dataset(run_id, data_table_name)
        for run_id, data_table_name in zip(run_ids, table_names)
    ]

    click.echo("Merging dataframes...")
    df = merge_wandb_run_dataframes(datasets)

    click.echo("Transforming dataframe into scale data...")
    scale_batch_data = transform_df_into_scale_data(df, n_responses=len(datasets), run_ids=run_ids)
    click.echo(f"Generated {len(scale_batch_data)} data points.")

    click.echo(f"Truncating to {task_limit} data points, starting from {task_index_start}.")
    scale_batch_data = scale_batch_data[task_index_start : task_limit + task_index_start]

    if output_data_to_csv:
        click.echo("Saving data to csv...")
        click.echo("Note, this will not preserve the `fields` information")
        # Save scale_batch_data to a csv file
        df = pd.DataFrame(
            {
                "attachments_json": [json.dumps(row["attachments"]) for row in scale_batch_data],
                "metadata": [json.dumps(row["metadata"]) for row in scale_batch_data],
            }
        )
        df.to_csv(f"{scale_batch_name}.csv")
        click.echo(f"Saved data to {scale_batch_name}.csv")
    else:
        click.echo("Creating batch from api...")
        scale_cli = scaleapi.ScaleClient(api_key=scale_api_key)
        scale_proj = scale_cli.get_project(scale_project_name)
        batch_operation = "Created"
        try:
            batch = scale_cli.create_batch(
                project=scale_proj.name,
                batch_name=scale_batch_name,
                self_label_batch=self_label_batch,
                calibration_batch=calibration_batch,
            )
        except scaleapi.exceptions.ScaleDuplicateResource:
            if allow_existing_batch:
                click.echo(f"Batch {scale_batch_name} already exists, using existing batch")
                batch = scale_cli.get_batch(scale_batch_name)
                batch_operation = "Updated"
            else:
                raise

        tasks = []
        click.echo("Creating tasks...")
        for i, task in enumerate(scale_batch_data):
            tasks.append(
                scale_cli.create_task(
                    scaleapi.TaskType.TextCollection,
                    project=scale_proj.name,
                    project_param_version=scale_proj.version,
                    batch=batch.name,
                    tags=task_tags.split(","),
                    **task,
                )
            )
            if (i + 1) % task_log_interval == 0:
                click.echo(f"Created {i + 1} tasks...")

        click.echo(f"{batch_operation} batch {batch.name} with {len(tasks)} tasks")
        click.echo(batch)
        if finalize:
            click.echo("Finalizing batch...")
            batch.finalize()
            click.echo(f"Finalized batch {batch.name}")
        else:
            click.echo(f"Batch {batch.name} is not finalized")


if __name__ == "__main__":
    generate_scale_batch()
