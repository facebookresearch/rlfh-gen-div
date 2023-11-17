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
from datetime import datetime
from typing import Dict, List

import click
import pandas as pd
import scaleapi
import wandb
from markdownTable import markdownTable


def make_field(field_number: int) -> dict:
    return {
        "type": "form",
        "field_id": f"form_query_{field_number}",
        "title": f"Comparison {field_number}",
        "fields": [
            {
                "type": "number",
                "use_slider": True,
                "field_id": f"Summary Rating {field_number}",
                "title": "Summary Rating",
                "min": -4,
                "max": 4,
                "hint": (
                    "The scale is: \ndefinitely A, very likely A, likely A, possibly A, uncertain"
                    ", possibly B, likely B, very likely B, definitely B."
                ),
                "digits": 1,
                "step": 1,
                "required": True,
                "description": "Which summary is preferred, and how confident are you in that summary?",
                "prefix": "Summary A",
                "suffix": "Summary B",
                "min_responses_required": 1,
                "max_responses_required": 1,
                "extra_content": "Confidence",
            }
        ],
    }


def get_wandb_run_dataset(run_id: str, path: str = "ucl-dark/rlvsil-main") -> pd.DataFrame:
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"

    api = wandb.Api()

    artifact = api.artifact(f"{path}/run-{run_id}-testtext_table:latest", type="run_table")

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
    for i, ds in enumerate(datasets[1:]):
        ds0 = ds0.merge(
            ds[["article", "response"]],
            on="article",
            how="inner",
            suffixes=[f" {i}", ""] if (i + 2 < len(datasets)) else [f" {i}", f" {i + 1}"],
        )
    return ds0


def format_entry(post: str, sum_a: str, sum_b: str) -> str:
    tldr = "TL;DR:"
    if post[-len(tldr) :] == tldr:
        post = post[: -len(tldr)]
    summary_table = (
        markdownTable([{"Summary A": sum_a.replace("\n", "<br>"), "Summary B": sum_b.replace("\n", "<br>")}])
        .setParams(row_sep="markdown", quote=False)
        .getMarkdown()
    )
    # return f"## Post\n\n{post}\n\n## Summary A\n\n{sums[0]}\n\n## Summary B\n\n{sums[-1]}"
    return (
        f"## Post\n\n{html.escape(post.replace('</s>', ''), quote=False)}\n\n## Summaries\n\n{summary_table}"
    )


def make_test_data(row: pd.Series, n_responses: int, run_ids: List[str], duplicate_rows: List) -> dict:
    test_data: List[dict] = []
    i = 1
    indices = list(range(n_responses))
    run_ids = run_ids.copy()
    # shuffle run_ids and indices in the same way
    random.Random(42).shuffle(run_ids)
    random.Random(42).shuffle(indices)
    references = []
    queried_summaries: Dict[str, int] = dict()
    queried_run_ids: List[List[str]] = []
    adjustment = 0
    for i, summary in enumerate(row[[f"response {i}" for i in indices]]):
        # Don't query the same summary twice
        if summary in queried_summaries:
            queried_run_ids[queried_summaries[summary]].append(run_ids[i])
            adjustment += 1
            continue
        queried_summaries[summary] = i - adjustment
        queried_run_ids.append([run_ids[i]])

        # flip a coin to decide which summary is A and which is B
        if random.random() > 0.5:
            summary_a = summary
            summary_b = row["reference"]
            reference = 1
        else:
            summary_a = row["reference"]
            summary_b = summary
            reference = 0

        test_data.append(
            {
                "type": "text",
                "content": format_entry(row["article"], summary_a, summary_b),
                "forms": [f"form_query_{len(test_data)}"],
            }
        )
        references.append(reference)

    if len(queried_run_ids) < len(run_ids):
        # click.echo(f"Found matches in generations from run ids: {queried_run_ids}")
        # click.echo(queried_summaries)
        duplicate_rows.append(row)
    return {
        "attachments": test_data,
        "metadata": {
            "references": references,
            "run_ids": queried_run_ids,
            "created": datetime.now().isoformat(),
        },
        "fields": [make_field(i) for i in range(len(test_data))],
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
):
    """Gets data from wandb run, formats it into a batch to send to Scale AI data labelling service"""

    # Get data from wandb run
    run_ids = wandb_run_ids.split(",")

    click.echo(f"Getting data from wandb runs {run_ids}...")
    datasets = [get_wandb_run_dataset(run_id) for run_id in run_ids]

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
                project=scale_proj.name, batch_name=scale_batch_name, self_label_batch=self_label_batch
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
