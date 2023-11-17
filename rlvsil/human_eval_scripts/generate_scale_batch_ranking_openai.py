"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import html
import random
import json
from datetime import datetime
from typing import List

import click
import pandas as pd
import scaleapi
from datasets import load_dataset


def make_input_example_tldr(post, title, subreddit):
    return f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR:"


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
                "Remember to consider whether the summary captures the essence of the post,"
                " is clear, accurate, serves the same purpose as the post and is concise."
            ),
            "options_flexible": False,
            "num_options": num_items_to_rank,
            "positions_flexible": False,
            "required": True,
            "option_label": "Summary",
        }
    ]


def get_openai_dataset(response_limit: int) -> pd.DataFrame:
    """Get the OpenAI preference dataset, format it correctly into a dataframe and return it."""
    ds = load_dataset("UCL-DARK/openai-tldr-summarisation-preferences", use_auth_token=True)["train"]
    ds = ds.filter(
        lambda example: len([s["policy"] for s in example["summaries"] if s["policy"] == "ref"]) > 0,
        batched=False,
    )
    match_post = ds[0]["info"]["post"]
    results = []
    inner_results = {
        "article": make_input_example_tldr(
            ds[0]["info"]["post"], ds[0]["info"]["title"], ds[0]["info"]["subreddit"]
        ),
        "responses": [],
        "preferred": [],
        "ref": [s["text"] for s in ds[0]["summaries"] if s["policy"] == "ref"][0],
    }
    for row in ds:
        if row["info"]["post"] != match_post:
            if len(inner_results["responses"]) == response_limit:
                for i, res in enumerate(inner_results["responses"][:response_limit]):
                    inner_results[f"response {i}"] = res
                del inner_results["responses"]
                inner_results["preferred"] = inner_results["preferred"][:response_limit]
                results.append(copy.deepcopy(inner_results))

            match_post = row["info"]["post"]
            inner_results = {
                "article": make_input_example_tldr(
                    row["info"]["post"], row["info"]["title"], row["info"]["subreddit"]
                ),
                "responses": [],
                "preferred": [],
                "ref": [s["text"] for s in row["summaries"] if s["policy"] == "ref"][0],
            }

        inner_results["responses"].append([s["text"] for s in row["summaries"] if s["policy"] != "ref"][0])
        inner_results["preferred"].append(row["summaries"][row["choice"]]["policy"] != "ref")

    return pd.DataFrame(results)


def format_entry(post: str, summaries: List[str]) -> str:
    tldr = "TL;DR:"
    if post[-len(tldr) :] == tldr:
        post = post[: -len(tldr)]
    return (
        f"# Post\n\n{html.escape(post.replace('</s>', ''), quote=False)}\n\n"
        + "# Summaries\n\n"
        + "\n\n".join(
            [
                f"## Summary {i + 1}\n\n{html.escape(summaries[i].replace('</s>', ''), quote=False)}"
                for i in range(len(summaries))
            ]
        )
    )


def make_test_data(row: pd.Series) -> dict:
    num_choices = len(row["preferred"])
    summaries = row[[f"response {i}" for i in range(num_choices)]].tolist()
    ref_index = random.randint(0, num_choices - 1)
    summaries = summaries[:ref_index] + [row["ref"]] + summaries[ref_index:]

    return {
        "attachments": [{"type": "text", "content": format_entry(row["article"], summaries)}],
        "metadata": {
            "preferred": row["preferred"],
            "created": datetime.now().isoformat(),
            "ref": row["ref"],
            "ref_index": ref_index,
        },
        "fields": make_fields(len(summaries)),
    }


def transform_df_into_scale_data(df: pd.DataFrame) -> List[dict]:
    """Takes a dataframe w/ post and summaries and transforms it into a list of dicts for scale."""
    result = [make_test_data(row) for _, row in df.iterrows()]
    return result


@click.command()
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
    """Gets data from OpenAI dataset, formats it into a batch to send to Scale AI data labelling service"""

    click.echo("Getting data from OpenAI dataset and formatting it")
    df = get_openai_dataset(response_limit=3)
    click.echo("Generated dataframe with shape: " + str(df.shape))

    click.echo("Randomly Sampling down to limit")
    indices = random.sample(range(len(df)), task_limit)
    df = df.iloc[indices]

    click.echo("Transforming dataframe into scale data...")
    scale_batch_data = transform_df_into_scale_data(df)
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
