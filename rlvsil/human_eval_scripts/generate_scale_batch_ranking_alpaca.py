"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import html
import json
import random
from datetime import datetime
from typing import List

import click
import pandas as pd
import scaleapi
from datasets import load_dataset


def make_fields(num_items_to_rank: int) -> list:
    return [
        {
            "type": "ranking_order",
            "field_id": "Rank Responses",
            "title": "Rank the following responses from best to worst.",
            "num_items_to_rank": num_items_to_rank,
            "first_label": "Best",
            "last_label": "Worst",
            "options_flexible": False,
            "num_options": num_items_to_rank,
            "positions_flexible": False,
            "required": True,
            "option_label": "Response",
        }
    ]


def format_entry(instruction: str, responses: List[str]) -> str:
    return (
        f"# Instruction\n\n{html.escape(instruction.replace('</s>', ''), quote=False)}\n\n"
        + "# Responses\n\n"
        + "\n\n".join(
            [
                f"## Response {i + 1}\n\n{html.escape(responses[i].replace('</s>', ''), quote=False)}"
                for i in range(len(responses))
            ]
        )
    )


def make_test_data(row: pd.Series) -> dict:
    reference = row["reference"]
    output = row["output"]

    # randomly shuffle responses
    random_responses = [reference, output]
    random.shuffle(random_responses)

    # Record run ids of each option
    reference_idx = random_responses.index(reference)

    return {
        "attachments": [{"type": "text", "content": format_entry(row["instruction"], random_responses)}],
        "metadata": {
            "reference_idx": reference_idx,
            "created": datetime.now().isoformat(),
            "model_name": row["generator_model"],
            "reference_name": row["generator_reference"],
            "dataset": row["dataset"],
        },
        "fields": make_fields(len(random_responses)),
    }


def transform_df_into_scale_data(df: pd.DataFrame) -> List[dict]:
    """Takes a dataframe w/ instruction and responses and transforms it into a list of dicts for scale."""
    result = [make_test_data(row) for _, row in df.iterrows()]
    return result


def load_outputs(outputs_path: str) -> pd.DataFrame:
    """Load outputs from a path, either csv or json."""
    if outputs_path.endswith(".csv"):
        return pd.read_csv(outputs_path)
    elif outputs_path.endswith(".json"):
        with open(outputs_path) as f:
            return pd.DataFrame(json.load(f))
    else:
        raise ValueError(f"Unknown file type for {outputs_path}")


@click.command()
@click.argument("scale_api_key")
@click.argument("scale_project_name")
@click.argument("scale_batch_name")
@click.option("--model_outputs", help="Path to model outputs")
@click.option("--reference_outputs", help="Path to reference outputs")
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
    model_outputs: str,
    reference_outputs: str,
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
    """Gets data from json files, formats it into a batch to send to Scale AI data labelling service"""
    # load model outputs
    model_outputs_df = load_outputs(model_outputs)
    if not reference_outputs:
        # Get reference outputs from dataset
        dataset_name = model_outputs_df["dataset"].iloc[0]
        if "ALPACAEVAL_REFERENCE_OUTPUTS" in dataset_name:
            dataset_name = "tatsu-lab/alpaca_eval"
        reference_outputs_dict = load_dataset(dataset_name)
        reference_outputs_df = list(reference_outputs_dict.values())[0].to_pandas()
    else:
        reference_outputs_df = load_outputs(reference_outputs)

    # merge model outputs and reference outputs
    df = pd.merge(
        model_outputs_df,
        reference_outputs_df,
        on=["instruction"],
        suffixes=("_model", "_reference"),
    )
    df.rename(
        columns={"output_model": "output", "output_reference": "reference", "dataset_model": "dataset"},
        inplace=True,
    )

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
                batch = scale_cli.get_batch(scale_batch_name)
                if batch.project != scale_proj.name:
                    raise ValueError(
                        f"Batch {scale_batch_name} already exists, but is"
                        f"associated with project {batch.project}. "
                        "Please delete the batch or change the batch name."
                    )
                batch_operation = "Updated"
                click.echo(f"Batch {scale_batch_name} already exists, using existing batch")
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
