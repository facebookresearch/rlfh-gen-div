"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""Requeues all tasks in a given batch that have corrupted model outputs."""
from datetime import datetime

import click
import pandas as pd
import scaleapi


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


def get_tasks_from_batch(scale_api_key: str, batch_name: str, project_name: str) -> list:
    """Get all tasks in a batch."""
    api = scaleapi.ScaleClient(api_key=scale_api_key)
    tasks = api.get_tasks(batch_name=batch_name, project_name=project_name)
    return tasks


def filter_corrupted_tasks(tasks: list) -> list:
    """Filter out tasks that have corrupted model outputs."""
    corrupted_tasks = []
    for task in tasks:
        corrupted_subtasks = []
        corrupted_refs = []
        corrupted_run_ids = []
        metadata = task.metadata
        for attachment, refs, run_ids in zip(
            task.params["attachments"], metadata["references"], metadata["run_ids"]
        ):
            if "\n" in attachment["content"].split("|\n|")[-1]:
                corrupted_subtasks.append(attachment)
                corrupted_refs.append(refs)
                corrupted_run_ids.append(run_ids)

        for i, corrupted_subtask in enumerate(corrupted_subtasks):
            corrupted_subtask.update(forms=[f"form_query_{i}"])
            corrupted_content = corrupted_subtask["content"].split("|\n|")
            corrupted_content[-1] = corrupted_content[-1].replace("\n", "<br>")
            corrupted_subtask["content"] = "|\n|".join(corrupted_content)

        if len(corrupted_subtasks) > 0:
            corrupted_tasks.append(
                dict(
                    attachments=corrupted_subtasks,
                    metadata=dict(
                        references=corrupted_refs,
                        run_ids=corrupted_run_ids,
                        created=datetime.now().isoformat(),
                    ),
                    fields=[make_field(i) for i in range(len(corrupted_subtasks))],
                    tags=task.tags,
                )
            )

    return corrupted_tasks


@click.argument("new_scale_batch_name")
@click.argument("scale_batch_name")
@click.argument("scale_project_name")
@click.argument("scale_api_key")
@click.option("--dry_run", type=bool, default=True)
@click.option("--task_log_interval", default=100)
@click.option("--finalize", type=bool, default=True)
@click.command()
def redo_batch(
    scale_api_key: str,
    scale_project_name: str,
    scale_batch_name: str,
    new_scale_batch_name: str,
    dry_run: bool,
    task_log_interval: int,
    finalize: bool,
):
    tasks = get_tasks_from_batch(scale_api_key, scale_batch_name, scale_project_name)
    corrupted_tasks = filter_corrupted_tasks(tasks)

    if dry_run:
        click.echo("Dry run")
        click.echo(f"Found {len(corrupted_tasks)} corrupted tasks")
        return

    scale_cli = scaleapi.ScaleClient(api_key=scale_api_key)

    try:
        new_batch = scale_cli.create_batch(
            project=scale_project_name, batch_name=new_scale_batch_name, self_label_batch=False
        )
    except scaleapi.exceptions.ScaleDuplicateResource:
        click.echo(f"Batch {new_scale_batch_name} already exists, using existing batch")
        new_batch = scale_cli.get_batch(new_scale_batch_name)

    tasks = []
    click.echo("Creating tasks...")
    for i, task in enumerate(corrupted_tasks):
        tasks.append(
            scale_cli.create_task(
                scaleapi.TaskType.TextCollection,
                project=scale_project_name,
                batch=new_batch.name,
                **task,
            )
        )
        if (i + 1) % task_log_interval == 0:
            click.echo(f"Created {i + 1} tasks...")

    click.echo(new_batch)
    if finalize:
        click.echo("Finalizing batch...")
        new_batch.finalize()
        click.echo(f"Finalized batch {new_batch.name}")
    else:
        click.echo(f"Batch {new_batch.name} is not finalized")


if __name__ == "__main__":
    redo_batch()
