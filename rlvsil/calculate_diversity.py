"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""Calculate diversity metrics for wandb runs and accompanying text tables."""
import random
import json
import os
import typing as t
from pprint import pprint

import click
import pandas as pd

import wandb
from diversity import DEFAULT_CONFIGS, calculate_diversity_metrics


def get_artifact_json(path: str, api: wandb.Api) -> t.Tuple[t.List[t.List[t.Any]], t.List[str]]:
    artifact = api.artifact(path, type="run_table")

    # Check whether artiface.file() already exists
    if not os.path.exists(artifact.file()):
        artifact.download()
    else:
        click.echo(f"File {path} already exists.")

    with open(artifact.file()) as f:
        res = json.load(f)

    return res["data"], res["columns"]


def get_wandb_table_outputs(
    run_ids: t.List[str],
    table_name: str,
    path: str = "ucl-dark/rlvsil-main",
    use_references: bool = False,
    no_table_ids: bool = False,
    max_tables: int = 16,
    keep_rewards: bool = False,
) -> t.List[t.Union[t.List[str], t.Tuple[t.List[str], t.List[float]]]]:
    """Get outputs from wandb table."""
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
    api = wandb.Api()

    run_id_idx = 0
    run_id = run_ids[run_id_idx]
    click.echo(f"Getting outputs from wandb for run_id, run_table: {run_id}, {table_name}")
    jsons: t.List[t.List[t.Any]] = []
    table_id = 0
    tables_got = 0
    if no_table_ids:
        data, columns = get_artifact_json(f"{path}/run-{run_id}-{table_name}:latest", api)
        jsons.extend(data)
        click.echo("Got single table as no ids.")
    else:
        while True:
            try:
                data, columns = get_artifact_json(
                    f"{path}/run-{run_id}-{table_name}text_table_{table_id}:latest", api
                )
                jsons.extend(data)
                table_id += 1
                tables_got += 1
                if tables_got >= max_tables:
                    click.echo(f"All {max_tables} Tables Found.")
                    break
                if tables_got >= 1 and use_references:
                    click.echo("1 Table Found, all we need when using references.")
                    break
                continue
            except wandb.errors.CommError:
                click.echo(
                    f"No more tables found for run_id, run_table, at id: {run_id}, {table_name}, {table_id}"
                )

            try:
                run_id_idx += 1
                run_id = run_ids[run_id_idx]
                table_id = 0
                click.echo(f"Getting outputs from wandb for run_id, run_table: {run_id}, {table_name}")
            except IndexError:
                click.echo(f"All run_ids found, but only found {tables_got} tables.")
                break

    df = pd.DataFrame(data=jsons, columns=columns)

    key = "reference" if use_references else "response"
    responses = df.groupby("input").agg(lambda x: list(x))[key].tolist()
    if not keep_rewards:
        return responses

    rewards = df.groupby("input").agg(lambda x: list(x))["rewards"].tolist()
    return list(zip(responses, rewards))


def load_sample_decodes_json(path: str) -> t.List[t.List[str]]:
    """Load sample decodes json."""
    with open(path) as f:
        res = json.load(f)

    res = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in res]

    return res


def simulate_bon(
    outputss: t.List[t.Tuple[t.List[str], t.List[float]]],
    simulate_bon_for_per_input: int,
    simulate_bon_for_per_input_n_outputs: int,
) -> t.List[t.List[str]]:
    """Simulate best of N for per input metrics."""
    res = []
    for outputs, rewards in outputss:
        outputs_rewardss = list(zip(outputs, rewards))
        output_res = []
        for _ in range(simulate_bon_for_per_input_n_outputs):
            # Sample simulate_bon_for_per_input outputs without replacement
            outputs_rewards = random.sample(outputs_rewardss, simulate_bon_for_per_input)
            outputs_rewards = sorted(outputs_rewards, key=lambda x: x[1], reverse=True)
            output = outputs_rewards[0][0]
            output_res.append(output)
        res.append(output_res)

    return res


@click.command()
@click.option("--run_ids")
@click.option("--decodes", default=None, help="Path to decodes.json.")
@click.option("--model_name", default=None, help="Name to log to wandb for this diversity measure")
@click.option("--table_name", default="best_of_N", help="Name of table to load outputs from.")
@click.option("--diversity_metrics", default="all", help="list of diversity metrics to use.")
@click.option(
    "--limit_outputs",
    default=None,
    type=int,
    help="Whether to put a limit on number of outputs processed (useful for debugging)",
)
@click.option("--no_per_input", is_flag=True, help="Whether to calculate per input diversity metrics.")
@click.option("--no_overall_input", is_flag=True, help="Whether to calculate overall diversity metrics.")
@click.option("--no_table_ids", is_flag=True, help="Whether tables are ided or not.")
@click.option("--sample_overall", is_flag=True, help="Whether to sample in calculating overall metrics.")
@click.option(
    "--use_references", is_flag=True, help="Whether to use reference outputs rather than model outputs."
)
@click.option("--max_tables", default=16, type=int, help="Max number of tables to get.")
@click.option(
    "--simulate_bon_for_per_input",
    default=None,
    type=int,
    help="Value of N to simulate from wider N for per_input metrics.",
)
@click.option(
    "--simulate_bon_for_per_input_n_outputs",
    default=32,
    type=int,
    help="Number of outputs to simulate from wider N for per_input metrics.",
)
@click.option("wandb_project", default="rlvsil-main", help="Wandb project to log to.")
@click.option("wandb_entity", default="ucl-dark", help="Wandb entity to log to.")
def main(
    run_ids: t.Optional[str],
    decodes: t.Optional[str],
    model_name: t.Optional[str],
    table_name: str,
    diversity_metrics: str,
    limit_outputs: t.Optional[int],
    no_per_input: bool = False,
    no_overall_input: bool = False,
    no_table_ids: bool = False,
    sample_overall: bool = False,
    use_references: bool = False,
    max_tables: int = 16,
    simulate_bon_for_per_input: t.Optional[int] = None,
    simulate_bon_for_per_input_n_outputs: int = 8,
    wandb_project: str = "rlvsil-main",
    wandb_entity: str = "ucl-dark",
):
    """Calculate diversity metrics for wandb runs and accompanying text tables."""
    if no_table_ids and not (no_overall_input or no_per_input):
        raise ValueError(
            "No table ids only works with just across-input metrics (i.e. no overall and no per input.)"
        )
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        tags=["diversity"],
        config={
            "table_name": table_name,
            "run_id": run_ids,
            "diversity_metrics": diversity_metrics,
            "limit_outputs": limit_outputs,
            "no_per_input": no_per_input,
            "sample_overall": sample_overall,
            "no_overall_input": no_overall_input,
            "no_table_ids": no_table_ids,
            "model_name": model_name,
            "use_references": use_references,
            "decodes": decodes,
            "max_tables": max_tables,
            "simulate_bon_for_per_input": simulate_bon_for_per_input,
            "simulate_bon_for_per_input_n_outputs": simulate_bon_for_per_input_n_outputs,
        },
    )

    if run_ids is not None:
        click.echo("Getting outputs from wandb")
        outputss = get_wandb_table_outputs(
            run_ids.split(","),
            table_name,
            use_references=use_references,
            no_table_ids=no_table_ids,
            max_tables=max_tables,
            keep_rewards=simulate_bon_for_per_input is not None,
            path=f"{wandb_entity}/{wandb_project}",
        )
        log_prefix = "run_id, run_table: " + run_ids + ", " + table_name
    elif decodes is not None:
        click.echo("Loading outputs from decodes.json")
        outputss = load_sample_decodes_json(decodes)
        log_prefix = "decodes: " + decodes
    else:
        raise ValueError("Must provide either run_id or decodes.json")

    if simulate_bon_for_per_input:
        click.echo("Simulating BON for per input metrics.")
        outputss = simulate_bon(outputss, simulate_bon_for_per_input, simulate_bon_for_per_input_n_outputs)

    diversity_metrics_config = DEFAULT_CONFIGS.copy()
    if diversity_metrics != "all":
        diversity_metrics_config = {
            k: v for k, v in diversity_metrics_config.items() if k in diversity_metrics
        }
    print(diversity_metrics_config)

    print("Calculating for", log_prefix)
    run_results = calculate_diversity_metrics(
        outputss[:limit_outputs], diversity_metrics_config, no_per_input, no_overall_input, sample_overall
    )
    print("Outputs for", log_prefix)
    pprint(run_results)
    run_results["run_id"] = run_ids
    run_results["model_name"] = model_name
    run_results["table_name"] = table_name
    wandb.log(run_results)
    wandb.finish()


if __name__ == "__main__":
    main()
