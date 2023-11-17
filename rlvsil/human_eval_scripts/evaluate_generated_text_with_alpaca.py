"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os
import sys
import typing as t

import click
import pandas as pd
import wandb
from alpaca_eval import main as alpaca_main
from datasets import load_dataset

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))
from dataset.summarisation_formatting import (make_input_example_cnndm,
                                              make_input_example_tldr)

DATASET_TLDR = "tldr"
DATASET_CNNDM = "cnndm"


DATASET_TO_EVAL_CONFIGS = {
    DATASET_CNNDM: "cnndm_gpt4_eval",
    DATASET_TLDR: "tldr_gpt4_eval",
}


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


def get_wandb_table(
    run_id: str,
    table_name: str,
    path: str = "ucl-dark/rlvsil-main",
    use_references: bool = False,
    max_num_of_tables=1,
) -> pd.DataFrame:
    """Get outputs from wandb table."""
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
    api = wandb.Api()

    click.echo(f"Getting outputs from wandb for run_id, run_table: {run_id}, {table_name}")

    data, columns = get_artifact_json(f"{path}/run-{run_id}-{table_name}:latest", api)

    return pd.DataFrame(data=data, columns=columns)


def make_output_example(output, eos_token):
    return f"\nTL;DR: {output}{eos_token}"


def make_rf_samples_tldr(examples, eos_token):
    instructions, output_1s, output_2s, choices = [], [], [], []
    for i in range(len(examples["info"])):
        instructions.append(
            make_input_example_tldr(
                examples["info"][i]["post"],
                examples["info"][i]["title"],
                examples["info"][i]["subreddit"],
            )[: -len("\nTL;DR:")]
        )
        output_1s.append(
            make_output_example(
                examples["summaries"][i][0]["text"],
                eos_token,
            )[len("\nTL;DR:  ") :]
        )
        output_2s.append(
            make_output_example(
                examples["summaries"][i][1]["text"],
                eos_token,
            )[len("\nTL;DR:  ") :]
        )

        choices.append(examples["choice"][i] + 1)
    return {
        "instruction": instructions,
        "output_1": output_1s,
        "output_2": output_2s,
        "label": choices,
    }


def make_rf_samples_cnndm(examples, eos_token):
    instructions, output_1s, output_2s, choices = [], [], [], []
    for i in range(len(examples["info"])):
        instructions.append(
            make_input_example_cnndm(
                examples["info"][i]["article"],
            )[: -len("\nTL;DR:")]
        )
        output_1s.append(
            make_output_example(
                examples["summaries"][i][0]["text"],
                eos_token,
            )[len("\nTL;DR:  ") :]
        )
        output_2s.append(
            make_output_example(
                examples["summaries"][i][1]["text"],
                eos_token,
            )[len("\nTL;DR:  ") :]
        )
        choices.append(examples["choice"][i] + 1)
    return {
        "instruction": instructions,
        "output_1": output_1s,
        "output_2": output_2s,
        "label": choices,
    }


def get_dataset(dataset_name: str, num_of_samples: int = 500):
    if dataset_name == DATASET_TLDR:
        ds = load_dataset("UCL-DARK/openai-tldr-summarisation-preferences", use_auth_token=True)
        make_rf_samples = make_rf_samples_tldr
        columns_to_remove = ["info", "split", "summaries", "choice", "worker", "batch"]
        ds = ds["test"].shuffle(seed=13)
    elif dataset_name == DATASET_CNNDM:
        ds = load_dataset("openai/summarize_from_feedback", "comparisons")
        ds = ds.filter(lambda x: "cnndm" in x["batch"])
        make_rf_samples = make_rf_samples_cnndm
        columns_to_remove = ["info", "summaries", "choice", "worker", "batch", "split", "extra"]
        ds = ds["validation"].shuffle(seed=13)
    else:
        raise Exception(f"Unkown dataset {dataset_name}")

    ds = ds.map(
        make_rf_samples,
        fn_kwargs=dict(eos_token=""),  # type: ignore
        batched=True,
    ).remove_columns(columns_to_remove)

    if num_of_samples:
        ds = ds[:num_of_samples]

    return ds


def calculate_accuracy(eval_results, df):
    if len(eval_results) != len(df):
        return -999
    n_correct = 0
    for label, eval_res in zip(df["label"], eval_results):
        # Label and preferences are either 1 or 2. The eval res outputs 2 if
        # the first output is preferred, so we have to swap it
        n_correct += 1 - (label - 1) == (eval_res["preference"] - 1)
        # We could do acc += 3 - label == eval_res["preference"] but that's
        # confusing
    return n_correct / len(df)


@click.command()
@click.option("--wandb_run_id", default=None)
@click.option("--wandb_table_name", default="testtext_table")
@click.option(
    "--reference_wandb_run_id",
    default=None,
    help="wandb run id to get the references from, if not from the dataset/wandb table.",
)
@click.option(
    "--reference_wandb_run_table",
    default=None,
    help="wandb table to get the references from, if not from the dataset/wandb table.",
)
@click.option("--reference_model_name", default=None, help="Name of the reference model.")
@click.option(
    "--output_folder",
    default="results",
    help="Where (if anywhere) to save the results.",
)
@click.option(
    "--dataset_name",
    default=None,
    help="Name of a dataset to load and evaluate.",
)
@click.option("--num_instances", default=500, help="Number of instances to evaluate.")
@click.option(
    "--model_name", default=None, help="Pretty name of the model to put in alpaca_eval leaderboard."
)
@click.option(
    "--dataset_type", default=DATASET_TLDR, help="What type of dataset being evaluated for the wandb table"
)
@click.option("--update_wandb", default=False, help="Whether to update wandb run")
def run(
    wandb_run_id: str,
    wandb_table_name: str,
    reference_wandb_run_id: str,
    reference_wandb_run_table: str,
    reference_model_name: str,
    output_folder: str,
    dataset_name: str,
    num_instances: int,
    model_name: str,
    dataset_type: str,
    update_wandb: bool,
):
    if dataset_name:
        ds = get_dataset(dataset_name=dataset_name, num_of_samples=num_instances)
        df = pd.DataFrame(ds)
        # df = df.drop_duplicates(subset="instruction")
        annotators_config = DATASET_TO_EVAL_CONFIGS[dataset_name]
        model_name = f"{dataset_name}-evaluation"
    else:
        df = get_wandb_table(run_id=wandb_run_id, table_name=wandb_table_name)
        df = df.rename(columns={"input": "instruction", "response": "output_1", "reference": "output_2"})
        annotators_config = DATASET_TO_EVAL_CONFIGS[dataset_type]
        model_name = model_name or f"{wandb_run_id}_{wandb_table_name}"
        df["reference_generator"] = "dataset_reference"

        if reference_wandb_run_id is not None and reference_wandb_run_table is not None:
            df_ref = get_wandb_table(run_id=reference_wandb_run_id, table_name=reference_wandb_run_table)
            df_ref.rename(columns={"input": "instruction"}, inplace=True)
            df = pd.merge(df, df_ref, on="instruction", how="inner")
            df["output_2"] = df["response"]
            reference_model_name = (
                reference_model_name or f"{reference_wandb_run_id}_{reference_wandb_run_table}"
            )
            df["reference_generator"] = reference_model_name

        df["instruction"] = df["instruction"].apply(lambda x: x[: -len("\nTL;DR:")])
        df = df[: min(num_instances, len(df))]

    df["generator"] = model_name

    res = alpaca_main.evaluate(
        model_outputs=df.loc[:, ["instruction", "output_1"]].rename(columns={"output_1": "output"}),
        reference_outputs=df.loc[:, ["instruction", "output_2"]].rename(columns={"output_2": "output"}),
        annotators_config=annotators_config,
        is_return_instead_of_print=True,
        annotation_kwargs=dict(is_ordered=True),
        name=model_name,
        is_overwrite_leaderboard=True,
    )

    click.echo("Main results:")
    click.echo(res[0])
    accuracy = None

    if update_wandb and wandb_run_id is not None:
        click.echo(f"Updating wandb run {wandb_run_id}")
        wapi = wandb.Api()
        run = wapi.run(f"ucl-dark/rlvsil-main/{wandb_run_id}")
        if reference_wandb_run_id is not None and reference_wandb_run_table is not None:
            new_summary = {
                f"{k}_vs_{reference_model_name}": v for k, v in res[0].loc[model_name].to_dict().items()
            }
        else:
            new_summary = res[0].loc[model_name].to_dict()
        run.summary.update(new_summary)
        run.update()

    if dataset_name and "label" in df.columns:
        accuracy = {"accuracy": calculate_accuracy(res[1], df), "dataset": dataset_name}
        click.echo("Accuracy of GPT-4 Annotator on dataset:")
        click.echo(accuracy)

    if output_folder:
        eval_results = [res[0].to_json(), res[1]]
        datasource = dataset_name or f"{wandb_run_id}_{wandb_table_name}"
        datasource += f"_{num_instances}"
        if reference_wandb_run_id is not None and reference_wandb_run_table is not None:
            datasource += f"reference_{reference_wandb_run_id}_{reference_wandb_run_table}"

        os.makedirs(f"{output_folder}/{datasource}", exist_ok=True)

        click.echo(f"Saving results to {output_folder}/{datasource}/")

        with open(f"{output_folder}/{datasource}/evaluation_result.json", "w") as f:
            json.dump(eval_results, f)

        df.to_csv(f"{output_folder}/{datasource}/data_sample.csv")

        if accuracy:
            with open(f"{output_folder}/{datasource}/accuracy.json", "w") as f:
                json.dump(accuracy, f)


if __name__ == "__main__":
    """
    - Example of calling this script using the table from
      https://wandb.ai/ucl-dark/rlvsil-main/artifacts/run_table/run-9226361-3-testtext_table/82f3c88dec63e3b188d9:
    python human_eval_scripts/evaluate_generated_text_with_alpaca.py \
    --wandb_run_ids=9226361-3 --wandb_table_name=test \
    --output_folder=/path-to-folder/

    - Example using a dataset:
    python human_eval_scripts/evaluate_generated_text_with_alpaca.py \
    --dataset_name=tldr --output_folder=/path-to-folder/

    IMPORTANT!!!
    ------------
    The annotators_config that this script is using is: rlvsil_eval_configs.yaml.
    At your local environment please copy the files
    rlvsil/alpaca_eval_configs/* to alpaca_eval/evaluators_configs/. The
    alpaca_eval library uses alpaca_eval/evaluators_configs/ by default, so the
    config files should be placed there.

    Tips
    ----
    - How to get the wandb run id for a model e.g. 2.7b OPT SFT trained on 100% dataset:
    python human_eval_scripts/model_spec_to_run_id.py 2.7b 100 sl
    500 samples per dataset and always use the test split.

    python human_eval_scripts/model_spec_to_run_id.py 2.7b 100 sl
    2930376-1 - 2.7b: 70.3406813627
    python human_eval_scripts/model_spec_to_run_id.py 125m 100 sl
    66802681-6 - 125m: 99.1983967936
    python human_eval_scripts/model_spec_to_run_id.py 350m 100 sl
    66802681-33 - 350m: 98.59437751
    python human_eval_scripts/model_spec_to_run_id.py 1.3b 100 sl
    66802681-40 - 1.3b: 97.6
    python human_eval_scripts/model_spec_to_run_id.py 6.7b 100 sl
    2930376-2 - 6.7b
    """
    run()
