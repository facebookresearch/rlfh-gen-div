"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""A simple script that loads run_id_to_model_name and given model specifications returns the run id."""
import json

import click


@click.command()
@click.argument("model_size", type=str)
@click.argument("dataset_size", type=str)
@click.argument("model_type", type=str)
@click.option(
    "--run_id_to_model_name_path",
    default="save_data/run_id_to_model_name.json",
    help="Path to run_id_to_model_name.json",
    type=click.Path(exists=True),
)
def main(model_size: str, dataset_size: str, model_type: str, run_id_to_model_name_path: str):
    if model_type == "sl":
        with open(run_id_to_model_name_path) as f:
            run_id_to_model_name = json.load(f)
    elif model_type == "bon":
        with open("save_data/bon_run_id_to_model_name.json") as f:
            run_id_to_model_name = json.load(f)
    elif model_type == "rl":
        with open("save_data/rl_run_id_to_model_name.json") as f:
            run_id_to_model_name = json.load(f)
    else:
        raise ValueError("model_type must be one of sl, bon, rl")

    for run_id, model_spec in run_id_to_model_name.items():
        if str(model_spec["model_size"]) == str(model_size) and str(model_spec["dataset_size"]) == str(
            dataset_size
        ):
            print(run_id)
            return
    raise ValueError("No run id found for model size, dataset_size", model_size, dataset_size, model_type)


if __name__ == "__main__":
    main()
