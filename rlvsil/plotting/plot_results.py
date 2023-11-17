"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""Utility Functions for plotting results, from wandb and elsewhere."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

INIT_MODELS = {"2699779-3", "2699779-2", "2699779-1", "2712505-1", "2699779-4"}

OOD_SL_MODELS = {
    "relationship": {
        "125m": ("/checkpoint/christoforos/rlvsil/rlvsil-main/5991923-1/final", "5991923-1"),
        "350m": ("/checkpoint/christoforos/rlvsil/rlvsil-main/5842166-13/final", "5842166-13"),
        "1.3b": ("/checkpoint/christoforos/rlvsil/rlvsil-main/5842166-22/final", "5842166-22"),
        "2.7b": ("/checkpoint/christoforos/rlvsil/rlvsil-main/4577049-4/final", "4577049-4"),
        "6.7b": ("/checkpoint/christoforos/rlvsil/rlvsil-main/4577049-10/final", "4577049-10"),
    },
    "length": {
        "125m": ("/checkpoint/christoforos/rlvsil/rlvsil-main/5842166-5/final", "5842166-5"),
        "350m": ("/checkpoint/christoforos/rlvsil/rlvsil-main/5991923-14/final", "5991923-14"),
        "1.3b": ("/checkpoint/christoforos/rlvsil/rlvsil-main/5991923-20/final", "5991923-20"),
        "2.7b": ("/checkpoint/christoforos/rlvsil/rlvsil-main/5258480-8/final", "5258480-8"),
        "6.7b": ("/checkpoint/christoforos/rlvsil/rlvsil-main/4577049-14/final", "4577049-14"),
    },
    "sentiment": {
        "125m": ("/checkpoint/christoforos/rlvsil/rlvsil-main/5842166-3/final", "5842166-3"),
        "350m": ("/checkpoint/christoforos/rlvsil/rlvsil-main/5842166-18/final", "5842166-18"),
        "1.3b": ("/checkpoint/christoforos/rlvsil/rlvsil-main/5842166-21/final", "5842166-21"),
        "2.7b": ("/checkpoint/christoforos/rlvsil/rlvsil-main/5220650-6/final", "5220650-6"),
        "6.7b": ("/checkpoint/christoforos/rlvsil/rlvsil-main/4577049-15/final", "4577049-15"),
    },
}

OOD_RL_MODELS = {
    "relationship": {
        "125m": "10ar3zyt",
        "350m": "ynnufvya",
        "1.3b": "2bakkq98",
        "2.7b": "ym4gvt1q",
        "6.7b": "yin9v1rr",
    },
    "length": {
        "125m": "29b5mvl5",
        "350m": "xmv5jur8",
        "1.3b": "1yeil4ro",
        "2.7b": "fugehm0u",
        "6.7b": "ar8mlsyp",
    },
    "sentiment": {
        "125m": "2wxh6k3t",
        "350m": "skw9wtjw",
        "1.3b": "2dcsqy26",
        "2.7b": "3sanlnim",
        "6.7b": "37wy3v41",
    },
}

OOD_RM_MODELS = {
    "relationship": {
        "125m": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-125M-relationship-final/final_full_normalised",
            "rm-125M-relationship-final",
        ),
        "350m": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-350M-relationship-final/final_full_normalised",
            "rm-350M-relationship-final",
        ),
        "1.3b": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-1.3B-relationship-final/final_full_normalised",
            "rm-1.3B-relationship-final",
        ),
        "2.7b": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-2.7B-relationship-final/final_full_normalised",
            "rm-2.7B-relationship-final",
        ),
        "6.7b": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-6.7B-relationship-final/final_full_normalised",
            "rm-6.7B-relationship-final",
        ),
    },
    "length": {
        "125m": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-125M-length-final/final_full_normalised",
            "rm-125M-length-final",
        ),
        "350m": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-350M-length-final/final_full_normalised",
            "rm-350M-length-final",
        ),
        "1.3b": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-1.3B-length-final/final_full_normalised",
            "rm-1.3B-length-final",
        ),
        "2.7b": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-2.7B-length-final/final_full_normalised",
            "rm-2.7B-length-final",
        ),
        "6.7b": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-6.7B-length-final/final_full_normalised",
            "rm-6.7B-length-final",
        ),
    },
    "sentiment": {
        "125m": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-125M-sentiment-final/final_full_normalised",
            "rm-125M-sentiment-final",
        ),
        "350m": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-350M-sentiment-final/final_full_normalised",
            "rm-350M-sentiment-final",
        ),
        "1.3b": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-1.3B-sentiment-final/final_full_normalised",
            "rm-1.3B-sentiment-final",
        ),
        "2.7b": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-2.7B-sentiment-final/final_full_normalised",
            "rm-2.7B-sentiment-final",
        ),
        "6.7b": (
            "checkpoint/christoforos/rlvsil/rlvsil-rf/rm-6.7B-sentiment-final/final_full_normalised",
            "rm-6.7B-sentiment-final",
        ),
    },
}


OOD_BON_MODELS = {
    "relationship": {
        "125m": "bon_125m_relationships_eval",
        "350m": "bon_350m_relationships_eval",
        "1.3b": "bon_1.3b_relationships_eval",
        "2.7b": "bon_2.7b_relationships_eval_ishita",
        "6.7b": "bon_6.7b_relationships_eval_ishita",
    },
    "length": {
        "125m": "bon_125m_length_eval_1",
        "350m": "bon_350m_length_eval",
        "1.3b": "bon_1.3b_length_eval",
        "2.7b": "bon_2.7b_length_eval_ishita",
        "6.7b": "bon_6.7b_length_eval_ishita",
    },
    "sentiment": {
        "125m": "bon_125m_sentiment_eval_1",
        "350m": "bon_350m_sentiment_eval_1",
        "1.3b": "bon_1.3b_sentiment_eval",
        "2.7b": "bon_2.7b_sentiment_eval_ishita",
        "6.7b": "bon_6.7b_sentiment_eval_ishita",
    },
}


def get_runs_from_wandb(
    wandb_api: wandb.Api, project_name: str = "rlvsil-main", entity_name: str = "ucl-dark", **kwargs: dict
):
    return list(wandb_api.runs(f"{entity_name}/{project_name}", **kwargs))


def id_to_run(
    wandb_api: wandb.Api, run_id: str, project_name: str = "rlvsil-main", entity_name: str = "ucl-dark"
):
    return wandb_api.run(f"{entity_name}/{project_name}/{run_id}")


def ood_sl_model_runs(wandb_api: wandb.Api, project_name: str = "rlvsil-main", entity_name: str = "ucl-dark"):
    results = []
    for split, task_dict in OOD_SL_MODELS.items():
        for size, (run_path, run_id) in task_dict.items():
            run = id_to_run(wandb_api, run_id, project_name, entity_name)
            results.append(
                {
                    "task": split,
                    "size": size,
                    "run_id": run_id,
                    "run_path": run_path,
                    "run": run,
                    "model_name": run.config["model_name"],
                    "dataset": run.config["dataset"],
                }
            )
    return pd.DataFrame(results)


def ood_rm_model_runs(wandb_api: wandb.Api, project_name: str = "rlvsil-main", entity_name: str = "ucl-dark"):
    results = []
    for split, task_dict in OOD_RM_MODELS.items():
        for size, (run_path, run_id) in task_dict.items():
            run_main = id_to_run(wandb_api, run_id, project_name, entity_name)
            run_rf = id_to_run(wandb_api, run_id, "rlvsil-rf", entity_name)
            results.append(
                {
                    "task": split,
                    "size": size,
                    "run_id": run_id,
                    "run_path": run_path,
                    "run_main": run_main,
                    "run_rf": run_rf,
                    "model_name": run_main.config["model_name"],
                    # "dataset": run_main.config["dataset"],
                }
            )
    return pd.DataFrame(results)


def ood_rl_model_runs(wandb_api: wandb.Api, project_name: str = "rlvsil-main", entity_name: str = "ucl-dark"):
    results = []
    for split, task_dict in OOD_RL_MODELS.items():
        for size, run_id in task_dict.items():
            run_main = id_to_run(wandb_api, run_id, project_name, entity_name)
            results.append(
                {
                    "task": split,
                    "size": size,
                    "run_id": run_id,
                    "run": run_main,
                    "model_name": run_main.config["model_name"],
                    "dataset": run_main.config["dataset"],
                }
            )
    return pd.DataFrame(results)


def ood_bon_model_runs(
    wandb_api: wandb.Api, project_name: str = "rlvsil-main", entity_name: str = "ucl-dark"
):
    results = []
    for split, task_dict in OOD_BON_MODELS.items():
        for size, run_id in task_dict.items():
            run_main = id_to_run(wandb_api, run_id, project_name, entity_name)
            results.append(
                {
                    "task": split,
                    "size": size,
                    "run_id": run_id,
                    "run": run_main,
                    # "model_name": run_main.config["model_name"],
                }
            )
    return pd.DataFrame(results)


def wandb_run_to_dict(run, config_keys: list = [], summary_keys: list = []):
    row = {"run": run, "id": run.id, "name": run.name}
    for key in config_keys:
        row[key] = run.config.get(key)
    for key in summary_keys:
        row[key] = run.summary.get(key)
    return row


def wandb_runs_to_df(runs: list, config_keys: list = [], summary_keys: list = []):
    results = []
    for run in runs:
        results.append(wandb_run_to_dict(run, config_keys, summary_keys))
    return pd.DataFrame(results)


def size_to_num(size):
    if size[-1].lower() == "m":
        mult = 1000000
    if size[-1].lower() == "b":
        mult = 1000000000
    return float(size[:3]) * mult


def convert_sl_wandb_df(df):
    df["model_size"] = df["model_name"].map(lambda mod_name: mod_name.split("-")[-1].lower())
    df["model_size_param"] = df["model_size"].map(size_to_num)
    df["dataset_size"] = df["dataset"].map(lambda x: int(x) if x in {"10", "50"} else 100)
    df.loc[df["run_id"].isin(INIT_MODELS), "dataset_size"] = 0
    df["log_model_size_param"] = np.log(df["model_size_param"])
    # df["test_shift"] = df["batch_name"].map(lambda x: "ood" if "trainood" in x else "id")
    # df["split"] = df["batch_name"].map(lambda x: x.split("_")[-1])
    # df["post_length"] = df["post"].map(lambda x: len(x.split(" ")))
    return df


def convert_rm_wandb_df(df):
    df["model_size"] = df["model_name"].map(lambda mod_name: mod_name.split("-")[-2].lower())
    df["model_size_param"] = df["model_size"].map(size_to_num)
    # df["dataset_size"] = df["dataset"].map(lambda x: int(x) if x in {"10", "50"} else 100)
    df.loc[df["run_id"].isin(INIT_MODELS), "dataset_size"] = 0
    df["log_model_size_param"] = np.log(df["model_size_param"])
    # df["test_shift"] = df["batch_name"].map(lambda x: "ood" if "trainood" in x else "id")
    # df["split"] = df["batch_name"].map(lambda x: x.split("_")[-1])
    # df["post_length"] = df["post"].map(lambda x: len(x.split(" ")))
    return df


def plot_statistic_over_time(statistic_name, parameter_set, runs):
    """
    Plots a statistic over time for a set of runs, with a different line for each run.

    Parameters:
    - statistic_name (str): Name of the statistic to plot.
    - parameter_set (dict): Dictionary of parameters for each run, with run ID as the key.
    - runs (list): List of run IDs to include in the plot.
    """

    # Create a new figure and axis object for the plot
    fig, ax = plt.subplots()

    # Loop over the list of run IDs and plot the specified statistic for each run
    for run_id in runs:
        # Load the run using the wandb API
        run = wandb.Api().run(run_id)

        # Get the history of the specified statistic for the run
        history = run.history(keys=[statistic_name])

        # Extract the values of the specified statistic from the history
        values = [x[statistic_name] for x in history]

        # Extract the parameters for the run from the parameter set
        params = parameter_set[run_id]

        # Create a label for the line based on the parameters
        label = ", ".join([f"{k}={v}" for k, v in params.items()])

        # Plot the values over time as a line with the appropriate label
        ax.plot(values, label=label)

    # Set the x-axis label to time and the y-axis label to the specified statistic name
    ax.set_xlabel("Time")
    ax.set_ylabel(statistic_name)

    # Add a legend to the plot
    ax.legend()

    # Show the plot
    plt.show()
