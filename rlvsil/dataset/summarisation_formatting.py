"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Callable, Dict, Optional

import numpy as np
from datasets import DatasetDict

FIFTY_PC_RNG = 50
TEN_PC_RNG = 10


DATASET_RANGE: Dict[int, Callable[[int], np.ndarray]] = {
    10: lambda upper: np.random.default_rng(TEN_PC_RNG).integers(0, upper, upper // 10),
    50: lambda upper: np.random.default_rng(FIFTY_PC_RNG).integers(0, upper, upper // 2),
}


SUBREDDITS = {"relationship_advice", "AskReddit", "relationships", "tifu", "dating_advice"}


LEN_SPLIT = 245


def make_rf_input_example_tldr(post, title, subreddit, output, eos_token):
    return f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR: {output}{eos_token}"


def make_input_example_tldr(post, title, subreddit):
    return f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR:"


def make_rf_input_example_cnndm(article, output, eos_token):
    return f"ARTICLE: {article}\nTL;DR: {output}{eos_token}"


def make_input_example_cnndm(article):
    return f"ARTICLE: {article}\nTL;DR:"


def make_filtered_dataset(
    dataset: DatasetDict,
    dataset_structured_subset: Optional[str],
    dataset_random_subset: Optional[int],
    info_key: Optional[str] = None,
    rm_dataset: bool = False,
) -> DatasetDict:
    get_subreddit = lambda x: x[info_key]["subreddit"] if info_key else x["subreddit"]  # noqa: E731
    get_post = lambda x: x[info_key]["post"] if info_key else x["post"]  # noqa: E731

    if dataset_random_subset is not None:
        new_dataset = DatasetDict(
            {
                split: dataset[split].select(DATASET_RANGE[dataset_random_subset](len(dataset[split])))
                if split in {"train", "validation"}
                else dataset[split]
                for split in dataset.keys()
            }
        )
    elif (subset := dataset_structured_subset) is not None:
        if subset in SUBREDDITS:
            new_dataset = dataset.filter(lambda eg: get_subreddit(eg) == subset)
            new_dataset["full_validation"] = dataset["validation"]
            new_dataset["ood_validation"] = dataset["validation"].filter(
                lambda eg: get_subreddit(eg) != subset
            )
            new_dataset["full_test"] = dataset["test"]
            new_dataset["ood_test"] = dataset["test"].filter(lambda eg: get_subreddit(eg) != subset)
        elif subset == "length":
            new_dataset = dataset.filter(lambda eg: len(get_post(eg).split(" ")) <= LEN_SPLIT)
            new_dataset["full_validation"] = dataset["validation"]
            new_dataset["ood_validation"] = dataset["validation"].filter(
                lambda eg: len(get_post(eg).split(" ")) > LEN_SPLIT
            )
            new_dataset["full_test"] = dataset["test"]
            new_dataset["ood_test"] = dataset["test"].filter(
                lambda eg: len(get_post(eg).split(" ")) > LEN_SPLIT
            )
        elif subset == "sentiment":
            indices_filename = (
                "dataset/indices/sentiment_indices.npy"
                if not rm_dataset
                else "dataset/indices/rm_sentiment_indices.npy"
            )
            sentiment_indices = np.load(indices_filename, allow_pickle=True).item()
            new_dataset = DatasetDict(
                {split: dataset[split].select(sentiment_indices[split]) for split in dataset.keys()}
            )
            complement_indices = {
                split: np.setdiff1d(np.arange(len(dataset[split])), sentiment_indices[split])
                for split in dataset.keys()
            }
            new_dataset["full_validation"] = dataset["validation"]
            new_dataset["ood_validation"] = dataset["validation"].select(complement_indices["validation"])
            new_dataset["full_test"] = dataset["test"]
            new_dataset["ood_test"] = dataset["test"].select(complement_indices["test"])
        else:
            raise ValueError(f"Unknown subset {subset}")
    else:
        new_dataset = dataset
    return new_dataset
