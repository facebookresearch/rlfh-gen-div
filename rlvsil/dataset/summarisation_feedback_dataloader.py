"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, default_data_collator

from dataset.summarisation_formatting import (make_filtered_dataset,
                                              make_input_example_tldr,
                                              make_rf_input_example_cnndm,
                                              make_rf_input_example_tldr)


def make_rf_samples_tldr(examples, eos_token):
    input_0s, input_1s, choices = [], [], []
    for i in range(len(examples["info"])):
        input_0s.append(
            make_rf_input_example_tldr(
                examples["info"][i]["post"],
                examples["info"][i]["title"],
                examples["info"][i]["subreddit"],
                examples["summaries"][i][0]["text"],
                eos_token,
            )
        )
        input_1s.append(
            make_rf_input_example_tldr(
                examples["info"][i]["post"],
                examples["info"][i]["title"],
                examples["info"][i]["subreddit"],
                examples["summaries"][i][1]["text"],
                eos_token,
            )
        )
        choices.append(examples["choice"][i])
    return {
        "input_0": input_0s,
        "input_1": input_1s,
        "label": choices,
    }


def make_rf_samples_cnndm(examples, eos_token):
    input_0s, input_1s, choices = [], [], []
    for i in range(len(examples["info"])):
        input_0s.append(
            make_rf_input_example_cnndm(
                examples["info"][i]["article"],
                examples["summaries"][i][0]["text"],
                eos_token,
            )
        )
        input_1s.append(
            make_rf_input_example_cnndm(
                examples["info"][i]["article"],
                examples["summaries"][i][1]["text"],
                eos_token,
            )
        )
        choices.append(examples["choice"][i])
    return {
        "input_0": input_0s,
        "input_1": input_1s,
        "label": choices,
    }


def tokenize(batch, *, tokenizer, add_labels: bool = True):
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"
    model_inputs = {
        **{f"input_0_{k}": v for k, v in tokenizer(batch["input_0"], truncation=True, padding=False).items()},
        **{f"input_1_{k}": v for k, v in tokenizer(batch["input_1"], truncation=True, padding=False).items()},
    }
    if add_labels:
        model_inputs["label"] = batch["label"]

    tokenizer.truncation_side = "right"
    return model_inputs


def make_cnndm_raw_text_dataset(args, eos_token):
    ds = load_dataset("openai/summarize_from_feedback", "comparisons")
    ds = ds.filter(lambda x: "cnndm" in x["batch"])
    ds = ds.map(
        make_rf_samples_cnndm,
        fn_kwargs=dict(eos_token=eos_token),
        batched=True,
    )
    return ds


def make_tldr_raw_text_dataset(args, eos_token):
    ds = load_dataset("UCL-DARK/openai-tldr-summarisation-preferences", use_auth_token=True)
    ds = make_filtered_dataset(
        ds, args.dataset_structured_subset, args.dataset_random_subset, info_key="info", rm_dataset=True
    )
    ds = ds.map(
        make_rf_samples_tldr,
        fn_kwargs=dict(eos_token=eos_token),
        batched=True,
    )
    return ds


def make_raw_text_dataset(args, eos_token):
    if args.eval_dataset == "cnndm":
        return make_cnndm_raw_text_dataset(args, eos_token)
    return make_tldr_raw_text_dataset(args, eos_token)


def make_dataset(tokenizer, args):
    ds = make_raw_text_dataset(args, tokenizer.eos_token)
    ds = ds.map(
        tokenize,
        fn_kwargs=dict(tokenizer=tokenizer),
        batched=True,
        num_proc=8,
        remove_columns=ds["train"].column_names,
    )
    return ds


def get_summarisation_rf_dataloader(args, tokenizer) -> Dict[str, DataLoader]:
    ds = make_dataset(tokenizer, args)

    data_collator = DataCollatorWithPaddingTwoInputs(tokenizer)

    result = {
        name: DataLoader(
            dataset,
            collate_fn=data_collator,
            batch_size=args.batch_size,
            shuffle=name == "train",
            drop_last=args.rl_training,
        )
        for name, dataset in ds.items()
    }

    return result


@dataclass
class DataCollatorWithPaddingTwoInputs(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract two batches from features, and call pad on them each independently."""
        self.tokenizer.padding_side = "right"
        features_0, features_1, features_other = [], [], []
        for feature in features:
            features_0.append(
                {k.replace("input_0_", ""): v for k, v in feature.items() if k.startswith("input_0_")}
            )
            features_1.append(
                {k.replace("input_1_", ""): v for k, v in feature.items() if k.startswith("input_1_")}
            )
            features_other.append(
                {
                    k: v
                    for k, v in feature.items()
                    if not k.startswith("input_0") and not k.startswith("input_1")
                }
            )
        batch_0 = self.tokenizer.pad(
            features_0,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_1 = self.tokenizer.pad(
            features_1,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_other = default_data_collator(features_other, self.return_tensors)
        batch = dict(
            **{f"input_0_{k}": v for k, v in batch_0.items()},
            **{f"input_1_{k}": v for k, v in batch_1.items()},
            **batch_other,
        )
        return batch
