"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from dataclasses import dataclass
from typing import Dict, List

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (DataCollatorForTokenClassification,
                          DataCollatorWithPadding)

from dataset.summarisation_feedback_dataloader import tokenize as rf_tokenize
from dataset.summarisation_formatting import (make_filtered_dataset,
                                              make_input_example_cnndm,
                                              make_input_example_tldr,
                                              make_rf_input_example_tldr)


def make_rf_prediction_samples(examples, tokenizer):
    inputs = [
        make_rf_input_example_tldr(p, t, sr, o, tokenizer.eos_token)
        for p, t, sr, o in zip(
            examples["post"],
            examples["title"],
            examples["subreddit"],
            examples["summary"],
        )
    ]
    return {"inputs": inputs}


def make_tldr_samples(examples):
    inputs = [
        make_input_example_tldr(p, t, sr)
        for p, t, sr in zip(examples["post"], examples["title"], examples["subreddit"])
    ]
    outputs = [" " + summary for summary in examples["summary"]]
    return {"queries": inputs, "outputs": outputs}


def make_cnndm_samples(examples):
    inputs = [make_input_example_cnndm(ar) for ar in examples["article"]]
    outputs = [" " + highlights for highlights in examples["highlights"]]
    return {"queries": inputs, "outputs": outputs}


def tokenization(example, *, tokenizer, rl_training, max_source_length, max_target_length):
    prev_padding_side = tokenizer.padding_side
    prev_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if rl_training:
        # For RL, we want to pad to the left so that the batch of inputs are
        # right-aligned, and hence generation can be batched reasonably well
        input_ids = tokenizer(
            example["queries"], max_length=max_source_length, truncation=True, padding=False
        )["input_ids"]
        model_inputs = {"labels": example["outputs"], "input_ids": input_ids}
    else:  # not rl training
        inputs = example["queries"]
        labels = example["outputs"]
        input_idss = tokenizer(
            [i + l for i, l in zip(inputs, labels)],
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
        )["input_ids"]

        # add eos_token_ids to end of each inputs in input_idss
        for input_ids in input_idss:
            if input_ids[-1] != tokenizer.eos_token_id:
                input_ids.append(tokenizer.eos_token_id)

        # Mask out source tokens in model targets.
        source_input_idss = tokenizer(
            inputs,
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
        )["input_ids"]
        label_idss: List[List[int]] = []
        for source_input_ids, input_ids in zip(source_input_idss, input_idss):
            label_ids = input_ids.copy()
            label_ids[: len(source_input_ids)] = [-100] * len(source_input_ids)
            label_idss.append(label_ids)
        model_inputs = {"input_ids": input_idss, "labels": label_idss}

    tokenizer.padding_side = prev_padding_side
    tokenizer.truncation_side = prev_truncation_side
    return model_inputs


@dataclass
class DataCollatorWithLabelPaddingWithSide(DataCollatorForTokenClassification):
    padding_side: str = "left"

    def __call__(self, examples):
        prev_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = self.padding_side
        padding_keys = ["input_ids", "labels", "attention_mask"]
        batch = super().__call__([{k: v for k, v in ex.items() if k in padding_keys} for ex in examples])
        batch.update({k: [ex[k] for ex in examples] for k in examples[0].keys() if k not in padding_keys})
        self.tokenizer.padding_side = prev_padding_side
        return batch


@dataclass
class DataCollatorWithPaddingWithSide(DataCollatorWithPadding):
    padding_side: str = "left"

    def __call__(self, examples):
        prev_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = self.padding_side
        padding_keys = ["input_ids"]
        batch = super().__call__([{k: v for k, v in ex.items() if k in padding_keys} for ex in examples])
        batch.update({k: [ex[k] for ex in examples] for k in examples[0].keys() if k not in padding_keys})
        self.tokenizer.padding_side = prev_padding_side
        return batch


def make_summarisation_collate_fn(args, tokenizer):
    if args.rl_training:
        return DataCollatorWithPaddingWithSide(tokenizer, padding_side="left")
    else:
        return DataCollatorWithLabelPaddingWithSide(tokenizer, padding_side="left")


def make_tldr_raw_text_dataset(
    summarisation_dataset_queries: bool, dataset_structured_subset: str, dataset_random_subset: str
):
    ds_name = "UCL-DARK/openai-tldr-filtered"
    if summarisation_dataset_queries:
        ds_name += "-queries"
    ds = load_dataset(ds_name, use_auth_token=True)
    ds = make_filtered_dataset(ds, dataset_structured_subset, dataset_random_subset)
    ds = ds.map(make_tldr_samples, batched=True)
    return ds


def make_cnndm_raw_text_dataset():
    ds = load_dataset("cnn_dailymail", "3.0.0")
    ds = ds.map(make_cnndm_samples, batched=True)
    return ds


def make_raw_text_dataset(args):
    if args.eval_dataset == "cnndm":
        return make_cnndm_raw_text_dataset()
    return make_tldr_raw_text_dataset(
        args.summarisation_dataset_queries, args.dataset_structured_subset, args.dataset_random_subset
    )


def make_summarisation_dataset(args, tokenizer):
    # TODO: Adjust this for public dataset
    ds = make_raw_text_dataset(args)
    ds = ds.map(
        tokenization,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            rl_training=args.rl_training,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        ),
        batched=True,
        num_proc=8,
    )
    return ds


def get_summarisation_dataloader(args, tokenizer) -> Dict[str, DataLoader]:
    ds = make_summarisation_dataset(args, tokenizer)
    collate_fn = make_summarisation_collate_fn(args, tokenizer)

    result = {
        name: DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            shuffle=name == "train",
            drop_last=args.rl_training,
        )
        for name, dataset in ds.items()
    }
    return result


def make_rf_prediction_dataset(args, tokenizer):
    ds_name = "UCL-DARK/openai-tldr-filtered"
    if args.summarisation_dataset_queries:
        ds_name += "-queries"
    ds = load_dataset(ds_name, use_auth_token=True)
    ds = ds.map(make_rf_prediction_samples, fn_kwargs=dict(tokenizer=tokenizer), batched=True)
    ds = ds.map(rf_tokenize, fn_kwargs=dict(tokenizer=tokenizer, add_labels=False), batched=True, num_proc=8)
    return ds
