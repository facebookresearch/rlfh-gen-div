"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
# From https://github.com/yizhongw/Tk-Instruct/tree/main/src/compute_metrics.py
# And then formatted and adjusted
import logging
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from rouge import Rouge
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer, PreTrainedTokenizer

import wandb

logger = logging.getLogger(__name__)


rouge_scorer = Rouge()

bleu = BLEU(effective_order=True)


class GPTTokenizer:
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

    def tokenize(self, s):
        tokens = self.gpt_tokenizer.tokenize(s)
        # GPT2 uses Byte-level BPE, which will include space as part of the word.
        # But for the first word of a sentence, there is no space before it.
        # So, we remove all the added spaces ("Ġ").
        tokens = [t.lstrip("Ġ") for t in tokens]
        return tokens


xlingual_tokenizer = GPTTokenizer()


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def bleu_score(prediction, ground_truth, xlingual=False):
    return bleu.sentence_score(prediction, [ground_truth]).score / 100


def exact_match_score(prediction, ground_truth, xlingual=False):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def rouge1_score(prediction, ground_truth, xlingual=False):
    # if xlingual:
    #     rouge_scorer = rouge_scorer.RougeScorer(["rouge1"], tokenizer=xlingual_tokenizer)
    # else:
    #     rouge_scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    try:
        scores = rouge_scorer.get_scores(hyps=prediction, refs=ground_truth)[0]
    except ValueError:
        print("Model returns empty prediction: ", prediction)
        return 0.0
    return scores["rouge-1"]["f"]


def rougeL_score(prediction, ground_truth, xlingual=False):
    # if xlingual:
    #     rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=xlingual_tokenizer)
    # else:
    #     rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    try:
        scores = rouge_scorer.get_scores(hyps=prediction, refs=ground_truth)[0]
    except ValueError:
        print("Model returns empty prediction: ", prediction)
        return 0.0
    return scores["rouge-l"]["f"]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    # Logic for when either prediction or reference is 0
    # Actually call the metric
    for ground_truth in ground_truths:
        if len(ground_truth) == 0:
            if len(prediction) == 0:
                score = 1.0
            else:
                score = 0.0
        elif len(prediction) == 0:
            score = 0.0
        else:
            score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(
        references
    ), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL, bleu_res = 0.0, 0.0, 0.0, 0.0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        bleu_res += metric_max_over_ground_truths(
            bleu_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    exact_match = exact_match / len(references)
    rouge1 = rouge1 / len(references)
    rougeL = rougeL / len(references)
    bleu_res = bleu_res / len(references)
    metrics = {"exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL, "bleu": bleu_res}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, group_prefix: str = "", xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = defaultdict(list)
    for pred, gold, group in zip(predictions, references, groups):
        examples_by_group[group].append((pred, gold))

    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{group_prefix}/{group}/{metric}"] = value
    return results


def compute_all_ni_metrics(
    predictions: List[str],
    token_predictions: List[int],
    references: List[str],
    categories: List[str],
    tasks: List[str],
    domains: List[str],
    inputs: List[str],
    tokenizer: PreTrainedTokenizer,
    rewards: Optional[List[float]] = None,
) -> Dict[str, Any]:
    result = compute_metrics(predictions=predictions, references=references)
    result_per_task = compute_grouped_metrics(
        predictions=predictions, references=references, groups=tasks, group_prefix="task"
    )
    result.update(result_per_task)
    result_per_category = compute_grouped_metrics(
        predictions=predictions, references=references, groups=categories, group_prefix="category"
    )
    result.update(result_per_category)
    result_per_domain = compute_grouped_metrics(
        predictions=predictions, references=references, groups=domains, group_prefix="domain"
    )
    result.update(result_per_domain)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in token_predictions]
    response_data = list(
        map(
            list,
            zip(  # type: ignore
                inputs,
                predictions,
                [r[0] for r in references],
                ([None] * len(predictions)) if rewards is None else rewards,
            ),
        )
    )
    text_table = wandb.Table(columns=["input", "response", "reference", "reward"], data=response_data)
    result.update({"text_table": text_table})
    result.update(
        dict(
            average_prediction_length=np.mean(prediction_lens),
            median_prediction_length=np.median(prediction_lens),
            prediction_length_distribution=prediction_lens,
        )
    )
    if rewards is not None:
        result.update(
            average_reward=np.mean(rewards),
            median_reward=np.median(rewards),
            reward_distribution=np.array(rewards),
        )
    return result


def compute_ni_metrics(dataset, preds, tokenizer):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    references, tasks, categories, domains, inputs = [], [], [], [], []
    for e in dataset:
        references.append(e["Instance"]["output"])
        tasks.append(e["Task"])
        categories.append(e["Categories"][0])
        domains.append(e["Domains"][0])
        inputs.append(e["Instance"]["input"])
    return compute_all_ni_metrics(
        predictions=decoded_preds,
        token_predictions=preds,
        references=references,
        categories=categories,
        domains=domains,
        tasks=tasks,
        inputs=inputs,
        tokenizer=tokenizer,
    )
