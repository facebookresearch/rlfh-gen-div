"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

# Code adapted from https://github.com/GuyTevet/diversity-eval.
# For license see https://github.com/GuyTevet/diversity-eval/blob/master/LICENSE
# or LICENSE in this folder
import random

import sentence_transformers
import torch
from openai.embeddings_utils import get_embeddings
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline

# local
import diversity.metric as metric
import diversity.utils as utils


class CosineSimilarity(metric.SimilarityMetric):
    def __init__(self, config):
        super().__init__(config)

        # validate config
        self.uint_assert("n")

    def ngram_cosine_distance(selff, ngram1, ngram2):
        """
        Calc cosine distannce between (ngram1, ngram2) [[derived from (str1, str2)]] in the n-gram space.
        """

        def intersection(lst1, lst2):
            return list(set(lst1) & set(lst2))

        # acceleration step - if no intersection -> dist = 1.
        if len(intersection(ngram1, ngram2)) == 0:
            return 1.0
        else:
            n_space = list(set().union(ngram1, ngram2))

            # vectorize
            vectors = []
            for n_gram in [ngram1, ngram2]:
                vectors.append([n_gram.count(e) for e in n_space])

            return cosine(vectors[0], vectors[1])  # uv/|u||v|

    def ngram_cosine_similarity(self, str1, str2, n):
        ngrams = utils.lines_to_ngrams([str1, str2], n)
        return 1 - self.ngram_cosine_distance(ngrams[0], ngrams[1])

    def __call__(self, resp_a, resp_b):
        super().__call__(resp_b, resp_b)
        return self.ngram_cosine_similarity(resp_a, resp_b, n=self.config["n"])


class SentBertSimilarity(metric.SimilarityMetric):
    def __init__(self, config):
        super().__init__(config)

        self.model_name = "bert-large-nli-stsb-mean-tokens"  # FIXME - hard coded
        self.model = sentence_transformers.SentenceTransformer(self.model_name)
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda:0"))

    # @functools.cache
    def embed(self, sentence):
        return self.model.encode(sentence)

    # @functools.cache
    def sent_bert_cosine_similarity(self, resps_1, resps_2):
        embeds_1 = self.model.encode(resps_1)
        embeds_2 = self.model.encode(resps_2)
        return cosine_similarity(embeds_1, embeds_2).diagonal()

    # @functools.cache
    def __call__(self, resp_a, resp_b):
        super().__call__(resp_b, resp_b)
        return self.sent_bert_cosine_similarity(resp_a, resp_b)


class NLISimilarity(metric.SimilarityMetric):
    default_config = {"model_name": "roberta-large-mnli", "top_k": 1}

    def __init__(self, config):
        super().__init__(config)
        self.label_to_weight = {
            "CONTRADICTION": 1,
            "NEUTRAL": 0,
            "ENTAILMENT": -1,
        }
        self.top_k = self.config["top_k"]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pipeline = pipeline("text-classification", model="roberta-large-mnli", device=device)

    def process_output(self, output):
        return sum(label_dict["score"] * self.label_to_weight[label_dict["label"]] for label_dict in output)

    def nli_similarity(self, resps_1: str, resps_2: str) -> list:
        inputs = []
        for resp_1, resp_2 in zip(resps_1, resps_2):
            inputs.append(f"{resp_1}.</s></s>{resp_2}.")
            inputs.append(f"{resp_2}.</s></s>{resp_1}.")

        model_outputs = self.pipeline(inputs, top_k=self.top_k, truncation=True, max_length=512)
        processed_outputs = [self.process_output(output) for output in model_outputs]
        return list(map(sum, zip(processed_outputs[::2], processed_outputs[1::2])))

    def __call__(self, resp_a, resp_b):
        super().__call__(resp_b, resp_b)
        single_instance = False
        if isinstance(resp_a, str):
            resp_a = [resp_a]
            resp_b = [resp_b]
            single_instance = True
        result = self.nli_similarity(resp_a, resp_b)
        if single_instance:
            return result[0]
        return result


class NLISampleSimilarity(NLISimilarity):
    """Does NLISimilarity over sentences sampled from the two sets of outputs.

    So if we get two sets of outputs, for each set we first split it into
    sentences, then we sample n sentences from each set, and then we do
    NLISimilarity on the two sets of sampled sentences.
    """

    default_config = {"model_name": "roberta-large-mnli", "top_k": 1, "n": 5}

    def __init__(self, config):
        super().__init__(config)
        self.label_to_weight = {
            "CONTRADICTION": 1,
            "NEUTRAL": 0,
            "ENTAILMENT": -1,
        }
        self.n = self.config["n"]

    def sample_sentences(self, response_set_a, response_set_b):
        """Sample n * num_responses sentences from the response sets after splitting into sentences"""
        sentences_a, sentences_b = [], []
        for resp_a, resp_b in zip(response_set_a, response_set_b):
            resp_a_sentences = resp_a
            resp_b_sentences = resp_b
            # Cut them to be the same length
            min_len = min(len(resp_a_sentences), len(resp_b_sentences))
            resp_a_sentences = resp_a_sentences[:min_len]
            resp_b_sentences = resp_b_sentences[:min_len]
            # Sample n sentences from each
            sentences_a.extend(random.sample(resp_a_sentences, min(self.n, len(resp_a_sentences))))
            sentences_b.extend(random.sample(resp_b_sentences, min(self.n, len(resp_b_sentences))))
        return sentences_a, sentences_b

    def __call__(self, resp_a, resp_b):
        super().__call__(resp_b, resp_b)
        single_instance = False
        if isinstance(resp_a, str):
            resp_a = [resp_a]
            resp_b = [resp_b]
            single_instance = True
        sent_a, sent_b = self.sample_sentences(resp_a, resp_b)
        resp_size_increase = len(sent_a) / len(resp_a)
        result = self.nli_similarity(sent_a, sent_b)
        # Scale results down by increase of size of resp_a
        result = [r / resp_size_increase for r in result]
        if single_instance:
            return result[0]
        return result


class OpenAIEmbeddingsSimilarity(metric.SimilarityMetric):
    def __init__(self, config):
        super().__init__(config)

        self.engine = "text-embedding-ada-002"

    def embed(self, sentence):
        if type(sentence) == str:
            sentence = [sentence]
        embeddings = get_embeddings(sentence, self.engine)
        # Retry any inputs that return nans
        # embeddings should be a list of 1536-dimensional vectors, but some elements
        # are a single-element list with a nan. Retry those elements
        elements_to_retry = [i for i, e in enumerate(embeddings) if len(e) == 1 and np.isnan(e[0])]
        retries = 0
        while len(elements_to_retry) > 0 and retries < 5:
            retried_embeddings = get_embeddings([sentence[i] for i in elements_to_retry], self.engine)
            for i, e in zip(elements_to_retry, retried_embeddings):
                embeddings[i] = e
            elements_to_retry = [i for i, e in enumerate(embeddings) if len(e) == 1 and np.isnan(e[0])]
            retries += 1
        if len(elements_to_retry) > 0:
            print(f"Failed to get embeddings for {elements_to_retry} elements")
            # Just set elements_to_retry to all zeros
            for i in elements_to_retry:
                embeddings[i] = np.zeros(1536)
        return embeddings

    def openai_embedding_cosine_similarity(self, resps_1, resps_2):
        embeds_1 = self.embed(resps_1)
        embeds_2 = self.embed(resps_2)
        return cosine_similarity(embeds_1, embeds_2).diagonal()

    def __call__(self, resp_a, resp_b):
        super().__call__(resp_b, resp_b)
        return self.openai_embedding_cosine_similarity(resp_a, resp_b)
