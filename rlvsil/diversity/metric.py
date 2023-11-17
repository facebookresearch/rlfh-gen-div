"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

# Code adapted from https://github.com/GuyTevet/diversity-eval.
# For license see https://github.com/GuyTevet/diversity-eval/blob/master/LICENSE
# or LICENSE in this folder
import csv
import os
import tempfile
from abc import ABC, abstractmethod

import numpy as np

# locals
import diversity.utils as utils

global_score_cache = {}
similarity2diversity_function = lambda sim_score_list: 1 - np.mean(sim_score_list)


class Metric(ABC):
    use_me = False  # static var indicates to run files whether or not to use this metric
    default_config: dict = {}  # static var, specifies the default config for run files

    def __init__(self, config):
        self.config = config

        # validate config
        assert type(self.config) == dict, "Metric config must be dict type."

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def uint_assert(self, field_name):
        err_msg = "Required: {}(int) > 0".format(field_name)
        assert type(self.config.get(field_name, None)) == int, err_msg
        assert self.config[field_name] > 0, err_msg

    def input_path_assert(self, field_name):
        err_msg = "[{}] not exists.".format(field_name)
        assert os.path.exists(self.config.get(field_name, None)), err_msg


class DiversityMetric(Metric):
    required_input = "response_set"  # in most cases, the diversity metric input is the response set S_c

    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def __call__(self, response_set):
        # validate input
        # assert type(response_set) == list
        # assert all([type(e) == str for e in response_set])

        # place holder
        diversity_score = None
        return diversity_score


class SimilarityMetric(Metric):
    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def __call__(self, resp_a, resp_b):
        # validate input
        # assert type(resp_a) == type(resp_b) == str

        # place holder
        similarity_score = None
        return similarity_score


class Similarity2DiversityMetric(DiversityMetric):
    """
    Implements the diversity to similarity reduction specified on section 5 in the paper
    (https://arxiv.org/pdf/2004.02990.pdf)
    for any similarity metric.

    config:
        shared with the original similarity metric.

    usage:
        metric = Similarity2DiversityMetric(config, SimilarityMetricClassName)
        metric(response_set)

    inheritance guidelines:
        implement __init__ only

    inheritance example:
        see CosineSimilarity2Diversity
    """

    batched = False
    batch_size = 1

    def __init__(self, config, similarity_metric_class):
        super().__init__(config)
        assert issubclass(similarity_metric_class, SimilarityMetric)
        self.similarity_metric = similarity_metric_class(config)

    def __call__(self, response_set):
        super().__call__(response_set)

        if not self.batched:
            similarity_list = []
            for i in range(len(response_set)):
                for j in range(i):
                    similarity_list.append(self.similarity_metric(response_set[i], response_set[j]))
        else:
            pairs = []
            for i in range(len(response_set)):
                for j in range(i):
                    pairs.append((response_set[i], response_set[j]))
            # Calculate similarities in batches:
            similarity_list = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]
                similarity_list.extend(
                    self.similarity_metric(
                        [pair[0] for pair in batch_pairs], [pair[1] for pair in batch_pairs]
                    )
                )

        diversity_score = similarity2diversity_function(similarity_list)
        return diversity_score


class Similarity2DiversityFromFileMetric(DiversityMetric):
    required_input = "set_index"  # when reading results from a file, the input is the set index

    default_config = {
        "input_path": None,
        "num_sets": -1,
        "samples_per_set": -1,
    }  # required fields - filled by run files

    def __init__(self, config):
        super().__init__(config)

        # validate input
        self.uint_assert("num_sets")
        self.uint_assert("samples_per_set")
        self.input_path_assert("input_path")

        # define cache
        metric_name = utils.CamleCase2snake_case(type(self).__name__)
        self.config["cache_file"] = os.path.join(
            tempfile.gettempdir(),
            os.path.basename(self.config["input_path"].replace(".csv", "_{}_scores.tsv".format(metric_name))),
        )
        self.config["input_tsv"] = os.path.join(
            tempfile.gettempdir(),
            os.path.basename(self.config["input_path"].replace(".csv", "_{}_input.tsv".format(metric_name))),
        )

    @abstractmethod
    def calc_scores(self):
        # input: input_csv
        # output: save score file (as temp_file)
        pass

    def create_input_tsv(self):
        # reformat input_csv for to a tsv file, as an input for sentence similarity neural models

        out_fields = ["index", "sentence1_id", "sentence2_id", "sentence1", "sentence2"]
        f_in = open(self.config["input_path"], "r")
        f_out = open(self.config["input_tsv"], "w")
        reader = csv.DictReader(f_in, dialect="excel")
        writer = csv.DictWriter(f_out, fieldnames=out_fields, dialect="excel-tab")
        writer.writeheader()

        for idx, in_row in enumerate(reader):
            for i in range(self.config["samples_per_set"]):
                for j in range(i):
                    writer.writerow(
                        {
                            "index": idx,
                            "sentence1_id": i,
                            "sentence2_id": j,
                            "sentence1": in_row["resp_{}".format(i)],
                            "sentence2": in_row["resp_{}".format(j)],
                        }
                    )

        f_in.close()
        f_out.close()

    def get_similarity_scores(self):

        global global_score_cache  # Here we save the scores in memory for cheaper access

        # fetch or calc scores
        if self.config["cache_file"] in global_score_cache.keys():
            scores = global_score_cache[self.config["cache_file"]]
        else:
            if not os.path.isfile(self.config["cache_file"]) or self.config.get("ignore_cache", False):
                self.calc_scores()
            with open(self.config["cache_file"], "r") as cache_f:
                scores = cache_f.read().split("\n")[:-1]
                assert len(scores) == self.config["num_sets"] * sum(
                    [i for i in range(self.config["samples_per_set"])]
                )  # choose(samples_per_set, 2)
                scores = [float(e) for e in scores]
                scores = np.reshape(scores, [self.config["num_sets"], -1])
                global_score_cache[self.config["cache_file"]] = scores  # cache
        return scores

    def __call__(self, response_set_idx):

        # validate input
        assert type(response_set_idx) == int

        similarity_list = self.get_similarity_scores()[response_set_idx, :]
        diversity_score = similarity2diversity_function(similarity_list)
        return diversity_score


class AveragedNgramDiversityMetric(DiversityMetric):
    """
    Calculates the mean values of an n-gram based diversity metric in range n in [n_min, n_max].

    config:
        shared with the original n-gram metric.
        n_min(int) > 0 - Specify the lowest n-gram value to be averaged
        n_max(int) > 0 - Specify the highest n-gram value to be averaged

    usage:
        metric = AveragedNgramDiversityMetric(config, NgramMetricClassName)
        metric(response_set)

    inheritance guidelines:
        implement __init__ only

    inheritance example:
        see AveragedDistinctNgrams
    """

    def __init__(self, config, ngram_metric_class):
        super().__init__(config)

        # validate config
        self.uint_assert("n_min")
        self.uint_assert("n_max")
        err_msg = "AveragedNgramMetric config must include n_max > n_min > 0 (int) representing n-gram size."
        assert self.config["n_max"] > self.config["n_min"] > 0, err_msg

        # add n field
        self.config["n"] = self.config["n_min"]

        # instance ngram metric
        assert issubclass(ngram_metric_class, DiversityMetric)
        self.ngram_metric = ngram_metric_class(self.config)

    def __call__(self, response_set):
        super().__call__(response_set)

        ngrams_results = []
        for n in range(self.config["n_min"], self.config["n_max"] + 1):
            self.config["n"] = n
            result = self.ngram_metric(response_set)
            # print('{}, {}'.format(self.ngram_metric.config['n'], result))
            ngrams_results.append(result)
        return np.mean(ngrams_results)
