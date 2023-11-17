"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
# Code adapted from https://github.com/GuyTevet/diversity-eval.
# For license see https://github.com/GuyTevet/diversity-eval/blob/master/LICENSE
# or LICENSE in this folder
import sentence_transformers
from scipy.spatial.distance import cosine

# locals
import diversity.metric as metric
import diversity.similarity_metrics as similarity_metrics
import diversity.utils as utils


class DistinctNgrams(metric.DiversityMetric):

    default_config = {"n": 3}

    def __init__(self, config):
        super().__init__(config)

        # validate config
        self.uint_assert("n")

    def normalized_unique_ngrams(self, ngram_lists):
        """
        Calc the portion of unique n-grams out of all n-grams.
        :param ngram_lists: list of lists of ngrams
        :return: value in (0,1]
        """
        ngrams = [item for sublist in ngram_lists for item in sublist]  # flatten
        return len(set(ngrams)) / len(ngrams) if len(ngrams) > 0 else 0.0

    def __call__(self, response_set):
        super().__call__(response_set)
        return self.normalized_unique_ngrams(utils.lines_to_ngrams(response_set, n=self.config["n"]))


class ExpectationAdjustedDistinctNgrams(metric.DiversityMetric):
    # Taken from https://arxiv.org/abs/2202.13587

    default_config = {"n": 3, "vocab_size": 50257}
    name = "ead_averaged_distinct_ngrams"

    def __init__(self, config):
        super().__init__(config)
        self.name = "ead_averaged_distinct_ngrams"

        # validate config
        self.uint_assert("n")

    def ead_normalized_unique_ngrams(self, ngram_lists):
        """
        Calc expectation-adjusted portion of unique n-grams out of all n-grams.
        :param ngram_lists: list of lists of ngrams
        :return: value in (0,1]
        """
        ngrams = [item for sublist in ngram_lists for item in sublist]
        N = len(set(ngrams))
        C = len(ngrams)
        V = self.config["vocab_size"]

        try:
            ead = N / (V * (1 - ((V - 1) / V) ** C))
        except ZeroDivisionError:
            ead = 0.0
        return ead

    def __call__(self, response_set):
        super().__call__(response_set)
        return self.ead_normalized_unique_ngrams(utils.lines_to_ngrams(response_set, n=self.config["n"]))


class AveragedDistinctNgrams(metric.AveragedNgramDiversityMetric):

    use_me = True
    default_config = {"n_min": 1, "n_max": 5}
    name = "averaged_distinct_ngrams"

    def __init__(self, config):
        super().__init__(config, DistinctNgrams)
        self.name = "averaged_distinct_ngrams"


class AveragedExpectationAdjustedDistinctNgrams(metric.AveragedNgramDiversityMetric):

    use_me = True
    default_config = {"n_min": 1, "n_max": 5, "vocab_size": 50257}
    name = "ead_averaged_distinct_ngrams"

    def __init__(self, config):
        super().__init__(config, ExpectationAdjustedDistinctNgrams)
        self.name = "ead_averaged_distinct_ngrams"


class CosineSimilarity2Diversity(metric.Similarity2DiversityMetric):

    default_config = {"n": 3}
    name = "cosine_similarity_2d_diversity"

    def __init__(self, config):
        super().__init__(config, similarity_metrics.CosineSimilarity)


class AveragedCosineSimilarity(metric.AveragedNgramDiversityMetric):

    use_me = True
    default_config = {"n_min": 1, "n_max": 5}

    def __init__(self, config):
        super().__init__(config, CosineSimilarity2Diversity)


class SentBertFromSim(metric.Similarity2DiversityMetric):

    use_me = True
    name = "sent_bert_from_sim"
    batched = True
    batch_size = 1024

    def __init__(self, config):
        super().__init__(config, similarity_metrics.SentBertSimilarity)


class NLIFromSim(metric.Similarity2DiversityMetric):

    use_me = True
    name = "nli_from_sim"
    batched = True
    batch_size = 1024

    default_config = {"model_name": "roberta-large-mnli", "top_k": 1}

    def __init__(self, config):
        super().__init__(config, similarity_metrics.NLISimilarity)


class NLISampleFromSim(metric.Similarity2DiversityMetric):

    use_me = True
    name = "nli_sample_from_sim"
    batched = True
    batch_size = 1024

    default_config = {"model_name": "roberta-large-mnli", "top_k": 1, "n": 5}

    def __init__(self, config):
        super().__init__(config, similarity_metrics.NLISampleSimilarity)


class OpenAiEmbeddingsFromSim(metric.Similarity2DiversityMetric):

    use_me = True
    name = "openai_from_sim"
    batched = True
    batch_size = 256

    default_config = {"engine": "text-embedding-ada-002", "top_k": 1}

    def __init__(self, config):
        super().__init__(config, similarity_metrics.OpenAIEmbeddingsSimilarity)


if __name__ == "__main__":

    def print_metric(metric, resp_set):
        print("{0}: {1:0.3f}".format(type(metric).__name__, metric(resp_set)))

    # TEST
    resp_set = ["i am going", "i am going", "lets go i i"]
    config = {"n": 3}
    print_metric(CosineSimilarity2Diversity(config), resp_set)
    print_metric(DistinctNgrams(config), resp_set)

    avg_config = {"n_min": 1, "n_max": 5}
    print_metric(AveragedCosineSimilarity(avg_config), resp_set)
    print_metric(AveragedDistinctNgrams(avg_config), resp_set)
