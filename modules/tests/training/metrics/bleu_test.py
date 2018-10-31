# pylint: disable=protected-access

import math

from numpy.testing import assert_almost_equal
from allennlp.common.testing import AllenNlpTestCase

from modules.training.metrics import BLEU


class BleuTest(AllenNlpTestCase):

    def setUp(self):
        super().setUp()
        self.metric = BLEU(ngram_weights=(0.5, 0.5))

    def test_score(self):
        self.metric.reset()
        self.metric(["this a test sentence".split()], ["this is a test sentence".split()])
        assert self.metric._precision_matches == {
                1: 4,
                2: 2,
        }
        assert self.metric._precision_totals == {
                1: 4,
                2: 3,
        }
        assert self.metric._prediction_lengths == 4
        assert self.metric._reference_lengths == 5
        bleu = self.metric.get_metric(reset=True)["BLEU"]
        bleu_check = math.exp(1.0 - 5 / 4 +  # brevity penalty
                              0.5 * (math.log(4) - math.log(4)) +  # 1-gram score
                              0.5 * (math.log(2) - math.log(3)))   # 2-gram score
        assert_almost_equal(bleu, bleu_check)
