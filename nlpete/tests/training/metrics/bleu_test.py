# pylint: disable=protected-access,not-callable

from collections import Counter

import numpy as np
import torch

from allennlp.common.testing import AllenNlpTestCase

from nlpete.training.metrics import BLEU


class BleuTest(AllenNlpTestCase):

    def setUp(self):
        super().setUp()
        self.metric = BLEU(ngram_weights=(0.5, 0.5))

    def test_get_valid_tokens_mask(self):
        tensor = torch.tensor([[1, 2, 3, 0],
                               [0, 1, 1, 0]])
        result = self.metric._get_valid_tokens_mask(tensor, set((0,)))
        result = result.long().numpy()
        check = np.array([[1, 1, 1, 0],
                          [0, 1, 1, 0]])
        np.testing.assert_array_equal(result, check)

    def test_ngrams(self):
        tensor = torch.tensor([[1, 2, 3, 0],
                               [1, 1, 2, 1]])

        # Unigrams.
        counts = Counter(self.metric._ngrams(tensor, 1, set((0,))))
        unigram_check = {
                (1,): 4,
                (2,): 2,
                (3,): 1,
        }
        assert counts == unigram_check

        # Bigrams.
        counts = Counter(self.metric._ngrams(tensor, 2, set((0,))))
        bigram_check = {
                (1, 2): 2,
                (2, 3): 1,
                (1, 1): 1,
                (2, 1): 1
        }
        assert counts == bigram_check

        # Trigrams.
        counts = Counter(self.metric._ngrams(tensor, 3, set((0,))))
        trigram_check = {
                (1, 2, 3): 1,
                (1, 1, 2): 1,
                (1, 2, 1): 1
        }
        assert counts == trigram_check

        # ngram size too big, no ngrams produced.
        counts = Counter(self.metric._ngrams(tensor, 5, set((0,))))
        assert counts == {}
