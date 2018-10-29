# pylint: disable=protected-access,not-callable

import json

import numpy as np
import pytest
from scipy.misc import logsumexp
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import ModelTestCase

from modules.models import CopyNet  # pylint: disable=unused-import
from modules.data.dataset_readers import CopyNetDatasetReader  # pylint: disable=unused-import


class CopyNetTest(ModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("modules/tests/fixtures/copynet/experiment.json",
                          "modules/tests/fixtures/copynet/copyover.tsv")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)

    def test_vocab(self):
        vocab = self.model.vocab
        assert vocab.get_vocab_size(self.model._target_namespace) == 8
        assert "hello" not in vocab._token_to_index[self.model._target_namespace]
        assert "world" not in vocab._token_to_index[self.model._target_namespace]

    def test_missing_copy_token_raises(self):
        param_overrides = json.dumps({"vocabulary": {"tokens_to_add": None}})
        with pytest.raises(ConfigurationError):
            self.ensure_model_can_train_save_and_load(self.param_file, overrides=param_overrides)

    def test_train_instances(self):
        inputs = self.instances[0].as_tensor_dict()
        source_tokens = inputs["source_tokens"]
        target_tokens = inputs["target_tokens"]
        copy_indicators = inputs["copy_indicators"]

        assert list(source_tokens["tokens"].size()) == [11]
        assert list(target_tokens["tokens"].size()) == [10]
        assert list(copy_indicators.size()) == [10, 9]

        assert target_tokens["tokens"][0] == self.model._start_index
        assert target_tokens["tokens"][4] == self.model._oov_index
        assert target_tokens["tokens"][5] == self.model._oov_index
        assert target_tokens["tokens"][-1] == self.model._end_index

    def test_get_ll_contrib(self):
        # batch_size = 3, trimmed_input_len = 3
        #
        # In the first instance, the contribution to the likelihood should
        # come from both the generation scores and the copy scores, since the
        # token is in the source sentence and the target vocabulary.
        # In the second instance, the contribution should come only from the
        # generation scores, since the token is not in the source sentence.
        # In the third instance, the contribution should come only from the copy scores,
        # since the token is in the source sequence but is not in the target vocabulary.

        vocab = self.model.vocab

        generation_scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # these numbers are arbitrary.
                                          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                          [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]])
        # shape: (batch_size, target_vocab_size)

        copy_scores = torch.tensor([[1.0, 2.0, 1.0],  # these numbers are arbitrary.
                                    [1.0, 2.0, 3.0],
                                    [2.0, 2.0, 3.0]])
        # shape: (batch_size, trimmed_input_len)

        target_tokens = torch.tensor([vocab.get_token_index("tokens", self.model._target_namespace),
                                      vocab.get_token_index("the", self.model._target_namespace),
                                      self.model._oov_index])
        # shape: (batch_size,)

        copy_indicators = torch.tensor([[0, 1, 0],
                                        [0, 0, 0],
                                        [1, 0, 1]])
        # shape: (batch_size, trimmed_input_len)

        copy_mask = torch.tensor([[1.0, 1.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [1.0, 1.0, 1.0]])
        # shape: (batch_size, trimmed_input_len)

        # This is what the log likelihood result should look like.
        ll_check = np.array([
                # First instance.
                logsumexp(np.array([generation_scores[0, target_tokens[0].item()].item(), 2.0])) -
                logsumexp(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 2.0])),

                # Second instance.
                generation_scores[1, target_tokens[1].item()].item() -
                logsumexp(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])),

                # Third instance.
                logsumexp(np.array([2.0, 3.0])) -
                logsumexp(np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 2.0, 2.0, 3.0]))
        ])

        # This is what the selective_weights result should look like.
        selective_weights_check = np.stack([
                np.array([0., 1., 0.]),
                np.array([0., 0., 0.]),
                np.exp([2.0, float("-inf"), 3.0]) / (np.exp(2.0) + np.exp(3.0)),
        ])

        ll_actual, selective_weights_actual = self.model._get_ll_contrib(generation_scores,
                                                                         copy_scores,
                                                                         target_tokens,
                                                                         copy_indicators,
                                                                         copy_mask)

        np.testing.assert_almost_equal(ll_actual.data.numpy(),
                                       ll_check, decimal=6)

        np.testing.assert_almost_equal(selective_weights_actual.data.numpy(),
                                       selective_weights_check, decimal=6)
