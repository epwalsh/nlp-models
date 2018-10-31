# pylint: disable=protected-access

import numpy as np

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from allennlp.data import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN

from nlpete.data.dataset_readers import CopyNetDatasetReader  # pylint: disable=unused-import


class TestCopyNetReader(AllenNlpTestCase):

    def setUp(self):
        super(TestCopyNetReader, self).setUp()
        params = Params.from_file("nlpete/tests/fixtures/copynet/experiment.json")
        self.reader = DatasetReader.from_params(params["dataset_reader"])
        instances = self.reader.read("nlpete/tests/fixtures/copynet/copyover.tsv")
        self.instances = ensure_list(instances)
        self.vocab = Vocabulary.from_params(params=params["vocabulary"], instances=instances)

    def test_vocab_namespaces(self):
        assert self.vocab.get_vocab_size("target_tokens") > 5

    def test_instances(self):
        assert len(self.instances) == 2
        assert set(self.instances[0].fields.keys()) == set(("source_tokens", "source_to_source",
                                                            "target_tokens", "target_to_source",
                                                            "source_to_target", "metadata"))

    def test_tokens(self):
        fields = self.instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == \
            ["@start@", "these", "tokens", "should", "be", "copied", "over", ":", "hello", "world", "@end@"]
        assert fields["metadata"]["source_tokens"] == \
            ["these", "tokens", "should", "be", "copied", "over", ":", "hello", "world"]
        assert [t.text for t in fields["target_tokens"].tokens] == \
            ["@start@", "the", "tokens", "\"", "hello", "world", "\"", "were", "copied", "@end@"]
        assert fields["metadata"]["target_tokens"] == \
            ["the", "tokens", "\"", "hello", "world", "\"", "were", "copied"]

    def test_target_to_source_array(self):
        target_to_source = self.instances[0].fields["target_to_source"]

        # shape should be (target_length, source_length - 2)
        assert target_to_source.array.shape == (10, 9)

        check = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],  # @START@
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],  # the
                          [0, 1, 0, 0, 0, 0, 0, 0, 0],  # tokens
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],  # "
                          [0, 0, 0, 0, 0, 0, 0, 1, 0],  # hello
                          [0, 0, 0, 0, 0, 0, 0, 0, 1],  # world
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],  # "
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],  # were
                          [0, 0, 0, 0, 1, 0, 0, 0, 0],  # copied
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]]) # @END@
        np.testing.assert_equal(target_to_source.array, check)

    def test_source_to_target(self):
        source_to_target_field = self.instances[0].fields["source_to_target"]
        source_to_target_field.index(self.vocab)
        tensor = source_to_target_field.as_tensor(source_to_target_field.get_padding_lengths())
        check = np.array([self.vocab.get_token_index("these", "target_tokens"),
                          self.vocab.get_token_index("tokens", "target_tokens"),
                          self.vocab.get_token_index("should", "target_tokens"),
                          self.vocab.get_token_index("be", "target_tokens"),
                          self.vocab.get_token_index("copied", "target_tokens"),
                          self.vocab.get_token_index("over", "target_tokens"),
                          self.vocab.get_token_index(":", "target_tokens"),
                          self.vocab.get_token_index("hello", "target_tokens"),
                          self.vocab.get_token_index("world", "target_tokens")])
        np.testing.assert_equal(tensor.numpy(), check)
        assert tensor[1].item() != self.vocab.get_token_index(DEFAULT_OOV_TOKEN, "target_tokens")

    def test_source_to_source_array(self):
        tokens = ["@START@", "a", "cat", "is", "a", "cat", "@END@"]
        result = self.reader._create_source_to_source_array([Token(x) for x in tokens])
        check = np.array([[1, 0, 0, 1, 0],  # a
                          [0, 1, 0, 0, 1],  # cat
                          [0, 0, 1, 0, 0],  # is
                          [1, 0, 0, 1, 0],  # a
                          [0, 1, 0, 0, 1]]) # cat
        np.testing.assert_equal(result, check)
