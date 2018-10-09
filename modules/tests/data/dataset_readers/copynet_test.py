import numpy as np

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from modules.data.dataset_readers import CopyNetDatasetReader


class TestCopyNetReader(AllenNlpTestCase):

    def setUp(self):
        super(TestCopyNetReader, self).setUp()
        self.reader = CopyNetDatasetReader()
        instances = self.reader.read("modules/tests/fixtures/copynet/copyover.tsv")
        self.instances = ensure_list(instances)

    def test_tokens(self):
        assert len(self.instances) == 2
        fields = self.instances[0].fields
        assert [t.text for t in fields["source_tokens"].tokens] == \
            ["@start@", "these", "tokens", "should", "be", "copied", "over", ":", "hello", "world", "@end@"]
        assert [t.text for t in fields["target_tokens"].tokens] == \
            ["@start@", "the", "tokens", "\"", "hello", "world", "\"", "were", "copied", "@end@"]

    def test_source_indices(self):
        source_indices = self.instances[0].fields["source_indices"]

        # shape should be (target_length, source_length - 2)
        assert source_indices.array.shape == (10, 9)

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
        np.testing.assert_equal(source_indices.array, check)
