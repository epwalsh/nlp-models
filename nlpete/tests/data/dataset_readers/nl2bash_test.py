import numpy as np

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from nlpete.data.dataset_readers import NL2BashDatasetReader


class TestNL2BashReader(AllenNlpTestCase):

    def setUp(self):
        super(TestNL2BashReader, self).setUp()
        self.reader = NL2BashDatasetReader("target_tokens")
        instances = self.reader.read("nlpete/tests/fixtures/nl2bash/train.tsv")
        self.instances = ensure_list(instances)

    def test_tokens(self):
        assert len(self.instances) == 3
        fields = self.instances[2].fields
        assert [t.text for t in fields["source_tokens"].tokens] == \
            ["@start@", "Extracts", " ", "a", " ", "bz", "2", " ", "file", ".", "@end@"]
        assert [t.text for t in fields["target_tokens"].tokens] == \
            ["@start@", "bunzip", "2", " ", "file", ".", "bz", "2", "@end@"]

    def test_target_to_source(self):
        target_to_source = self.instances[2].fields["target_to_source"]

        # shape should be (target_length, source_length - 2)
        assert target_to_source.array.shape == (9, 9)

        check = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],  # @START@
                          [0, 0, 0, 0, 0, 0, 0, 0, 0],  # bunzip
                          [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 2
                          [0, 1, 0, 1, 0, 0, 1, 0, 0],  # \s
                          [0, 0, 0, 0, 0, 0, 0, 1, 0],  # file
                          [0, 0, 0, 0, 0, 0, 0, 0, 1],  # .
                          [0, 0, 0, 0, 1, 0, 0, 0, 0],  # bz
                          [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 2
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]]) # @END@
        np.testing.assert_equal(target_to_source.array, check)

    def test_preprocess_target(self):
        # pylint: disable=protected-access
        tgt_str = "sudo find ."
        check = "find ."
        assert self.reader._preprocess_target(tgt_str) == check

        tgt_str = "$find ."
        check = "find ."
        assert self.reader._preprocess_target(tgt_str) == check

        tgt_str = "$ find ."
        check = "find ."
        assert self.reader._preprocess_target(tgt_str) == check

        tgt_str = "# find ."
        check = "find ."
        assert self.reader._preprocess_target(tgt_str) == check

        tgt_str = "ls -l | /bin/grep"
        check = "ls -l | grep"
        assert self.reader._preprocess_target(tgt_str) == check

        tgt_str = "ls -l | ~/bin/grep"
        check = "ls -l | ~/bin/grep"
        assert self.reader._preprocess_target(tgt_str) == check
