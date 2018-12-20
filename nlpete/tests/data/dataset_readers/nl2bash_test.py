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
