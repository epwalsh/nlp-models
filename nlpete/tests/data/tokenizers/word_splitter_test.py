from typing import List

from allennlp.common.testing import AllenNlpTestCase

from nlpete.data.tokenizers.word_splitter import NL2BashWordSplitter


class TestNL2BashWordSplitter(AllenNlpTestCase):

    def setUp(self):
        super().setUp()
        self.word_splitter = NL2BashWordSplitter()

    def _assert_tokens_equal(self, sentence: str, token_check: List[str]):
        result = [x.text for x in self.word_splitter.split_words(sentence)]
        assert result == token_check

    def test_split_words(self):
        sentence = "/bin/find $dir"
        token_check = ["/", "bin", "/", "find", " ", "$", "dir"]
        self._assert_tokens_equal(sentence, token_check)
