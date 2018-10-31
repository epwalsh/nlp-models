import re
from typing import List

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter


@WordSplitter.register('nl2bash')
class NL2BashWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` which keeps runs of (unicode) letters, digits, and whitespace
    together, while every other non-whitespace character becomes a separate word.
    """

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        # We use the [^\W\d_] pattern as a trick to match unicode letters
        tokens = [Token(m.group(), idx=m.start())
                  for m in re.finditer(r'[^\W\d_]+|\s+|\d+|\S', sentence)]
        return tokens
