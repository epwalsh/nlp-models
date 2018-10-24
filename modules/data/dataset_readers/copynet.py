import logging
from typing import List, Dict

import numpy as np
from overrides import overrides

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("copynet")
class CopyNetDatasetReader(Seq2SeqDatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``CopyNet`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string><tab><target_sequence_string>.

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    """

    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        # The only reason we override __init__ is so that we can ensure `source_add_start_token`
        # is True. This is because the CopyNet model always assumes the start token
        # will be part of the source sentence.
        super().__init__(source_tokenizer=source_tokenizer,
                         target_tokenizer=target_tokenizer,
                         source_token_indexers=source_token_indexers,
                         target_token_indexers=target_token_indexers,
                         source_add_start_token=True,
                         lazy=lazy)

    def _preprocess_source(self, source_string: str) -> str:  # pylint: disable=no-self-use
        """
        Apply preprocessing steps to the source string. Right now this doesn't
        do anything because it's meant to be overridden by subclasses.
        """
        return source_string

    def _preprocess_target(self, target_string: str) -> str:  # pylint: disable=no-self-use
        """
        Apply preprocessing steps to the target string. Right now this doesn't
        do anything because it's meant to be overridden by subclasses.
        """
        return target_string

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        source_string = self._preprocess_source(source_string)
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_string is not None:
            target_string = self._preprocess_target(target_string)
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            # For each token in the target sentence, we keep track of the index
            # of every token in the source sentence that matches.
            copy_indicator_array: List[List[int]] = []
            for tgt_tok in tokenized_target[1:-1]:
                source_index_list: List[int] = []
                for src_tok in tokenized_source[1:-1]:
                    if tgt_tok.text == src_tok.text:
                        source_index_list.append(1)
                    else:
                        source_index_list.append(0)
                copy_indicator_array.append(source_index_list)
            copy_indicator_array.insert(0, [0] * len(tokenized_source[1:-1]))
            copy_indicator_array.append([0] * len(tokenized_source[1:-1]))

            # shape: (target_length, source_length)
            copy_indicator_field = ArrayField(np.array(copy_indicator_array))

            return Instance({"source_tokens": source_field,
                             "target_tokens": target_field,
                             "copy_indicators": copy_indicator_field})
        else:
            return Instance({'source_tokens': source_field})
