import logging
from typing import List, Dict

import numpy as np
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from modules.data.fields import CopyMapField


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("copynet")
class CopyNetDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``CopyNet`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string><tab><target_sequence_string>.

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    target_namespace : ``str``, required
        The vocab namespace for the targets. This needs to be passed to the dataset reader
        in order to construct the CopyMapField.
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
                 target_namespace: str,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._target_namespace = target_namespace
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")
                if not line:
                    continue
                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                source_sequence, target_sequence = line_parts
                yield self.text_to_instance(source_sequence, target_sequence)

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

    @staticmethod
    def _create_copy_indicator_array(tokenized_source: List[Token],
                                     tokenized_target: List[Token]) -> np.array:
        copy_indicator_array: List[List[int]] = []
        for target_token in tokenized_target[1:-1]:
            source_index_list: List[int] = [int(target_token.text.lower() == source_token.text.lower())
                                            for source_token in tokenized_source[1:-1]]
            copy_indicator_array.append(source_index_list)
        copy_indicator_array.insert(0, [0] * len(tokenized_source[1:-1]))
        copy_indicator_array.append([0] * len(tokenized_source[1:-1]))
        return np.array(copy_indicator_array)

    @staticmethod
    def _create_source_duplicates_array(tokenized_source: List[Token]) -> np.array:
        out_array: List[List[int]] = []
        for token in tokenized_source[1:-1]:
            array_slice: List[int] = [int(token.text.lower() == other.text.lower())
                                      for other in tokenized_source[1:-1]]
            out_array.append(array_slice)
        return np.array(out_array)

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        source_string = self._preprocess_source(source_string)
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        # For token in the source sentence, we store a sparse array containing
        # indicators for each other source token that matches. This gives us
        # a matrix of shape `(source_length, source_length)` where the (i,j)th entry
        # is a 1 if the ith token matches the jth token.
        source_duplicates_array = self._create_source_duplicates_array(tokenized_source)
        source_duplicates_field = ArrayField(source_duplicates_array)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        target_pointer_field = CopyMapField(tokenized_source[1:-1], self._target_namespace)

        fields_dict = {
                "source_tokens": source_field,
                "source_duplicates": source_duplicates_field,
                "target_pointers": target_pointer_field,
        }

        if target_string is not None:
            target_string = self._preprocess_target(target_string)
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            # For each token in the target sentence, we keep track of the index
            # of every token in the source sentence that matches.
            copy_indicator_array = self._create_copy_indicator_array(tokenized_source,
                                                                     tokenized_target)
            # shape: (target_length, source_length)
            copy_indicator_field = ArrayField(copy_indicator_array)

            fields_dict["target_tokens"] = target_field
            fields_dict["copy_indicators"] = copy_indicator_field

        return Instance(fields_dict)
