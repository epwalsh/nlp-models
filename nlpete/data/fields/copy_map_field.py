from typing import Dict, List, Optional

from overrides import overrides
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.data import Field


class CopyMapField(Field[torch.Tensor]):

    def __init__(self,
                 source_tokens: List[Token],
                 target_namespace: str) -> None:
        self._source_tokens = source_tokens
        self._target_namespace = target_namespace
        self._mapping_array: Optional[List[List[int]]] = None

    @overrides
    def index(self, vocab: Vocabulary):
        self._mapping_array = [vocab.get_token_index(x.text, self._target_namespace)
                               for x in self._source_tokens]

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {"num_tokens": len(self._source_tokens)}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_length = padding_lengths["num_tokens"]
        padded_tokens = pad_sequence_to_length(self._mapping_array, desired_length)
        tensor = torch.LongTensor(padded_tokens)
        return tensor

    @overrides
    def empty_field(self) -> 'CopyMapField':
        return CopyMapField([], self._target_namespace)
