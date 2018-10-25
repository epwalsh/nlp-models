from typing import Dict, List, Optional

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.data.tokenizers import Token
from allennlp.data import Field


class CopyMapField(Field[torch.Tensor]):

    def __init__(self,
                 source_tokens: List[Token],
                 source_namespace: str,
                 target_namespace: str) -> None:
        self._source_tokens = source_tokens
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._mapping_array: Optional[List[List[int]]] = None

    @overrides
    def index(self, vocab: Vocabulary):
        pass

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        pass
