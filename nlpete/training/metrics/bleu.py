from collections import Counter
import math
from typing import Iterable, Tuple, Dict, List, Set, Generator

from overrides import overrides
import torch

from allennlp.training.metrics import Metric


@Metric.register("bleu")
class BLEU(Metric):
    """
    Bilingual Evaluation Understudy (BLEU) is a common metric used for evaluating
    the quality of machine translations against a set of reference translations.
    This implementation only considers a reference set of size 1.
    """

    def __init__(self,
                 ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25)) -> None:
        self._ngram_weights = ngram_weights
        self._precision_matches: Dict[int, int] = Counter()
        self._precision_totals: Dict[int, int] = Counter()
        self._prediction_lengths = 0
        self._reference_lengths = 0

    @overrides
    def reset(self) -> None:
        self._precision_matches = Counter()
        self._precision_totals = Counter()
        self._prediction_lengths = 0
        self._reference_lengths = 0

    @staticmethod
    def _ngrams(tensor: torch.Tensor,
                ngram_size: int,
                exclude_indices: Set[int] = None) -> Generator[Tuple[int, ...], None, None]:
        start_upper_bound = max((min((ngram_size, tensor.size(-1) - 1)), 1))
        start_positions = range(start_upper_bound)
        for start in start_positions:
            for tensor_slice in tensor[:, start:].split(ngram_size, dim=-1):
                if tensor_slice.size(-1) < ngram_size:
                    break
                for row in tensor_slice:
                    ngram = tuple(x.item() for x in row)
                    if exclude_indices and any(x in ngram for x in exclude_indices):
                        continue
                    yield ngram

    def _get_modified_precision(self,
                                predicted_tokens: List[str],
                                reference_tokens: List[str],
                                ngram_size: int,
                                exclude_indices: Set[int] = None) -> Tuple[int, int]:
        predicted_ngram_counts = Counter(self._ngrams(predicted_tokens, ngram_size, exclude_indices))
        reference_ngram_counts = Counter(self._ngrams(reference_tokens, ngram_size, exclude_indices))
        clipped_matches = {ngram: min(count, reference_ngram_counts[ngram])
                           for ngram, count in predicted_ngram_counts.items()}
        return sum(clipped_matches.values()), sum(predicted_ngram_counts.values())

    def _get_brevity_penalty(self) -> float:
        if self._prediction_lengths > self._reference_lengths:
            return 1.0
        if self._reference_lengths == 0 or self._prediction_lengths == 0:
            return 0.0
        return math.exp(1.0 - self._reference_lengths / self._prediction_lengths)

    @staticmethod
    def _get_valid_tokens_mask(tensor: torch.Tensor, exclude_indices: Set[int]):
        valid_tokens_mask = torch.ones(tensor.size(), dtype=torch.uint8)
        for index in exclude_indices:
            valid_tokens_mask = valid_tokens_mask & (tensor != index)
        return valid_tokens_mask

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_targets: torch.Tensor,
                 exclude_indices: Set[int] = None) -> None:
        """
        Parameters
        ----------
        predictions : ``torch.LongTensor``, required
            Batched predicted tokens of shape `(batch_size, max_sequence_length)`.
        references : ``torch.LongTensor``, required
            Batched reference (gold) translations with shape `(batch_size, max_gold_sequence_length)`.
        exclude_indices : ``Set[int]``, optional (default = None)
            Indices to exclude when calculating ngrams.

        Returns
        -------
        None
        """
        for ngram_size, _ in enumerate(self._ngram_weights, start=1):
            precision_matches, precision_totals = self._get_modified_precision(
                    predictions, gold_targets, ngram_size, exclude_indices)
            self._precision_matches[ngram_size] += precision_matches
            self._precision_totals[ngram_size] += precision_totals
        if not exclude_indices:
            self._prediction_lengths += predictions.size(-1)
            self._reference_lengths += gold_targets.size(-1)
        else:
            valid_predictions_mask = self._get_valid_tokens_mask(predictions, exclude_indices)
            self._prediction_lengths += valid_predictions_mask.sum(-1).max().item()
            valid_gold_targets_mask = self._get_valid_tokens_mask(gold_targets, exclude_indices)
            self._reference_lengths += valid_gold_targets_mask.sum(-1).max().item()

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        brevity_penalty = self._get_brevity_penalty()
        ngram_scores = (weight * (math.log(self._precision_matches[n] + 1e-13) -
                                  math.log(self._precision_totals[n] + 1e-13))
                        for n, weight in enumerate(self._ngram_weights, start=1))
        bleu = brevity_penalty * math.exp(sum(ngram_scores))
        if reset:
            self.reset()
        return {"BLEU": bleu}
