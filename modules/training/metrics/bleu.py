from collections import Counter
import math
from typing import Iterable, Tuple, Dict, List

from overrides import overrides
from nltk import ngrams

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
    def _get_modified_precision(predicted_tokens: List[str],  # pylint: disable=invalid-name
                                reference_tokens: List[str],
                                n: int) -> Tuple[int, int]:
        predicted_ngram_counts = Counter(ngrams(predicted_tokens, n))
        reference_ngram_counts = Counter(ngrams(reference_tokens, n))
        clipped_matches = {ngram: min(count, reference_ngram_counts[ngram])
                           for ngram, count in predicted_ngram_counts.items()}
        return sum(clipped_matches.values()), sum(predicted_ngram_counts.values())

    def _get_brevity_penalty(self) -> float:
        if self._prediction_lengths > self._reference_lengths:
            return 1.0
        if self._reference_lengths == 0 or self._prediction_lengths == 0:
            return 0.0
        return math.exp(1.0 - self._reference_lengths / self._prediction_lengths)

    @overrides
    def __call__(self,
                 predictions: List[List[str]],
                 references: List[List[str]]) -> None:
        """
        Parameters
        ----------
        predictions : ``List[List[str]]``
            Batched predicted tokens.
        references : ``List[List[str]]``
            Batched reference translations.

        Returns
        -------
        None
        """
        for predicted_tokens, reference_tokens in zip(predictions, references):
            for n, _ in enumerate(self._ngram_weights, start=1):  # pylint: disable=invalid-name
                precision_matches, precision_totals = self._get_modified_precision(
                        predicted_tokens, reference_tokens, n)
                self._precision_matches[n] += precision_matches
                self._precision_totals[n] += precision_totals
            self._prediction_lengths += len(predicted_tokens)
            self._reference_lengths += len(reference_tokens)

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
