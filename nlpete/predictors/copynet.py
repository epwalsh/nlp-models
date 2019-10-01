import warnings

from overrides import overrides

from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("copynet")
class CopyNetPredictor(Predictor):
    """
    Predictor for the CopyNet model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        warnings.warn(
            "The 'copynet' predictor has been deprecated in favor of "
            "the 'seq2seq' predictor.",
            DeprecationWarning,
        )

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source_string": source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        source = json_dict["source_string"]
        return self._dataset_reader.text_to_instance(source)
