from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("copynet")
class CopyNetPredictor(Predictor):
    """
    Predictor for the CopyNet model.
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source_string" : source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        source = json_dict["source_string"]
        return self._dataset_reader.text_to_instance(source)
