
from typing import List

import torch.nn as nn
from transformers import EvalPrediction

class ComputeMetrics():
    def __init__(self, tasks: List[str]):
        self.tasks = tasks

        self._reset_metrics_accumulation()

        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def _reset_metrics_accumulation(self):
        self.asr_samples = 0
        self.tts_samples = 0
        self.asr_loss = 0
        self.tts_loss = 0

    def __call__(self, eval_prediction: EvalPrediction, compute_result: bool):

        predictions = eval_prediction.predictions
        label_ids = eval_prediction.label_ids
        inputs = eval_prediction.inputs

        assert 'is_asr' in inputs

        is_asr_mask = inputs['is_asr'] == 1
        not_asr_mask = inputs['is_asr'] == 0

        current_asr_tokens = inputs['attention_mask'][is_asr_mask].sum().item()


        self.asr_samples += current_asr_tokens
        self.tts_samples += inputs['attention_mask'].sum().item() - current_asr_tokens


        self.asr_loss += self.criterion(predictions[is_asr_mask].flatten(0, 1), label_ids[is_asr_mask].flatten(0, 1)).item()
        self.tts_loss += self.criterion(predictions[not_asr_mask].flatten(0, 1), label_ids[not_asr_mask].flatten(0, 1)).item()

        matrics = {}

        if compute_result:
            matrics["asr_loss"] = self.asr_loss / self.asr_samples
            matrics["tts_loss"] = self.tts_loss / self.tts_samples

            self._reset_metrics_accumulation()

        return matrics
