
from typing import List

import torch.nn as nn
from transformers import EvalPrediction

class ComputeMetrics():
    def __init__(self, trainer, tasks: List[str]):
        self.trainer = trainer
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

        gather_function = self.trainer.gather_function
        is_asr_mask = (gather_function((inputs['is_asr'])) == 1).bool()
        not_asr_mask = ~is_asr_mask

        attention_mask = gather_function((inputs['attention_mask']))

        current_asr_tokens = attention_mask[is_asr_mask].sum().item()

        self.asr_samples += current_asr_tokens
        self.tts_samples += attention_mask.sum().item() - current_asr_tokens

        self.asr_loss += self.criterion(predictions[is_asr_mask].flatten(0, 1), label_ids[is_asr_mask].flatten(0, 1)).item()
        self.tts_loss += self.criterion(predictions[not_asr_mask].flatten(0, 1), label_ids[not_asr_mask].flatten(0, 1)).item()

        metrics = {}

        if compute_result:
            metrics["asr_loss"] = self.asr_loss / self.asr_samples
            metrics["tts_loss"] = self.tts_loss / self.tts_samples

            self._reset_metrics_accumulation()

        return metrics
