import torch

from salt.data.task_datasets import SaltTextToSpeechDataset, SaltSpeechRecognitionDataset


class SaltTextToSpeechDatasetWithVoiceDescription(SaltTextToSpeechDataset):
    def prepare_text_tokens(self, inputs: dict):
        text = inputs["text"]
        voice_description = inputs["voice_description"]

        prompt = "Say '{text}' with {voice_dsc}".format(
            text=text, voice_dsc=voice_description
        )
        tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return torch.cat(
            [
                self.bos_id, tokens["input_ids"], self.eos_id,
            ],
            dim=1
        )


class SaltSpeechRecognitionDatasetWithVoiceDescription(SaltSpeechRecognitionDataset):
    def prepare_text_tokens(self, inputs: dict):
        text = inputs["text"]
        voice_description = inputs["voice_description"]

        prompt = "'{text}' is said with {voice_dsc}".format(
            text=text, voice_dsc=voice_description
        )
        tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return torch.cat(
            [
                self.bos_id, tokens["input_ids"], self.eos_id,
            ],
            dim=1
        )
