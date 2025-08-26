import torch

from salt.data.base_datasets import SaltDataset


class SaltTextToSpeechDataset(SaltDataset):
    def __getitem__(self, idx):
        row = self.dataset[idx]
        text_tokens = self.prepare_text_tokens(row)
        audio_tokens = self.prepare_audio_tokens(row)

        tokens = torch.cat((text_tokens, audio_tokens), dim=1)
        tokens = tokens.squeeze(0)

        audio_start = text_tokens.shape[0]
        labels = tokens.clone()
        labels[:audio_start] = -100

        return {
            "input_ids": tokens,
            "attention_mask": torch.ones(len(tokens)),
            "labels": labels,
        }


class SaltSpeechRecognitionDataset(SaltDataset):
    def __getitem__(self, idx):
        row = self.dataset[idx]
        text_tokens = self.prepare_text_tokens(row)
        audio_tokens = self.prepare_audio_tokens(row)

        tokens = torch.cat([audio_tokens, text_tokens], dim=1)
        tokens = tokens.squeeze(0)

        text_start = audio_tokens.shape[0]
        labels = tokens.clone()
        labels[:text_start] = -100

        return {
            "input_ids": tokens,
            "attention_mask": torch.ones(len(tokens)),
            "labels": labels,
        }
