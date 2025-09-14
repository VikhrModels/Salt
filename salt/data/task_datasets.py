import torch

from salt.data.base_datasets import SaltDataset


class SaltTextToSpeechDataset(SaltDataset):
    def _get_positional_ids(
        self, text_tokens: torch.Tensor, audio_tokens: torch.Tensor
    ):
        text_pos_ids = torch.arange(text_tokens.shape[-1])
        frame_length = audio_tokens.shape[-1] // text_tokens.shape[-1]
        audio_pos_ids = text_pos_ids.repeat_interleave(frame_length)
        if audio_tokens.shape[-1] != audio_tokens.shape[-1]:
            pad_length = audio_tokens.shape[-1] - audio_tokens.shape[-1]
            padding = torch.ones(pad_length) * text_pos_ids.shape[-1]
            audio_pos_ids = torch.cat([audio_pos_ids, padding], dim=-1)

        return torch.cat([text_pos_ids, audio_pos_ids], dim=-1)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        text_tokens = self.prepare_text_tokens(row)
        audio_tokens = self.prepare_audio_tokens(row)

        tokens = torch.cat((text_tokens, audio_tokens), dim=1)
        tokens = tokens.squeeze(0)

        audio_start = text_tokens.shape[-1]
        labels = tokens.clone()
        labels[:audio_start] = -100

        positional_ids = self._get_positional_ids(text_tokens, audio_tokens)

        return {
            "input_ids": tokens,
            "attention_mask": torch.ones(len(tokens)),
            "labels": labels,
            "positional_ids": positional_ids,
        }


class SaltSpeechRecognitionDataset(SaltDataset):
    def _get_position_ids(self, text_tokens: torch.Tensor, audio_tokens: torch.Tensor):
        audio_pos_ids = torch.arange(audio_tokens.shape[-1])
        frame_length = audio_tokens.shape[-1] // text_tokens.shape[-1]
        text_pos_ids = audio_pos_ids[:, ::frame_length]
        if text_tokens.shape[-1] != text_tokens.shape[-1]:
            text_pos_ids = text_pos_ids[:, : text_tokens.shape[-1]]

        return torch.cat([audio_pos_ids, text_pos_ids], dim=-1)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        text_tokens = self.prepare_text_tokens(row)
        audio_tokens = self.prepare_audio_tokens(row)

        tokens = torch.cat([audio_tokens, text_tokens], dim=1)
        tokens = tokens.squeeze(0)

        text_start = audio_tokens.shape[0]
        labels = tokens.clone()
        labels[:text_start] = -100

        positional_ids = self._get_positional_ids(text_tokens, audio_tokens)

        return {
            "input_ids": tokens,
            "attention_mask": torch.ones(len(tokens)),
            "labels": labels,
            "position_ids": positional_ids,
        }
