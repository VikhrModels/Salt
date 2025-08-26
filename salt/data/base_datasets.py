from abc import ABC, abstractmethod

from omegaconf import DictConfig

import torch
from torch.utils.data import Dataset
from datasets import Dataset as SizedDataset
from transformers import PreTrainedTokenizer

from salt.tokenization.audio_tokenizer import AudioTokenizer


class SaltDataset(ABC, Dataset):
    def __init__(self, config: DictConfig, dataset: SizedDataset, tokenizer: PreTrainedTokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

        self.start_audio_token_id = config.training.start_audio_token_id
        self.end_audio_token_id = config.training.end_audio_token_id
        self.bos_id = config.training.bos_id
        self.eos_id = config.training.eos_id

        self.n_codebooks = config.tokenization.n_codebooks
        self.n_special_tokens = config.training.n_special_tokens

    def __len__(self):
        return len(self.dataset)

    def prepare_text_tokens(self, inputs: dict) -> torch.Tensor:
        text = inputs["text"]
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return torch.cat(
            [
                self.bos_id, tokens["input_ids"], self.eos_id,
            ],
            dim=1
        )

    def prepare_audio_tokens(self, inputs: dict) -> torch.Tensor:
        audio_tokens = torch.tensor(inputs["audio_tokens"])
        audio_tokens = audio_tokens[:self.n_codebooks]
        audio_tokens = audio_tokens + len(self.tokenizer)
        return torch.cat(
            [
                self.start_audio_token_id, audio_tokens, self.end_audio_token_id,
            ],
            dim=1
        )

    @abstractmethod
    def __getitem__(self, index: int) -> torch.Tensor:
        pass


class SaltAudioDataset(SaltDataset):
    def __getitem__(self, idx):
        row = self.dataset[idx]
        audio_tokens = self.prepare_audio_tokens(row["audio_tokens"])
        tokens = audio_tokens.squeeze(0)

        return {
            "input_ids": tokens,
            "attention_mask": torch.ones(len(tokens)),
            "labels": tokens.clone(),
        }


class SaltTextDataset(SaltDataset):
    def __getitem__(self, idx):
        row = self.dataset[idx]
        text_tokens = self.prepare_text_tokens(row["text"])
        tokens = text_tokens.squeeze(0)

        return {
            "input_ids": tokens,
            "attention_mask": torch.ones(len(tokens)),
            "labels": tokens.clone(),
        }
