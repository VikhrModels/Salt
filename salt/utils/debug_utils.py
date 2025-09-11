import soundfile as sf
from omegaconf import DictConfig

import torch

from salt.tokenization.audio_tokenizer import AudioTokenizer
from salt.utils.loading_utils import load_audio_tokenizer


def verify_audio_reconstruction(quantizer: AudioTokenizer, audio_dict: dict):
    codes = quantizer.encode(audio_dict)
    reconstructed = quantizer.decode(codes)
    reconstructed = reconstructed.detach().cpu().numpy()

    sf.write("reconstructed.wav", reconstructed.ravel(), quantizer.sample_rate)


def verify_audio_from_tokens_reconstruction(config: DictConfig, tokens: torch.Tensor):
    quantizer = load_audio_tokenizer(config)
    reshaped_tokens = tokens.reshape(1, -1)
    audio = quantizer.decode(reshaped_tokens)
    audio = audio.detach().cpu().numpy()

    sf.write("/workspace/Salt/reconstructed.wav", audio.ravel(), quantizer.sample_rate)
