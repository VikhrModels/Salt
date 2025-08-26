import soundfile as sf
import torch
from omegaconf import DictConfig

from salt.tokenization.audio_tokenizer import (
    SpeechTokenizerWrapper,
    WavTokenizerWrapper,
    BigCodecWrapper,
)


def test_speech_tokenizer():
    data, sr = sf.read("test.wav")

    target_sr = 16_000

    cfg = DictConfig(
        {
            "tokenizer": {
                "n_codebooks": 8,
                "config_path": "../checkpoints/speech/config.json",
                "ckpt_path": "../checkpoints/speech/SpeechTokenizer.pt",
                "sample_rate": target_sr,
            },
            "start_audio_token_id": None,
            "end_audio_token_id": None,
            "n_text_tokens": 100_000,
            "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        }
    )
    speech_tokenizer = SpeechTokenizerWrapper(cfg)
    codes = speech_tokenizer.encode({"array": data, "sample_rate": sr})
    reconstructed = speech_tokenizer.decode(codes.t().reshape(1, -1))
    reconstructed = reconstructed.detach().cpu().numpy()

    sf.write("speech.wav", reconstructed.ravel(), target_sr)


def test_wav_tokenizer():
    data, sr = sf.read("test.wav")

    target_sr = 24_000

    cfg = DictConfig(
        {
            "tokenizer": {
                "config_path": "../checkpoints/wav/config.yaml",
                "ckpt_path": "../checkpoints/wav/wavtokenizer_large_unify_600_24k.ckpt",
                "sample_rate": target_sr,
            },
            "start_audio_token_id": None,
            "end_audio_token_id": None,
            "n_text_tokens": 100_000,
            "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        }
    )
    wav_tokenizer = WavTokenizerWrapper(cfg)
    codes = wav_tokenizer.encode({"array": data, "sample_rate": sr})
    reconstructed = wav_tokenizer.decode(codes.reshape(1, -1))
    reconstructed = reconstructed.detach().cpu().numpy()

    sf.write("wav.wav", reconstructed.ravel(), target_sr)


def test_bigcodec_tokenizer():
    data, sr = sf.read("test.wav")

    target_sr = 16_000

    cfg = DictConfig(
        {
            "tokenizer": {
                "ckpt_path": "../checkpoints/bigcodec/bigcodec.pt",
                "sample_rate": target_sr,
            },
            "start_audio_token_id": None,
            "end_audio_token_id": None,
            "n_text_tokens": 100_000,
            "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        }
    )
    bigcodec_tokenizer = BigCodecWrapper(cfg)
    codes = bigcodec_tokenizer.encode({"array": data, "sample_rate": sr})
    reconstructed = bigcodec_tokenizer.decode(codes.reshape(1, -1))
    reconstructed = reconstructed.detach().cpu().numpy()

    sf.write("bigcodec.wav", reconstructed.ravel(), target_sr)
