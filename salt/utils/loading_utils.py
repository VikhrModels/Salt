import torch
from omegaconf import DictConfig

from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from salt.tokenization import AudioTokenizerType
from salt.tokenization.audio_tokenizer import SpeechTokenizerWrapper, WavTokenizerWrapper, BigCodecWrapper


def load_model(config: DictConfig):
    torch_dtype = getattr(torch, config.torch_dtype)

    if config.checkpoint_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
            config.checkpoint_path,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
        )

    model.config.use_cache = False
    return model


def load_audio_tokenizer(config: DictConfig):
    audio_tokenizer_type = AudioTokenizerType[config.quantizer_type]
    if audio_tokenizer_type == AudioTokenizerType.speech:
        quantizer = SpeechTokenizerWrapper(config)
    elif audio_tokenizer_type == AudioTokenizerType.wav:
        quantizer = WavTokenizerWrapper(config)
    elif audio_tokenizer_type == AudioTokenizerType.bigcodec:
        quantizer = BigCodecWrapper(config)
    else:
        raise ValueError(f"Unknown audio tokenizer type {audio_tokenizer_type}.")

    return quantizer


def prepare_tokenizer(config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            {"pad_token": "[PAD]"}
        )
        tokenizer.pad_token = "[PAD]"
        config.n_special_tokens += 1

    tokenizer.add_special_tokens(
        {"additional_special_tokens": [config.start_audio_token, config.end_audio_token]}
    )
    start_audio_token_id = tokenizer._convert_token_to_id_with_added_voc(
        config.start_audio_token
    )
    end_audio_token_id = tokenizer._convert_token_to_id_with_added_voc(config.end_audio_token)
    config.start_audio_token_id = start_audio_token_id
    config.end_audio_token_id = end_audio_token_id

    return tokenizer


def prepare_model_and_tokenizer(config: DictConfig) -> tuple[nn.Module, PreTrainedTokenizer]:
    model = load_model(config.training)
    tokenizer = prepare_tokenizer(config.training)

    model.resize_token_embeddings(len(tokenizer) + config.tokenizer.n_audio_tokens)
    return model, tokenizer

