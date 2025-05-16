import argparse
import hashlib

import os
from typing import Any

from dotenv import load_dotenv

import numpy as np
import yaml

from datasets import DatasetDict
import torch

from WavTokenizer.encoder.utils import convert_audio
from WavTokenizer.decoder.pretrained import WavTokenizer
from BigCodec.vq.codec_encoder import CodecEncoder
from BigCodec.vq.codec_decoder import CodecDecoder

from src.quantize.wavtokenizer import quantize_wavtokenizer_batch

from src.utils.data import DATASET_2_LOAD_FUNCTION
from src.utils.decoding import (
    decode_audio_wav,
    decode_audio_speech,
    decode_audio_fish,
    decode_audio_bigcodec,
)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

parser = argparse.ArgumentParser(description="Train a model with configuration.")
parser.add_argument(
    "--config", type=str, help="Path to the config.yaml file", required=True
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="If set, example of reconstructed audio is saved",
)
args = parser.parse_args()

# Load config
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

path_to_cache = config["path_to_cache"]

config_path = config["quantizer_config_path"]
ckpt_path = config["quantizer_ckpt_path"]
quantizer_type = config["quantizer_type"]

data = config["raw_data"]
prepared_data_path = config["prepared_data_path"]

device = "cuda:0"


class BigCodecTokenizer:
    def __init__(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        encoder = CodecEncoder()
        encoder.load_state_dict(ckpt["CodecEnc"])
        self.encoder = encoder.eval().cuda()

        decoder = CodecDecoder()
        decoder.load_state_dict(ckpt["generator"])
        self.decoder = decoder.eval().cuda()

    def encode(self, wav):
        vq_emb = self.encoder(wav.unsqueeze(1))
        _, vq_code, _ = self.decoder(vq_emb, vq=True)
        return vq_code


def resample(audio: np.ndarray, sr: int, target_sr: int):
    audio = torch.tensor(audio, dtype=torch.float32)
    audio = audio.unsqueeze(0)

    if sr != target_sr:
        # 1 as last arg corresponds to mono audio
        audio = convert_audio(audio, sr, target_sr, 1)

    return audio


def quantize_speechtokenizer(row: dict[str, Any], quantizer):
    from speechtokenizer import SpeechTokenizer

    quantizer: SpeechTokenizer
    audio_data, sample_rate = row["audio"]["array"], row["audio"]["sampling_rate"]

    audio = resample(audio_data, sample_rate, quantizer.sample_rate)
    audio = audio.view(1, 1, len(audio))

    codes = quantizer.encode(audio)
    codes = codes.squeeze(1)
    codes = codes.cpu()

    return {"audio_tokens": codes.numpy()}


def quantize_wavtokenizer(row: dict[str, Any], quantizer):
    audio_data, sample_rate = row["audio"]["array"], int(row["audio"]["sampling_rate"])

    audio = resample(audio_data, sample_rate, 24000)
    bandwidth_id = torch.tensor([0])

    audio = audio.to(quantizer.head.out.weight.device)
    _, codes = quantizer.encode_infer(audio, bandwidth_id=bandwidth_id)
    codes = codes.squeeze(1)
    codes = codes.cpu()

    return {"audio_tokens": codes.numpy()}


def quantize_bigcodec_tokenizer(row: dict[str, Any], quantizer: BigCodecTokenizer):
    audio_data, sample_rate = row["audio"]["array"], row["audio"]["sampling_rate"]

    audio = resample(audio_data, sample_rate, 16000)
    codes = quantizer.encode(audio)
    codes = codes.squeeze(0, 1)
    codes = codes.cpu()

    return {"audio_tokens": codes.numpy()}


def quantize_bigcodec_tokenizer_batched(
    row: dict[str, Any], quantizer: BigCodecTokenizer
):
    sample_rate = row["audio"][0]["sampling_rate"]
    audio_arrays = []
    for arr in row["audio"]:
        resampled = resample(arr["array"], sample_rate, 16000)
        audio_arrays.append(resampled)

    max_length = max(array.shape[1] for array in audio_arrays)

    padded_audio_tensors = [
        torch.nn.functional.pad(
            tensor, (0, max_length - tensor.shape[1]), mode="constant", value=0
        )
        for tensor in audio_arrays
    ]

    audio = torch.stack(padded_audio_tensors, dim=0).squeeze(1)

    codes = quantizer.encode(audio)
    codes = codes.squeeze(0)
    codes = codes.cpu()

    return {"audio_tokens": codes.numpy()}


def quantize_fishtokenizer(row: dict[str, Any], quantizer):
    audio_data, sample_rate = row["audio"]["array"], row["audio"]["sampling_rate"]
    text = row["text"]

    audio = resample(audio_data, sample_rate, quantizer.sample_rate)
    audio = audio.unsqueeze(0)

    audio_tokens = quantizer.encode_audio(audio)
    semantic_tokens = quantizer.encode_text(text)

    audio_tokens = audio_tokens.cpu()
    semantic_tokens = semantic_tokens.cpu()

    return {
        "audio_tokens": audio_tokens.numpy(),
        "semantic_tokens": semantic_tokens.numpy(),
    }


def verify_decoding(example, quantizer, quantizer_type: str):
    if quantizer_type == "speech":
        codes = quantize_speechtokenizer(example, quantizer)["audio_tokens"]
        codes = torch.tensor(codes, dtype=torch.float32, device=device)
        flattened = codes.t().contiguous().view(1, -1)

        audio = decode_audio_speech(
            flattened,
            quantizer,
            quantizer.quantizer.bins + 1,
            codes.shape[-1],
        )

    elif quantizer_type == "wav":
        codes = quantize_wavtokenizer(example, quantizer)["audio_tokens"]
        codes = torch.tensor(codes, dtype=torch.long, device=device)

        audio = decode_audio_wav(
            codes,
            quantizer,
            quantizer.feature_extractor.encodec.quantizer.bins + 1,
            1,
        )

    elif quantizer_type == "big-codec":
        codes = quantize_bigcodec_tokenizer(example, quantizer)["audio_tokens"]
        codes = torch.tensor(codes, dtype=torch.long, device=device)
        codes = codes.view(1, -1, 1)

        audio = decode_audio_bigcodec(
            codes,
            quantizer,
            quantizer.decoder.quantizer.layers[0].codebook_size,
            quantizer.decoder.quantizer.num_quantizers,
        )

    elif quantizer_type == "fish":
        codes = quantize_fishtokenizer(example, quantizer)["audio_tokens"]
        codes = torch.tensor(codes, dtype=torch.long, device=device)
        flattened = codes.t().contiguous().view(1, -1)

        audio = decode_audio_fish(
            flattened,
            quantizer,
            quantizer.codebook_size + 1,
            quantizer.num_codebooks,
        )
    else:
        raise ValueError("Unknown tokenize type.")

    audio.write(f"test_quantization_{quantizer_type}.wav")


if __name__ == "__main__":
    config_path = config["quantizer_config_path"]
    ckpt_path = config["quantizer_ckpt_path"]

    train_dataset, val_dataset = DATASET_2_LOAD_FUNCTION[data](path_to_cache)
    hash_value = "_" + hashlib.md5(data.encode()).hexdigest()

    print(
        "Number of samples in dataset:",
        f"train - {len(train_dataset)}, val - {len(val_dataset)}",
    )

    if quantizer_type == "speech":
        from speechtokenizer import SpeechTokenizer

        quantizer = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
        quantizer.eval().to(device)

        codebook_size = quantizer.quantizer.bins

    elif quantizer_type == "wav":
        quantizer = WavTokenizer.from_pretrained0802(config_path, ckpt_path)
        quantizer = quantizer.to(device)

        codebook_size = quantizer.feature_extractor.encodec.quantizer.bins

    elif quantizer_type == "big-codec":
        quantizer = BigCodecTokenizer(ckpt_path)

    elif quantizer_type == "fish":
        from src.fish_tokenizer import FishAudioTokenizer

        quantizer = FishAudioTokenizer(ckpt_path, config_path)
    else:
        raise ValueError("Unknown tokenize type.")

    if args.debug:
        verify_decoding(train_dataset[0], quantizer, quantizer_type)
    else:
        if quantizer_type == "speech":
            print("Using speech tokenizer.")
            train_dataset = train_dataset.map(
                quantize_speechtokenizer,
                fn_kwargs={"quantizer": quantizer},
                remove_columns=["audio"],
                cache_file_name=os.path.join(
                    path_to_cache, f"tokenize_train_speech_{hash_value}"
                ),
            )
            val_dataset = val_dataset.map(
                quantize_speechtokenizer,
                fn_kwargs={"quantizer": quantizer},
                remove_columns=["audio"],
                cache_file_name=os.path.join(
                    path_to_cache, f"tokenize_val_speech_{hash_value}"
                ),
            )
        elif quantizer_type == "wav":
            print("Using wav tokenizer.")

            def filter_long_audio(examples):
                result_list = []
                for example in examples["audio"]:
                    is_ok = example["array"].shape[-1] < example["sampling_rate"] * 10
                    result_list.append(is_ok)

                return result_list

            # from multiprocess import set_start_method
            # set_start_method("spawn")

            # train_dataset = train_dataset.select(range(10000))
            train_dataset = train_dataset.filter(
                filter_long_audio, batched=True, keep_in_memory=True, num_proc=16
            )

            train_dataset = train_dataset.map(
                quantize_wavtokenizer_batch,
                remove_columns=["audio"],
                keep_in_memory=True,
                fn_kwargs={"quantizer": quantizer},
                batched=True,
                batch_size=50,
                # num_proc=16,
            )

            val_dataset = val_dataset.filter(
                filter_long_audio, batched=True, keep_in_memory=True, num_proc=16
            )

            val_dataset = val_dataset.map(
                quantize_wavtokenizer_batch,
                remove_columns=["audio"],
                keep_in_memory=True,
                fn_kwargs={"quantizer": quantizer},
                batched=True,
                batch_size=50,
                # num_proc=16,
            )
        elif quantizer_type == "big-codec":
            print("Using BigCodec tokenizer.")
            train_dataset = train_dataset.map(
                quantize_bigcodec_tokenizer_batched,
                batched=True,
                batch_size=2,
                fn_kwargs={"quantizer": quantizer},
                cache_file_name=os.path.join(
                    path_to_cache, f"tokenize_train_bigcodec_{hash_value}"
                ),
            )
            val_dataset = val_dataset.map(
                quantize_bigcodec_tokenizer_batched,
                batched=True,
                batch_size=2,
                fn_kwargs={"quantizer": quantizer},
                cache_file_name=os.path.join(
                    path_to_cache, f"tokenize_val_bigcodec_{hash_value}"
                ),
            )

        elif quantizer_type == "fish":
            print("Using fish tokenizer.")
            train_dataset = train_dataset.map(
                quantize_fishtokenizer,
                remove_columns=["audio"],
                fn_kwargs={"quantizer": quantizer},
                cache_file_name=os.path.join(
                    path_to_cache, f"tokenize_train_fish_{hash_value}"
                ),
            )
            val_dataset = val_dataset.map(
                quantize_fishtokenizer,
                fn_kwargs={"quantizer": quantizer},
                remove_columns=["audio"],
                cache_file_name=os.path.join(
                    path_to_cache, f"tokenize_val_fish_{hash_value}"
                ),
            )
        else:
            raise ValueError("Unknown tokenize type.")

        dataset = DatasetDict(
            {
                "train": train_dataset,
                "validation": val_dataset,
            }
        )

        dataset.push_to_hub(
            "Vikhrmodels/" + prepared_data_path, private=True, token=hf_token
        )
