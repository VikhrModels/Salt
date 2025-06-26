import argparse
import hashlib

import os

from typing import Any
import soundfile as sf

import sys

sys.path.append("BigCodec")

import numpy as np
import yaml

from datasets import DatasetDict
import torch


from BigCodec.vq.codec_decoder import CodecDecoder
from BigCodec.vq.codec_encoder import CodecEncoder

from src.utils.data import DATASET_2_LOAD_FUNCTION
from src.utils.decoding import decode_audio_bigcodec

import math
from scipy.signal import resample_poly

# load_dotenv()
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


def filter_long_audio(examples):
    result_list = []
    for example in examples["audio"]:
        is_ok = example["array"].shape[-1] < example["sampling_rate"] * 15
        result_list.append(is_ok)

    return result_list


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


def resample(audio: np.ndarray, sr: int, target_sr: int) -> torch.Tensor:
    """
    Меняет частоту дискретизации numpy-массива audio (диапазон –1…+1)
    с sr → target_sr и возвращает torch.Tensor формы [1, samples] (моно)
    или [1, channels, samples] (многоканал).
    """
    # 1) В float32 для точности
    audio_f = audio.astype(np.float32)

    # 2) Если частота не меняется — сразу в тензор
    if sr == target_sr:
        t = torch.from_numpy(audio_f)
        # если моно: [1, N], если stereo: [1, C, N]
        if t.ndim == 1:
            return t.unsqueeze(0)
        else:
            return t.T.unsqueeze(0)

    # 3) Готовим коэфы up/down через НОД
    g = math.gcd(sr, target_sr)
    up, down = target_sr // g, sr // g

    # 4) Ресэмплинг с антиалиасингом
    if audio_f.ndim == 1:
        out = resample_poly(audio_f, up, down)
        tensor = torch.from_numpy(out).unsqueeze(0)  # [1, N']
    else:
        # data shape (N, C) → stack → (N', C)
        chans = [
            resample_poly(audio_f[:, ch], up, down) for ch in range(audio_f.shape[1])
        ]
        out = np.stack(chans, axis=1)  # (N', C)
        tensor = torch.from_numpy(out.T).unsqueeze(0)  # [1, C, N']

    return tensor


def quantize_bigcodec_tokenizer(row: dict[str, Any], quantizer: BigCodecTokenizer):
    audio_data, sample_rate = row["audio"]["array"], row["audio"]["sampling_rate"]

    audio = resample(audio_data, sample_rate, 16000)
    audio = audio.to(device)
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
    audio = audio.to(device)

    codes = quantizer.encode(audio)
    codes = codes.squeeze(0)
    codes = codes.cpu()

    return {"audio_tokens": codes.numpy()}


def verify_decoding(example, quantizer, quantizer_type: str):
    if quantizer_type == "big-codec":
        codes = quantize_bigcodec_tokenizer(example, quantizer)["audio_tokens"]
        codes = torch.tensor(codes, dtype=torch.long, device=device)
        codes = codes.view(1, -1, 1)

        audio = decode_audio_bigcodec(
            codes,
            quantizer,
            quantizer.decoder.quantizer.layers[0].codebook_size,
            quantizer.decoder.quantizer.num_quantizers,
        )

    else:
        raise ValueError("Unknown tokenize type.")

    waveform, sample_rate = audio
    waveform = waveform.squeeze(0)
    wave_np = waveform.detach().cpu().numpy()

    if wave_np.ndim == 2 and wave_np.shape[0] < wave_np.shape[1]:
        wave_np = wave_np.T

    out_path = f"test_quantization_{quantizer_type}.wav"
    sf.write(out_path, wave_np, sample_rate)

    print(f"✅ Аудио сохранено в «{out_path}» при {sample_rate} Гц")


if __name__ == "__main__":
    config_path = config["quantizer_config_path"]
    ckpt_path = config["quantizer_ckpt_path"]

    train_dataset, val_dataset = DATASET_2_LOAD_FUNCTION[data](path_to_cache)
    hash_value = "_" + hashlib.md5(data.encode()).hexdigest()

    print(
        "Number of samples in dataset:",
        f"train - {len(train_dataset)}, val - {len(val_dataset)}",
    )

    if quantizer_type == "big-codec":
        quantizer = BigCodecTokenizer(ckpt_path)

    else:
        raise ValueError("Unknown tokenize type.")

    if args.debug:
        verify_decoding(train_dataset[0], quantizer, quantizer_type)
    else:
        if quantizer_type == "big-codec":
            print("Using BigCodec tokenizer.")

            train_dataset = train_dataset.filter(
                filter_long_audio, batched=True, keep_in_memory=True, num_proc=4
            )

            val_dataset = val_dataset.filter(
                filter_long_audio, batched=True, keep_in_memory=True, num_proc=4
            )

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
        else:
            raise ValueError("Unknown tokenize type.")

        dataset = DatasetDict(
            {
                "train": train_dataset,
                "validation": val_dataset,
            }
        )

        dataset.push_to_hub(
            "Vikhrmodels/" + prepared_data_path,
            private=False,
            token=hf_token,
            max_shard_size="2GB",
        )
