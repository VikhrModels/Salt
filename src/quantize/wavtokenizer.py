import torch
import numpy as np
from typing import Any, List

from WavTokenizer.encoder.utils import convert_audio
from WavTokenizer.decoder.pretrained import WavTokenizer

def resample(audio: np.ndarray, sr: int, target_sr: int):
    audio = torch.tensor(audio, dtype=torch.float32)
    audio = audio.unsqueeze(0)

    if sr != target_sr:
        # 1 as last arg corresponds to mono audio
        audio = convert_audio(audio, sr, target_sr, 1)

    return audio


def quantize_wavtokenizer_batch(rows: List[dict[str, Any]], quantizer: WavTokenizer):

    expected_tokens_count = []
    all_audio_data = []
    max_audio_length = 0
    for audio_item in rows["audio"]:
        audio_data, sample_rate = audio_item["array"], int(audio_item["sampling_rate"])
        audio = resample(audio_data, sample_rate, 24000)
        max_audio_length = max(max_audio_length, audio_data.shape[-1])

        expected_tokens_count.append(int(audio_data.shape[-1] / 24000 * 40))

        all_audio_data.append(audio)

    batched_audio_data = torch.zeros([len(all_audio_data), max_audio_length])
    for i, audio in enumerate(all_audio_data):
        batched_audio_data[i, :audio.shape[-1]] = audio

    with torch.no_grad():
        bandwidth_id = torch.tensor([0])
        _, codes = quantizer.encode_infer(batched_audio_data.to(quantizer.head.out.weight.device), bandwidth_id=bandwidth_id)

    codes = codes.squeeze(0)
    codes = codes.cpu()

    audio_tokens = []
    for i in range(codes.shape[0]):
        audio_tokens.append(codes[i, :expected_tokens_count[i]])

    return {"audio_tokens": audio_tokens}

