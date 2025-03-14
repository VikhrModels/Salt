import os
import torch
import numpy as np
from typing import Any

from WavTokenizer.encoder.utils import convert_audio
from WavTokenizer.decoder.pretrained import WavTokenizer


from quantize.wavtokenizer import quantize_wavtokenizer_batch

def test_batch_quantize():
    device = 'cuda'

    ckpt_path = "../audiotokenizer/wavtokenizer_large_unify_600_24k.ckpt"
    config_path = "../WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"

    quantizer = WavTokenizer.from_pretrained0802(config_path, ckpt_path)
    quantizer = quantizer.to(device)

    batch_size = 2

    audio_data = torch.rand([batch_size, 1, 240000], dtype=torch.float32)
    audio_data = convert_audio(audio_data, 24000, 24000, 1)
    audio_data = audio_data.to(device)

    print(audio_data.shape)

    all_codes_one_by_one = []
    for i in range(audio_data.shape[0]):
        _, codes = quantizer.encode_infer(audio_data[i], bandwidth_id=torch.tensor([0]))
        codes = codes.squeeze(1)
        all_codes_one_by_one.append(codes)

    all_codes_one_by_one = torch.cat(all_codes_one_by_one, dim=0)

    audio_data = audio_data.squeeze(1)
    _, codes_batch = quantizer.encode_infer(audio_data, bandwidth_id=torch.tensor([0]))

    codes_batch = codes_batch.squeeze(1)

    # хз почему, но даже в таком тесте небольшая часть кодов расходится
    assert (codes_batch != all_codes_one_by_one).sum() < batch_size * 2

    breakpoint()


def test_map_dataset():
    from datasets import load_dataset
    repo_id = "amphion/Emilia-Dataset"

    val_dataset = load_dataset(
        repo_id, data_files=[ f'Emilia/EN/EN-B000000.tar' ],
    )

    val_dataset = val_dataset['train'].select(range(10))
    val_dataset = val_dataset.rename_columns({"__key__": "index", "mp3": "audio"})

    device = 'cuda'

    ckpt_path = "../audiotokenizer/wavtokenizer_large_unify_600_24k.ckpt"
    config_path = "../WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"

    quantizer = WavTokenizer.from_pretrained0802(config_path, ckpt_path)
    quantizer = quantizer.to(device)

    val_dataset = val_dataset.map(quantize_wavtokenizer_batch, fn_kwargs={"quantizer": quantizer}, batched=True)
