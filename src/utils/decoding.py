from typing import Optional

import torch


def get_audio_padding_tokens(quantizer, device):
    # create audio without any sounds
    # seems to work better than random padding if
    # length of generated audio is not divisible by n_codebooks
    audio = torch.zeros((1, 1, 1))
    audio = audio.to(device)

    codes = quantizer.encode(audio)

    # Move tensor back to CPU and delete it to free GPU memory
    del audio
    torch.cuda.empty_cache()

    return {"audio_tokens": codes.squeeze(1)}


def get_audio_start_end_tokens(
    tokens: torch.Tensor,
    start_audio_token_id: Optional[int],
    end_audio_token_id: Optional[int],
):
    # find start index of audio tokens
    if start_audio_token_id is not None:
        start = torch.nonzero(tokens == start_audio_token_id)
        start = start[0, -1] + 1 if len(start) else 0
    else:
        start = 0

    # find end index of audio tokens
    if end_audio_token_id is not None:
        end = torch.nonzero(tokens == end_audio_token_id)
        end = end[0, -1] if len(end) else tokens.shape[-1]
    else:
        end = tokens.shape[-1]

    assert start < end, (
        f"Start of audio must be before end. Found: start - {start}, end - {end}"
    )

    return start, end


def decode_audio_bigcodec(
    tokens,
    quantizer,
    n_original_tokens,
    n_codebooks,
    start_audio_token_id: Optional[int] = None,
    end_audio_token_id: Optional[int] = None,
    device="cuda",
):
    # find audio start and end tokens
    start, end = get_audio_start_end_tokens(
        tokens, start_audio_token_id, end_audio_token_id
    )

    # subtract length of original vocabulary -> tokens in range [0, 1024)
    audio_tokens = tokens[start:end] % n_original_tokens
    audio_tokens = audio_tokens.reshape(1, -1, 1).to(device)
    emb = quantizer.decoder.vq2emb(audio_tokens).transpose(1, 2)
    audio = quantizer.decoder(emb, vq=False).squeeze().detach().cpu()

    del tokens
    del audio_tokens
    torch.cuda.empty_cache()

    return audio, 16000
