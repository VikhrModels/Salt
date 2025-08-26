from typing import Optional

import torch


def get_audio_start_end_tokens(
    tokens: torch.Tensor,
    start_audio_token_id: Optional[int],
    end_audio_token_id: Optional[int],
):
    # find start index of audio tokensre
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
        f"Start of audio token must be before end of audio token. Found: start - {start}, end - {end}"
    )

    return start, end
