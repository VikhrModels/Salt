import re


def create_audio_tokens(original_tokens_list):
    ready_tokens = []

    for token in original_tokens_list:
        ready_tokens.append(f"<|bigcodec_{token}|>")

    return ready_tokens


def extract_audio_tokens(ready_tokens):
    pattern = r"<\|bigcodec_(.+?)\|>"
    return [
        re.match(pattern, token).group(1)
        for token in ready_tokens
        if re.match(pattern, token)
    ]
