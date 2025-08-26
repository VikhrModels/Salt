from enum import Enum


class DataType(Enum):
    base = "base"
    audio = "audio"
    text = "text"
    tts = "tts"
    asr = "asr"
    instruct_tts = "instruct_tts"
    instruct_asr = "instruct_asr"
