import os
import subprocess
from pathlib import Path
from utils import remove_accent

from datasets import Audio, Dataset, load_dataset, concatenate_datasets, Value
from typing import Callable, Dict, Tuple

PrepareFn = Callable[[str, dict], Tuple[Dataset, Dataset]]
_DATASETS: Dict[str, PrepareFn] = {}


def register(name: str):
    """
    Регистрируем датасет в нашем реестре
    """

    def decorator(fn: PrepareFn) -> PrepareFn:
        if name in _DATASETS:
            raise ValueError(f"Датасет '{name}' уже зарегистрирован")
        _DATASETS[name] = fn
        return fn

    return decorator


def load_dataset_by_name(
    name: str, cache_dir: str = ".cache", **kwargs
) -> Tuple[Dataset, Dataset]:
    """Вызывает нужную prepare-функцию из реестра"""
    try:
        fn = _DATASETS[name]
    except KeyError:
        raise KeyError(
            f"Неизвестный датасет '{name}'. Доступные варианты: {list(_DATASETS)}"
        )
    return fn(cache_dir, kwargs)


@register("librispeech")
def _prep_librispeech(cache_dir: str, kwargs: dict) -> Tuple[Dataset, Dataset]:
    raw = load_dataset("openslr/librispeech_asr", "clean", cache_dir=cache_dir)
    processed = raw.remove_columns(["chapter_id"]).cast_column(
        "speaker_id", Value("string")
    )
    return processed["train.100"], processed["validation"]


@register("tedlium")
def _prep_tedlium(cache_dir: str, opts: dict) -> Tuple[Dataset, Dataset]:
    raw = load_dataset("LIUM/tedlium", "release1", cache_dir=cache_dir)
    processed = raw.remove_columns(["gender"])
    return processed["train"], processed["test"]


@register("tonespeak")
def _prep_tonespeak(cache_dir: str, opts: dict) -> Tuple[Dataset, Dataset]:
    raw = load_dataset("Vikhrmodels/ToneSpeak", cache_dir=cache_dir)
    processed = raw.rename_column("text_description", "prompt").map(remove_accent)
    return processed["train"], processed["validation"]


@register("tonebooks")
def _prep_tonebooks(cache_dir: str, opts: dict) -> Tuple[Dataset, Dataset]:
    raw = load_dataset("Vikhrmodels/ToneBooks", cache_dir=cache_dir)
    processed = (
        raw.rename_column("text_description", "prompt")
        .rename_column("mp3", "audio")
        .map(remove_accent)
    )
    splits = processed["train"].train_test_split(test_size=0.1)

    return splits["train"], splits["test"]


@register("emilia")
def _prep_emilia(cache_dir: str, opts: dict) -> Tuple[Dataset, Dataset]:
    repo_id = "amphion/Emilia-Dataset"
    num_samples = opts.get("num_samples", 1_000_000)
    
    file_list = [f"EN/EN-B{str(i).zfill(6)}.tar" for i in range(200)]
    
    dataset = load_dataset(
        repo_id, data_files=file_list, cache_dir=cache_dir, num_proc=16
    )
    subset = dataset.shuffle(seed=42)

    if num_samples is not None and num_samples < len(subset["train"]):
        subset = subset["train"].select(range(num_samples))

    def extract_text(batch):
        return {"text": [item["text"] for item in batch["json"]]}

    subset = subset.map(extract_text, batched=True)
    subset = subset.rename_columns({"__key__": "index", "mp3": "audio"})
    splits = subset.train_test_split(test_size=2048, seed=42)
    
    return splits["train"], splits["test"]


@register("emilia_multilang")
def _prep_emilia_multilang(cache_dir: str, opts: dict) -> Tuple[Dataset, Dataset]:
    repo_id = "amphion/Emilia-Dataset"
    num_samples = opts.get("num_samples", 1_000_000)
    use_test_config = opts.get("test_config", True)
    
    if use_test_config:
        lang_slugs = {
            "DE": 10, "EN": 10, "FR": 10, 
            "JA": 10, "KO": 10, "ZH": 10,
        }
    else:
        lang_slugs = {
            "DE": 90, "EN": 200, "FR": 100,
            "JA": 70, "KO": 40, "ZH": 200,
        }
    
    file_list = []
    for lang_slug, num_partitions in lang_slugs.items():
        file_list.extend([
            f"Emilia/{lang_slug}/{lang_slug}-B{str(i).zfill(6)}.tar"
            for i in range(num_partitions)
        ])
    
    dataset = load_dataset(
        repo_id, data_files=file_list, cache_dir=cache_dir, num_proc=16
    )
    subset = dataset.shuffle(seed=42)

    if num_samples is not None and num_samples < len(subset["train"]):
        subset = subset["train"].select(range(num_samples * len(lang_slugs)))

    def extract_text(batch):
        return {"text": [item["text"] for item in batch["json"]]}

    subset = subset.map(extract_text, batched=True)
    subset = subset.rename_columns({"__key__": "index", "mp3": "audio"})
    splits = subset.train_test_split(test_size=2048, seed=42)
    
    return splits["train"], splits["test"]


