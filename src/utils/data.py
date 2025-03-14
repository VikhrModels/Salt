import os
import subprocess
from pathlib import Path

from datasets import Audio, Dataset, load_dataset, Value
from huggingface_hub import hf_hub_download


def prepare_librispeech(cache_dir) -> tuple[Dataset, Dataset]:
    raw = load_dataset("openslr/librispeech_asr", "clean", cache_dir=cache_dir)
    processed = raw.remove_columns(["chapter_id"])
    processed = processed.cast_column("speaker_id", Value("string"))
    return processed["train.100"], processed["validation"]


def prepare_tedlium(cache_dir) -> tuple[Dataset, Dataset]:
    raw = load_dataset("LIUM/tedlium", "release1", cache_dir=cache_dir)
    processed = raw.remove_columns(["gender"])
    return processed["train"], processed["validation"]


def prepare_parler_tts(cache_dir) -> tuple[Dataset, Dataset]:
    raw_mls = load_dataset("parler-tts/mls_eng", cache_dir=cache_dir)
    processed_mls = raw_mls.remove_columns(
        ["begin_time", "end_time", "speaker_id", "book_id", "audio_duration"]
    )
    processed_mls = processed_mls.rename_column("transcript", "text")

    return processed_mls["train"], processed_mls["dev"]


def prepare_synthetic(cache_dir) -> tuple[Dataset, Dataset]:
    raw = load_dataset("homebrewltd/instruction-speech-encodec-v1", cache_dir=cache_dir)
    processed = raw.remove_columns(["prompt", "length"])
    processed = processed.rename_column("answer", "text")
    splits = processed["train"].train_test_split(test_size=0.1)

    return splits["train"], splits["test"]


def prepare_parler_tts_with_description(cache_dir) -> tuple[Dataset, Dataset]:
    audio = load_dataset("parler-tts/libritts_r_filtered", "clean", cache_dir=cache_dir)
    train_audio, val_audio = audio["train.clean.100"], audio["dev.clean"]

    columns = ["id", "text", "path", "text_description"]
    raw = load_dataset(
        "parler-tts/libritts-r-filtered-speaker-descriptions",
        "clean",
        cache_dir=cache_dir,
    )
    processed = raw.remove_columns(
        list(set(raw.column_names["dev.clean"]) - set(columns))
    )
    train_text, val_text = processed["train.clean.100"], processed["dev.clean"]

    assert train_audio["id"] == train_text["id"] and val_audio["id"] == val_text["id"]

    audio_features_train = train_audio["audio"]
    audio_features_val = val_audio["audio"]

    os.makedirs("cache/", exist_ok=True)

    train_text = train_text.map(
        lambda x, i: {"audio": audio_features_train[i]},
        with_indices=True,
        cache_file_name="cache/merge_train",
    )
    val_text = val_text.map(
        lambda x, i: {"audio": audio_features_val[i]},
        with_indices=True,
        cache_file_name="cache/merge_val",
    )
    return train_text, val_text


def prepare_homebrewltd(cache_dir) -> tuple[Dataset, Dataset]:
    dataset = load_dataset(
        "homebrewltd/instruction-speech-encodec-v1", "default", cache_dir=cache_dir
    )["train"]

    dataset = dataset.rename_column("answer", "text")
    splits = dataset.train_test_split(test_size=0.1)

    return splits["train"], splits["test"]


def prepare_urban_flan(cache_dir) -> tuple[Dataset, Dataset]:
    repo_id = "Vikhrmodels/urban_flan_dataset"
    dataset = load_dataset(
        repo_id, cache_dir=cache_dir
    )

    shuffled = dataset['train'].shuffle(seed=42)

    def fix_audio_format(example):
        return {
            "audio": {
                "array": example['audio_array'],
                "sampling_rate": example["sampling_rate"],
            }
        }
    shuffled = shuffled.map(fix_audio_format, keep_in_memory=True, remove_columns=["audio_array", "sampling_rate"])

    splits = shuffled.train_test_split(test_size=2048, seed=42)

    return splits["train"], splits["test"]


def _prepare_emilia(file_list, cache_dir, num_samples=None) -> tuple[Dataset, Dataset]:
    repo_id = "amphion/Emilia-Dataset"

    dataset = load_dataset(
        repo_id, data_files=file_list, cache_dir=cache_dir, num_proc=16
    )
    subset = dataset.shuffle(seed=42)

    if num_samples is not None and num_samples < len(subset['train']):
        subset = subset['train'].select(range(num_samples))

    def extract_text(batch):
        return {"text": [item["text"] for item in batch["json"]]}

    subset = subset.map(extract_text, batched=True)

    subset = subset.rename_columns({"__key__": "index", "mp3": "audio"})
    splits = subset.train_test_split(test_size=2048, seed=42)
    return splits["train"], splits["test"]

def prepare_emilia(cache_dir) -> tuple[Dataset, Dataset]:
    file_list = [f"EN/EN-B{str(i).zfill(6)}.tar" for i in range(200)]

    return _prepare_emilia(file_list, cache_dir, num_samples=1_000_000)

def prepare_emilia_multilang(cache_dir) -> tuple[Dataset, Dataset]:
    # max
    # lang_slugs = {
    #     'DE': 90,
    #     'EN': 200,
    #     'FR': 100,
    #     'JA': 70,
    #     'KO': 40,
    #     'ZH': 200,
    # }
    # Test
    lang_slugs = {
        'DE': 10,
        'EN': 10,
        'FR': 10,
        'JA': 10,
        'KO': 10,
        'ZH': 10,
    }
    file_list = []
    for lang_slug, num_partitions in lang_slugs.items():
        file_list.extend([f"Emilia/{lang_slug}/{lang_slug}-B{str(i).zfill(6)}.tar" for i in range(num_partitions)])

    return _prepare_emilia(file_list, cache_dir, num_samples=1_000_000 * len(lang_slugs))


def prepare_emilia_full(cache_dir) -> tuple[Dataset, Dataset]:

    file_list = None
    return _prepare_emilia(file_list, cache_dir)


def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir="/tmp/musiccaps",
    num_attempts=5,
    url_base="https://www.youtube.com/watch?v=",
):
    status = False

    command = f"""
        yt-dlp --quiet --force-keyframes-at-cuts --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" "{url_base}{video_identifier}"
    """.strip()

    attempts = 0
    while True:
        try:
            output = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, "Downloaded"


def prepare_musiccaps(cache_dir: str) -> tuple[Dataset, Dataset]:
    ds = load_dataset("google/MusicCaps", split="train")
    sampling_rate = 44100
    limit = None
    num_proc, writer_batch_size = 16, 1000

    if limit is not None:
        print(f"Limiting to {limit} examples")
        ds = ds.select(range(limit))

    data_dir = "../music_data"
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['ytid']}.wav")
        status = True
        if not os.path.exists(outfile_path):
            status = False
            status, log = download_clip(
                example["ytid"],
                outfile_path,
                example["start_s"],
                example["end_s"],
            )

        example["audio"] = outfile_path
        example["download_status"] = status
        return example

    ds = ds.rename_column("caption", "text")
    ds = ds.map(
        process,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        keep_in_memory=False,
    )
    ds = ds.filter(lambda x: x["download_status"]).cast_column(
        "audio", Audio(sampling_rate=sampling_rate)
    )

    splits = ds.train_test_split(test_size=0.1, seed=42)
    return splits["train"], splits["test"]


DATASET_2_LOAD_FUNCTION = {
    "emilia": prepare_emilia,
    "emilia_multilang": prepare_emilia_multilang,
    "emilia_full": prepare_emilia_full,
    "urban_flan": prepare_urban_flan,
    "homebrewltd": prepare_homebrewltd,
    "librispeech": prepare_librispeech,
    "musiccaps": prepare_musiccaps,
    "parler-tts": prepare_parler_tts,
    "parler_tts_with_description": prepare_parler_tts_with_description,
    "synthetic": prepare_synthetic,
    "tedlium": prepare_tedlium,
}
