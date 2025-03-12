from datasets import load_dataset
import torch
from torch.utils.data import Dataset, ConcatDataset


class Vikhr4oDatasetBase(Dataset):
    def __init__(self, dataset, tokenizer, quantizer, asr: bool, config):

        assert dataset is not None

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.quantizer = quantizer
        self.asr = asr

        if self.asr:
            self.n_codebooks = quantizer.asr_n_codebooks
        else:
            self.n_codebooks = quantizer.tts_n_codebooks

        self.n_special_tokens = config["n_special_tokens"]

        self.soa = tokenizer(config["start_audio_token"], return_tensors="pt")[
            "input_ids"
        ][:, -1:]

        self.eoa = tokenizer(config["end_audio_token"], return_tensors="pt")[
            "input_ids"
        ][:, -1:]

        self.eos = tokenizer(config["end_sequence_token"], return_tensors="pt")[
            "input_ids"
        ][:, -1:]

        self.bos = tokenizer(config["start_sequence_token"], return_tensors="pt")[
            "input_ids"
        ][:, -1:]

    def __len__(self):
        return len(self.dataset)

    def get_text_tokens(self, row):
        text_tokenized = self.tokenizer(row["text"].lower(), return_tensors="pt")
        return text_tokenized["input_ids"]

    def __getitem__(self, idx):
        row = self.dataset[idx]
        text_input_tokens = self.get_text_tokens(row)

        if self.asr:
            audio_input_tokens = self.quantizer.quantize_asr(row)
        else:
            audio_input_tokens = self.quantizer.quantize_tts(row)

        if self.asr:
            tokens = torch.cat(
                [
                    self.bos,
                    self.soa,
                    audio_input_tokens,
                    self.eoa,
                    text_input_tokens,
                    self.eos,
                ],
                dim=1,
            ).squeeze(0)
        else:
            tokens = torch.cat(
                [
                    self.bos,
                    text_input_tokens,
                    self.soa,
                    audio_input_tokens,
                    self.eoa,
                    self.eos,
                ],
                dim=1,
            ).squeeze(0)

        attention_mask = torch.ones(len(tokens))

        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "labels": tokens.clone(),
            "is_asr": torch.ones([1]) * self.asr,
        }


class Vikhr4oDatasetVoiceDescription(Vikhr4oDatasetBase):
    def get_text_tokens(self, row):
        if self.asr:
            text = "'{text}' is said with {voice_dsc}".format(
                text=row["text"], voice_dsc=row["text_description"]
            )
        else:
            text = "Say '{text}' with {voice_dsc}".format(
                text=row["text"], voice_dsc=row["text_description"]
            )
        text_tokenized = self.tokenizer(text, return_tensors="pt")
        return text_tokenized["input_ids"]


def prepare_text_field(row):
    return {"text": row["json"]["text"]}


def load_tokenized_data(data_path: str, few_val_samples=None):
    speech_path = data_path + "-speech"

    wav_path = data_path
    if '-wav-' not in data_path:
        wav_path = data_path + "-wav-unify"

    train, val = None, None

    try:
        speech = load_dataset(speech_path)
        train_speech, val_speech = speech["train"], speech["validation"]

        if "text" not in train_speech.column_names:
            train_speech = train_speech.map(prepare_text_field)
            val_speech = val_speech.map(prepare_text_field)

        train = train_speech.rename_column("audio_tokens", "audio_tokens_speech")
        val = val_speech.rename_column("audio_tokens", "audio_tokens_speech")

    except Exception as e:
        print(f"No speech data found for {data_path}.: {e}")

    try:
        wav = load_dataset(wav_path)
        train_wav, val_wav = wav["train"], wav["validation"]

        if (
            "text" not in train_wav.column_names
            and "text_description" not in train_wav.column_names
        ):
            train_wav = train_wav.map(prepare_text_field)
            val_wav = val_wav.map(prepare_text_field)

        if train is None:
            train = train_wav.rename_column("audio_tokens", "audio_tokens_wav")
            val = val_wav.rename_column("audio_tokens", "audio_tokens_wav")
        else:
            train = train.add_column("audio_tokens_wav", train_wav["audio_tokens"])
            val = val.add_column("audio_tokens_wav", val_wav["audio_tokens"])

    except Exception as e:
        print(f"No wav data found for {data_path}.: {e}")
        train_wav, val_wav = None, None

    if train is None and val is None:
        raise ValueError(f"No data found for {data_path}.")

    if val is not None and few_val_samples is not None and few_val_samples > 0:
        val = val.select(range(few_val_samples))

    return train, val


def load_train_val_splits(dataset: str, tokenizer, quantizer, config, few_val_samples=None):
    train_ds, val_ds = load_tokenized_data(dataset, few_val_samples=few_val_samples)
    train, val = [], []

    if "librispeech" in dataset or "emilia" in dataset or "music" in dataset or "mozilla":
        if "asr" in config["tasks"] and "ru" not in dataset:
            train.append(
                Vikhr4oDatasetBase(train_ds, tokenizer, quantizer, True, config)
            )
            val.append(Vikhr4oDatasetBase(val_ds, tokenizer, quantizer, True, config))

        if "tts" in config["tasks"]:
            train.append(
                Vikhr4oDatasetBase(train_ds, tokenizer, quantizer, False, config)
            )
            val.append(Vikhr4oDatasetBase(val_ds, tokenizer, quantizer, False, config))

    elif "with_description" in dataset:
        if "asr" in config["tasks"]:
            train.append(
                Vikhr4oDatasetVoiceDescription(
                    train_ds, tokenizer, quantizer, True, config
                )
            )
            val.append(
                Vikhr4oDatasetVoiceDescription(
                    val_ds, tokenizer, quantizer, True, config
                )
            )

        if "tts" in config["tasks"]:
            train.append(
                Vikhr4oDatasetVoiceDescription(
                    train_ds, tokenizer, quantizer, False, config
                )
            )
            val.append(
                Vikhr4oDatasetVoiceDescription(
                    val_ds, tokenizer, quantizer, False, config
                )
            )

    elif "homebrewltd" in dataset:
        if "asr" in config["tasks"]:
            train.append(
                Vikhr4oDatasetBase(train_ds, tokenizer, quantizer, True, config)
            )
            val.append(Vikhr4oDatasetBase(val_ds, tokenizer, quantizer, True, config))

    else:
        raise ValueError("Unknown dataset.")

    return train, val


def load_text_dataset(dataset_path: str, tokenizer, max_length: int):
    dataset = load_dataset(dataset_path)
    train, val = dataset["train"], dataset["validation"]
    template = "{instruction}\n{response}"

    train = train.map(
        lambda example: {
            "prompt": template.format(
                instruction=example["instruction"] + example["input"],
                response=example["output"],
            )
        }
    )

    val = val.map(
        lambda example: {
            "prompt": template.format(
                instruction=example["instruction"] + example["input"],
                response=example["output"],
            )
        }
    )

    train_tokenized = train.map(
        lambda examples: tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
    )

    val_tokenized = val.map(
        lambda examples: tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
    )

    train_tokenized = train_tokenized.map(lambda x: {"labels": x["input_ids"]})
    val_tokenized = val_tokenized.map(lambda x: {"labels": x["input_ids"]})

    train_tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    val_tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return train_tokenized, val_tokenized


def load_data(
    audio_datasets: list[str], tokenizer, quantizer, config, few_val_samples=None
) -> tuple[Dataset, Dataset]:
    train_datasets: list[Dataset] = []
    val_datasets: list[Dataset] = []

    for dataset in audio_datasets:
        train, val = load_train_val_splits(dataset, tokenizer, quantizer, config, few_val_samples=few_val_samples)

        for split in [train, val]:
            for dataset in split:
                print("Filter out long sequences")
                print("Before filtering:", len(dataset))
                dataset.dataset = dataset.dataset.filter( lambda x: len(x['audio_tokens_wav'][0]) < 512 )
                print("After filtering:", len(dataset))

        train_datasets.extend(train)
        val_datasets.extend(val)

    if len(config["text_data"]):
        for text_ds in config["text_data"]:
            train, val = load_text_dataset(text_ds, tokenizer, config["max_seq_length"])
            train_datasets.append(train)
            val_datasets.append(val)


    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)

