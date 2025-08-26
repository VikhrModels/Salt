from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig
from torch.utils.data import Dataset, ConcatDataset
from transformers import PreTrainedTokenizer

from salt.data import DataType
from salt.data.base_datasets import SaltAudioDataset, SaltTextDataset
from salt.data.instruct_datasets import SaltTextToSpeechDatasetWithVoiceDescription, \
    SaltSpeechRecognitionDatasetWithVoiceDescription
from salt.data.task_datasets import SaltTextToSpeechDataset, SaltSpeechRecognitionDataset


def filter_long_audio(dataset: DatasetDict, max_audio_tokens: int) -> DatasetDict:
    train = dataset["train"].filter(lambda item: len(item["audio_tokens"][0]) > max_audio_tokens)
    val = dataset["test"].filter(lambda item: len(item["audio_tokens"][0]) > max_audio_tokens)
    return DatasetDict({"train": train, "test": val})


def get_dataset_from_type(dataset_type: DataType, dataset_path: str, config: DictConfig, tokenizer: PreTrainedTokenizer) -> tuple[Dataset, Dataset]:
    dataset = load_dataset(dataset_path)

    if dataset_type != DataType.text:
        dataset = filter_long_audio(dataset, config.training.max_audio_tokens)

    if dataset_type == DataType.audio:
        train = SaltAudioDataset(config, dataset["train"], tokenizer)
        val = SaltAudioDataset(config, dataset["test"], tokenizer)
    elif dataset_type == DataType.text:
        train = SaltTextDataset(config, dataset["train"], tokenizer)
        val = SaltTextDataset(config, dataset["test"], tokenizer)
    elif dataset_type == DataType.tts:
        train = SaltTextToSpeechDataset(config, dataset["train"], tokenizer)
        val = SaltTextToSpeechDataset(config, dataset["test"], tokenizer)
    elif dataset_type == DataType.asr:
        train = SaltSpeechRecognitionDataset(config, dataset["train"], tokenizer)
        val = SaltSpeechRecognitionDataset(config, dataset["test"], tokenizer)
    elif dataset_type == DataType.instruct_tts:
        train = SaltTextToSpeechDatasetWithVoiceDescription(config, dataset["train"], tokenizer)
        val = SaltTextToSpeechDatasetWithVoiceDescription(config, dataset["test"], tokenizer)
    else:
        train = SaltSpeechRecognitionDatasetWithVoiceDescription(config, dataset["train"], tokenizer)
        val = SaltSpeechRecognitionDatasetWithVoiceDescription(config, dataset["test"], tokenizer)

    return train, val


def prepare_data(config: DictConfig, tokenizer: PreTrainedTokenizer):
    train_datasets: list[Dataset] = []
    val_datasets: list[Dataset] = []

    for data_config in config.data.datasets:
        data_type = DataType[data_config["type"]]
        train, val = get_dataset_from_type(data_type, data_config["path"], config, tokenizer)
        train_datasets.append(train)
        val_datasets.append(val)

    train_concat = ConcatDataset(train_datasets)
    val_concat = ConcatDataset(val_datasets)
    return train_concat, val_concat
