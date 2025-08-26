import os

import hydra
from datasets import DatasetDict
from dotenv import load_dotenv
from omegaconf import DictConfig

from salt.utils.data_lprocessing_utils import DATASET_2_LOAD_FUNCTION
from salt.utils.debug_utils import verify_audio_reconstruction
from salt.utils.loading_utils import load_audio_tokenizer


@hydra.main(config_path="configs", config_name="default.yaml")
def main(config: DictConfig):
    os.environ["HF_HOME"] = config.path_to_cache

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    tokenization_config = config
    tokenization_config.start_audio_token_id = None
    tokenization_config.end_audio_token_id = None
    tokenization_config.n_text_tokens = (
        10_000  # hard coded, no difference when preparing data
    )

    quantizer = load_audio_tokenizer(config.tokenizer)
    train_dataset, val_dataset = DATASET_2_LOAD_FUNCTION[tokenization_config.raw_data]()

    train_dataset = train_dataset.map(quantizer.encode, remove_columns=["audio"])
    val_dataset = val_dataset.map(quantizer.encode, remove_columns=["audio"])

    verify_audio_reconstruction(quantizer, train_dataset[0])

    quantized_dataset = DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
        }
    )

    quantized_dataset.push_to_hub(
        tokenization_config.prepared_data_path, private=True, token=hf_token
    )


if __name__ == "__main__":
    main()
