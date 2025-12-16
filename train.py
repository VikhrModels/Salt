from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer
import os
import yaml
import argparse
import torch
from src.dataset_builder import load_datasets, filter_by_length


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_ENTITY"] = config["wandb"]["entity"]
os.environ["WANDB_PROJECT"] = config["wandb"]["project"]

torch.backends.cudnn.benchmark = True


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["model"]["language_model"],
    max_seq_length=config["model"]["max_initial_seq_length"],
    dtype=torch.bfloat16,
    full_finetuning=True,
)

audio_special_tokens = [
    f"<|bigcodec_{i}|>" for i in range(config["audio_codec"]["codebook_size"])
]
tokenizer.add_tokens(audio_special_tokens)
model.resize_token_embeddings(len(tokenizer))


combined_train = load_datasets(config["datasets"], split_name="train")

train_dataset = filter_by_length(
    dataset=combined_train,
    tokenizer=tokenizer,
    max_text_length=config["model"]["max_text_length"],
    max_audio_tokens=config["model"]["max_audio_tokens"],
    num_proc=config["datasets"].get("num_proc", 1),
)

print(f"Training dataset ready: {len(train_dataset):,} examples\n")
