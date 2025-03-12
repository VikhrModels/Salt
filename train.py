import argparse
import math
import os
import yaml

from dataclasses import dataclass, field
from typing import List, Optional, Union, Any, Union, Dict, Tuple


from tqdm import tqdm
from dotenv import load_dotenv
import wandb

import torch
from torch.utils.data import DataLoader, RandomSampler

from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
)
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)

from src.data import load_data
from src.tokenizer import AudioTokenizer, get_start_tokens
from src.utils.training import save_checkpoint, get_exp_name, collate_fn

@dataclass
class SaltTrainingArguments(TrainingArguments):

    # Часть параметров переопределяем из конфига
    config: str = field(default="")

    output_dir: str = field(default="./results")

    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=1000)

    few_val_samples: int = field(default=128)

    save_strategy: str = field(default="epoch")
    save_total_limit: Optional[int] = field(default=1)

    optim: str = field(default="adamw_torch")
    report_to: str = field(default="wandb")
    logging_steps: int = field(default=50)
    dataloader_drop_last: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)

    torch_compile: bool = field(default=True)

def _build_model(training_args, config, new_embeddings_count):
    if checkpoint_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            cache_dir=path_to_cache,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            cache_dir=path_to_cache,
        )

    model.config.use_cache = False
    model.resize_token_embeddings(new_embeddings_count)

    return model


if __name__ == "__main__":

    hf_parser = HfArgumentParser(SaltTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()

    # Load config
    with open(training_args.config, "r") as file:
        config = yaml.safe_load(file)

    base_model = config["base_model"]
    checkpoint_path = config.get("checkpoint_path")
    save_dir = config["save_dir"]

    data = config["data"]

    start_audio_token = config["start_audio_token"]
    end_audio_token = config["end_audio_token"]

    path_to_cache = config["path_to_cache"]

    torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = config["allow_tf32"]

    load_dotenv()
    wandb.login(key=os.getenv("WB_KEY"))

    training_args.per_device_train_batch_size = config["train_batch_size"]
    training_args.per_device_eval_batch_size = config["eval_batch_size"]
    training_args.num_train_epochs = config["num_train_epochs"]

    training_args.weight_decay = float(config["weight_decay"])
    training_args.learning_rate = float(config["learning_rate"])
    training_args.max_grad_norm = float(config["max_grad_norm"])
    training_args.lr_scheduler_type = config["lr_scheduler_type"]
    training_args.num_warmup_steps = int(config["num_warmup_steps"])
    training_args.gradient_accumulation_steps = int(config["gradient_accumulation_steps"])

    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=path_to_cache)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            {"pad_token": "[PAD]"}
        )  # '[PAD]' is the new padding token
        tokenizer.pad_token = "[PAD]"
        config["n_special_tokens"] += 1

    tokenizer.add_special_tokens(
        {"additional_special_tokens": [start_audio_token, end_audio_token]}
    )
    n_tokens = len(tokenizer)
    print("Not audio tokens:", n_tokens)

    start_audio_token_id = tokenizer._convert_token_to_id_with_added_voc(start_audio_token)
    end_audio_token_id = tokenizer._convert_token_to_id_with_added_voc(end_audio_token)

    tokens_config = get_start_tokens(config["quantizer"], n_tokens)
    quantizer = AudioTokenizer(config["quantizer"], tokens_config)

    codebook_size = (
        config["quantizer"]["speech"]["n_new_tokens"]
        + config["quantizer"]["wav"]["n_new_tokens"]
    )
    print("New tokens:", codebook_size)
    train_dataset, val_dataset = load_data(data, tokenizer, quantizer, config)

    if training_args.few_val_samples > 0:
        for i, ds in enumerate(val_dataset.datasets):
            val_dataset.datasets[i].dataset = ds.dataset.select(range(training_args.few_val_samples))

    new_embeddings_count = n_tokens + codebook_size
    model = _build_model(training_args, config, new_embeddings_count=new_embeddings_count)

    trainer = Trainer(
        model,
        tokenizer=tokenizer,
        args=training_args,

        # Data settings
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda x: collate_fn(x, tokenizer, config["max_seq_length"]),
    )

    trainer.accelerator.log_with = [ 'wandb' ]
    trainer.accelerator.init_trackers(
        project_name=config["wandb_project_name"],
    )

    trainer.train()