import yaml

import sys

sys.path.append("BigCodec")


from dataclasses import dataclass, field
from typing import Optional

import torch

from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)

from BigCodec.vq.codec_decoder import CodecDecoder
from BigCodec.vq.codec_encoder import CodecEncoder
from src.compute_metrics import ComputeMetrics
from src.data import load_data
from src.tokenizer import AudioTokenizer, get_start_tokens
from src.utils.training import collate_fn


@dataclass
class SaltTrainingArguments(TrainingArguments):
    # Часть параметров переопределяем из конфига
    config: str = field(default="")

    output_dir: str = field(default="./results")

    # Checkpoints
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=3000)
    save_total_limit: Optional[int] = field(default=3)

    # Training
    optim: str = field(default="adamw_torch")
    torch_compile: bool = field(default=True)

    # Eval
    include_inputs_for_metrics: bool = field(default=True)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=3000)
    batch_eval_metrics: bool = field(default=True)

    # Metrics and eval
    report_to: str = field(default="wandb")
    logging_steps: int = field(default=50)
    batch_eval_metrics: bool = field(default=True)

    # Data
    dataloader_drop_last: bool = field(default=True)
    dataloader_num_workers: int = field(default=0)
    few_val_samples: int = field(default=128)
    remove_unused_columns: bool = field(default=False)


class BigCodecTokenizer:
    def __init__(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        encoder = CodecEncoder()
        encoder.load_state_dict(ckpt["CodecEnc"])
        self.encoder = encoder.eval().cuda()

        decoder = CodecDecoder()
        decoder.load_state_dict(ckpt["generator"])
        self.decoder = decoder.eval().cuda()

    def encode(self, wav):
        vq_emb = self.encoder(wav.unsqueeze(1))
        _, vq_code, _ = self.decoder(vq_emb, vq=True)
        return vq_code


def _build_model(new_embeddings_count):
    if checkpoint_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            attn_implementation="sdpa",
            # torch_dtype=torch.bfloat16,
            cache_dir=path_to_cache,
        )
    elif base_model is not None:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            attn_implementation="sdpa",
            # torch_dtype=torch.bfloat16,
            cache_dir=path_to_cache,
        )
    else:
        config = AutoConfig.from_pretrained(config_path)
        model = AutoModelForCausalLM.from_config(
            config=config,
            attn_implementation="sdpa",
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
    config_path = config["config_path"]
    checkpoint_path = config.get("checkpoint_path")

    asr_data = config["asr_data"]
    tts_data = config["tts_data"]

    start_audio_token = config["start_audio_token"]
    end_audio_token = config["end_audio_token"]

    path_to_cache = config["path_to_cache"]

    torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = config["allow_tf32"]

    training_args.per_device_train_batch_size = config["train_batch_size"]
    training_args.per_device_eval_batch_size = config["eval_batch_size"]
    training_args.num_train_epochs = config["num_train_epochs"]

    training_args.weight_decay = float(config["weight_decay"])
    training_args.learning_rate = float(config["learning_rate"])
    training_args.max_grad_norm = float(config["max_grad_norm"])
    training_args.lr_scheduler_type = config["lr_scheduler_type"]
    training_args.warmup_steps = int(config["num_warmup_steps"])
    training_args.gradient_accumulation_steps = int(
        config["gradient_accumulation_steps"]
    )

    if base_model is not None:
        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=path_to_cache)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_path, cache_dir=path_to_cache)

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
    print("Num non-audio tokens:", n_tokens)

    start_audio_token_id = tokenizer._convert_token_to_id_with_added_voc(
        start_audio_token
    )
    end_audio_token_id = tokenizer._convert_token_to_id_with_added_voc(end_audio_token)

    tokens_config = get_start_tokens(config["quantizer"], n_tokens)
    quantizer = AudioTokenizer(config["quantizer"], tokens_config)
    print(tokens_config)

    codebook_size = (
        config["quantizer"]["speech"]["n_new_tokens"]
        + config["quantizer"]["wav"]["n_new_tokens"]
        + config["quantizer"]["bigcodec"]["n_new_tokens"]
    )
    print("New tokens:", codebook_size)
    train_dataset, val_dataset = load_data(
        asr_data,
        tts_data,
        tokenizer,
        quantizer,
        config,
        few_val_samples=training_args.few_val_samples,
    )

    new_embeddings_count = n_tokens + codebook_size
    model = _build_model(new_embeddings_count=new_embeddings_count)

    # Костыль, чтобы не падало из-за отдельного параметра is_asr
    # Он нужен для вычисления метрик
    orig_model_forward = model.forward

    def crutch_is_asr(*args, **kwargs):
        del kwargs["is_asr"]
        return orig_model_forward(*args, **kwargs)

    model.forward = crutch_is_asr

    trainer = Trainer(
        model,
        tokenizer=tokenizer,
        args=training_args,
        # Data settings
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda x: collate_fn(x, tokenizer, config["max_seq_length"]),
    )
    trainer.compute_metrics = ComputeMetrics(trainer, config["tasks"])

    trainer.accelerator.log_with = ["wandb"]
    trainer.accelerator.init_trackers(
        project_name=config["wandb_project_name"],
    )

    trainer.train()
