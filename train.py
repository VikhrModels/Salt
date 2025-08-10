import os

os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

from unsloth import FastModel
from transformers import Trainer, TrainingArguments, Qwen2Model

from datasets import load_dataset, concatenate_datasets, Audio

from salt.dataset import SaltDataset
from salt.modeling import SaltForAudioGeneration
from salt.utils import salt_collate_fn

import torch

#ds_one = load_dataset("Vikhrmodels/ToneSlavic_quantized-bigcodec")
ds_two = load_dataset("Vikhrmodels/ToneWebinars_quantized-bigcodec")
ds_three = load_dataset("Vikhrmodels/ToneBooksPlus_quantized-bigcodec")
ds_four = load_dataset("Vikhrmodels/ToneSpeak_quantized-bigcodec")
ds_five = load_dataset("Vikhrmodels/ToneRuLS_quantized-bigcodec")

# Добавляем train split
train_ds = concatenate_datasets(
    [
        #ds_one["train"],
        ds_two["train"],
        ds_three["train"],
        ds_four["train"],
        ds_five["train"],
    ]
)

val_ds = concatenate_datasets(
    [
        #ds_one["validation"].select(range(279)),
        ds_two["validation"].select(range(279)),
        ds_three["validation"].select(range(279)),
        ds_four["validation"].select(range(279)),
        ds_five["validation"].select(range(279)),
    ]
)


train_ds = train_ds.remove_columns(
    [col for col in train_ds.column_names if col not in ["text", "audio_tokens"]]
)
val_ds = val_ds.remove_columns(
    [col for col in val_ds.column_names if col not in ["text", "audio_tokens"]]
)


language_model, tokenizer = FastModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    max_seq_length=2048,
    auto_model=Qwen2Model,
    trust_remote_code=True,
    full_finetuning=True

)


ready_train_dataset = SaltDataset(
    tokenizer=tokenizer,
    hf_dataset=train_ds,
    max_audio_tokens=4096,
    max_text_length=2048,
)

ready_val_dataset = SaltDataset(
    tokenizer=tokenizer,
    hf_dataset=val_ds,
    max_audio_tokens=4096,
    max_text_length=2048,
)


model = SaltForAudioGeneration(
    language_model=language_model,
    freeze_text_encoder=True,
    d_model=768,
    n_layers=8,
    n_heads=12,
    ff_mult=4.0,
    dropout=0.1,
    codebook_size=8192,
    max_audio_len=4096,
    soft_prompt_len=16,
    label_smoothing=0.1,
)

args = TrainingArguments(
    output_dir="tts_bigcodec_qwen_unsloth",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_ratio=0.06,
    num_train_epochs=3,
    bf16=True,  
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=2,
    report_to="wandb",
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ready_train_dataset,
    eval_dataset=ready_val_dataset,
    data_collator=salt_collate_fn,
)

trainer.train()
