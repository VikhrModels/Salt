import os

import hydra
import torch
import wandb

from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import TrainingArguments, Trainer

from salt.utils.data_utils import prepare_data
from salt.utils.loading_utils import prepare_model_and_tokenizer
from salt.utils.training_utils import collate_fn


@hydra.main(config_path="configs", config_name="default.yaml")
def main(config: DictConfig):
    os.environ["HF_HOME"] = config.training.path_to_cache
    torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = config["allow_tf32"]

    load_dotenv()
    wandb.login(key=os.getenv("WB_KEY"))

    training_args = TrainingArguments(
        output_dir=config.training.output_dir,

        # Training
        per_device_train_batch_size=config.training.train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        num_train_epochs=config.training.num_train_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_steps=config.training.num_warmup_steps,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        optim=config.training.optim,
        torch_compile=config.training.torch_compile,

        # Checkpoints
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,

        # Eval
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,

        # Logging
        report_to=["wandb"],
        logging_steps=50,
        run_name=config.training.wandb_project_name,
    )

    model, tokenizer = prepare_model_and_tokenizer(config)
    train_data, val_data = prepare_data(config, tokenizer)
    max_seq_length = config.training.max_text_tokens + config.training.max_audio_tokens

    trainer = Trainer(
        model,
        tokenizer=tokenizer,
        args=training_args,
        # Data settings
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=lambda x: collate_fn(x, tokenizer, max_seq_length),
    )

    trainer.train()



if __name__ == "__main__":
    main()
