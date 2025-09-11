import os

import hydra
import torch
import wandb

from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import TrainingArguments, Trainer

from salt.utils.data_utils import prepare_data
from salt.utils.debug_utils import verify_audio_from_tokens_reconstruction
from salt.utils.loading_utils import prepare_model_and_tokenizer, load_audio_tokenizer
from salt.utils.training_utils import collate_fn, fix_seed


DEBUG = False


@hydra.main(config_path="configs", config_name="default")
def main(config: DictConfig):
    if config.path_to_cache is not None:
        os.environ["HF_HOME"] = config.path_to_cache
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch._dynamo.config.suppress_errors = False
    torch._inductor.config.debug = True

    load_dotenv()
    wandb.login(key=os.getenv("WB_KEY"))

    fix_seed(42)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        # Training
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.num_warmup_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim,
        torch_compile=config.torch_compile,
        # Checkpoints
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        # Eval
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        # Logging
        report_to=["wandb"],
        logging_steps=50,
        run_name=config.wandb_project_name,
    )

    model, tokenizer = prepare_model_and_tokenizer(config)
    train_data, val_data = prepare_data(config, tokenizer)
    max_seq_length = config.max_text_tokens + config.max_audio_tokens

    if DEBUG:
        verify_audio_from_tokens_reconstruction(config, train_data[0]["input_ids"])

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
