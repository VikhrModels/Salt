from torch.utils.data import Dataset
from src.utils import create_audio_tokens


class SaltBaseDataset(Dataset):
    def __init__(
        self, tokenizer, hf_dataset, max_text_length=512, max_audio_sequence_length=2048
    ):
        self.tokenizer = tokenizer
        self.hf_dataset = hf_dataset
        self.max_total_length = max_text_length + max_audio_sequence_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]

        text = example["text"]
        audio_tokens = example["audio_tokens"]
        ready_tokens = create_audio_tokens(audio_tokens)

        conversation = [
            {
                "role": "system",
                "content": "Вы полезный помощник по озвучиванию текста",
            },
            {
                "role": "user",
                "content": f"Преобразуй это в аудио: {text}",
            },
            {"role": "assistant", "content": " ".join(ready_tokens)},
        ]

        chat_text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_conversation = conversation[:-1]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_length = len(
            self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        )

        tokenized = self.tokenizer(
            chat_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_total_length,
            return_tensors="pt",
            padding_side="right",
            add_special_tokens=False,
        )

        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
