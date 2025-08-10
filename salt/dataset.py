import torch
from torch.utils.data import Dataset

BOS_ID = 8192
EOS_ID = 8193
PAD_ID = 8194


class SaltDataset(Dataset):
    """
    Ожидает в hf_dataset колонку:
      - "text": str
      - "audio_tokens": List[int]  # только 0..8191, без спец-символов
    """

    def __init__(
        self,
        tokenizer,
        hf_dataset,
        max_text_length: int,
        max_audio_tokens: int,
        eos_id: int = EOS_ID,
        pad_id: int = PAD_ID,
    ):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.max_audio_tokens = max_audio_tokens
        self.eos_id = eos_id
        self.pad_id = pad_id

        # Удаляем лишние колонки
        self.hf_dataset = hf_dataset

        # Фильтруем по длине audio_tokens
        self.hf_dataset = self.hf_dataset.filter(
            lambda x: len(x["audio_tokens"]) <= self.max_audio_tokens
        )

        # Устанавливаем pad_token, если не задан
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token
            )

    def __len__(self):
        return len(self.hf_dataset)

    def _build_audio_targets(self, raw_tokens):
        """
        raw_tokens: iterable[int] из 0..8191 (без EOS/PAD)
        Возвращает:
          audio_targets: LongTensor[L_max]
          audio_pad_mask: BoolTensor[L_max]
          target_len: int   # длина с учётом EOS, без PAD
        """
        # приведём к списку int
        toks = list(map(int, raw_tokens)) if raw_tokens is not None else []
        # граничные случаи
        if len(toks) == 0:
            # хотя бы EOS
            seq = [self.eos_id]
        else:
            seq = toks + [self.eos_id]

        # обрезка/паддинг до max_audio_tokens
        if len(seq) > self.max_audio_tokens:
            # гарантируем, что последний — EOS
            seq = seq[: self.max_audio_tokens]
            if seq[-1] != self.eos_id:
                seq[-1] = self.eos_id

        target_len = len(seq)  # включает EOS

        if target_len < self.max_audio_tokens:
            pad_count = self.max_audio_tokens - target_len
            seq = seq + [self.pad_id] * pad_count

        audio_targets = torch.tensor(seq, dtype=torch.long)
        audio_pad_mask = audio_targets == self.pad_id
        return audio_targets, audio_pad_mask, target_len

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        text = item["text"]
        raw_audio = item["audio_tokens"]  # список int (0..8191)

        # --- текст ---
        tok = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].squeeze(0).to(torch.long)  # (S,)
        attention_mask = tok["attention_mask"].squeeze(0).to(torch.long)  # (S,)

        # --- аудио-цели ---
        audio_targets, audio_pad_mask, target_len = self._build_audio_targets(raw_audio)

        return {
            # имена под forward модели
            "text_input_ids": input_ids,  # (S,)
            "text_attention_mask": attention_mask,  # (S,)
            "audio_targets": audio_targets,  # (L_max,)
            "audio_pad_mask": audio_pad_mask,  # (L_max,) bool
            "target_lengths": torch.tensor(target_len),  # ()
        }
