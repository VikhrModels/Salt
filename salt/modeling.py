import torch
import torch.nn as nn
from .utils import CrossSelfAttnLayer
from typing import Optional, Tuple, List, Dict
import torch.nn.functional as F


BOS_ID = 8192
EOS_ID = 8193
PAD_ID = 8194


class SaltForAudioGeneration(nn.Module):
    def __init__(
        self,
        language_model,
        d_model: int = 768,
        n_layers: int = 8,
        n_heads: int = 12,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        codebook_size: int = 8192,
        max_audio_len: int = 8192,
        soft_prompt_len: int = 16,
        label_smoothing: float = 0.1,
        freeze_text_encoder: bool = True,
        rope_base: int = 10_000,
    ):
        super().__init__()
        self.language_model = language_model
        self.v_audio = codebook_size
        self.v_total = codebook_size + 3
        self.id_bos = BOS_ID
        self.id_eos = EOS_ID
        self.id_pad = PAD_ID
        self.max_audio_len = max_audio_len
        self.ignore_index = self.id_pad
        self.label_smoothing = label_smoothing

        if freeze_text_encoder:
            for param in self.language_model.parameters():
                param.requires_grad = False
            self.language_model.eval()

        language_model_dim = self.language_model.config.hidden_size
        self.llm_projector = nn.Linear(language_model_dim, d_model, bias=False)

        self.soft_prompt = nn.Parameter(torch.randn(soft_prompt_len, d_model) * 0.02)
        self.soft_norm = nn.RMSNorm(d_model)

        self.audio_tok_emb = nn.Embedding(
            self.v_total, d_model, padding_idx=self.id_pad
        )
        nn.init.normal_(self.audio_tok_emb.weight, std=0.02)

        d_ff = int(ff_mult * d_model)
        self.layers = nn.ModuleList(
            [
                CrossSelfAttnLayer(
                    d_model, n_heads, d_ff, dropout=dropout, rope_base=rope_base
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.v_total, bias=False)
        nn.init.xavier_uniform_(self.lm_head.weight)

        self.len_head = nn.Linear(d_model, 1, bias=False)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = self.language_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        mem = self.llm_projector(out.last_hidden_state)  # (B,S,D)
        # prepend soft prompt
        B = mem.size(0)
        sp = self.soft_prompt.unsqueeze(0).expand(B, -1, -1)  # (B,P,D)
        mem = torch.cat([self.soft_norm(sp), mem], dim=1)  # (B,P+S,D)
        mem_kpm = None
        if attention_mask is not None:
            sp_mask = torch.ones(
                B, sp.size(1), dtype=attention_mask.dtype, device=attention_mask.device
            )
            mem_kpm = torch.cat([sp_mask, attention_mask], dim=1) == 0
        return mem, mem_kpm

    def _shift_right(self, targets: torch.Tensor) -> torch.Tensor:
        B, L = targets.shape
        bos = torch.full(
            (B, 1), self.id_bos, dtype=targets.dtype, device=targets.device
        )
        return torch.cat([bos, targets[:, :-1]], dim=1)

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        audio_input_ids: Optional[
            torch.Tensor
        ] = None,  # (B,L_in), BOS optional (we add it if using targets)
        audio_targets: Optional[torch.Tensor] = None,  # (B,L_tgt): 0..8191 + EOS + PAD
        audio_pad_mask: Optional[
            torch.Tensor
        ] = None,  # (B,L_in) True=PAD for input ids
        target_lengths: Optional[
            torch.Tensor
        ] = None,  # (B,) true len incl EOS, excl PAD
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        mem, mem_kpm = self.encode_text(text_input_ids, text_attention_mask)

        if audio_input_ids is None and audio_targets is not None:
            audio_input_ids = self._shift_right(audio_targets)
        if audio_input_ids is None:
            raise ValueError("Provide audio_input_ids or audio_targets.")

        B, L = audio_input_ids.shape
        if L > self.max_audio_len + 1:
            raise ValueError(f"L={L} > max_audio_len+1={self.max_audio_len + 1}")

        if audio_pad_mask is None:
            audio_pad_mask = audio_input_ids == self.id_pad  # (B,L)

        x = self.audio_tok_emb(audio_input_ids)  # (B,L,D)

        # causal mask (L,L)
        causal = torch.full((L, L), float("-inf"), device=x.device)
        causal = torch.triu(causal, diagonal=1)

        for i, layer in enumerate(self.layers):
            x, _ = layer(
                x,
                memory=mem,
                causal_mask=causal,
                mem_kpm=mem_kpm,
                tgt_kpm=audio_pad_mask,
                cache=None,
                pos_offset=0,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B,L,V_total)

        out: Dict[str, torch.Tensor] = {"logits": logits}

        if return_loss and audio_targets is not None:
            L_t = audio_targets.size(1)
            logits = logits[:, :L_t, :]
            loss = F.cross_entropy(
                logits.reshape(-1, self.v_total),
                audio_targets.reshape(-1),
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
            )
            out["loss"] = loss

            if target_lengths is not None:
                pooled = mem.mean(dim=1)
                len_pred = self.len_head(pooled).squeeze(-1)
                len_loss = F.smooth_l1_loss(len_pred, target_lengths.to(len_pred.dtype))
                out["len_loss"] = len_loss
                out["loss"] = out["loss"] + 0.05 * len_loss

        return out

    @torch.inference_mode()
    def generate(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        max_new_frames: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.1,
        stop_on_eos: bool = True,
        use_length_head: bool = True,
    ) -> torch.Tensor:
        device = text_input_ids.device
        mem, mem_kpm = self.encode_text(text_input_ids, text_attention_mask)
        B = text_input_ids.size(0)

        if use_length_head and max_new_frames is None:
            pooled = mem.mean(dim=1)
            len_pred = self.len_head(pooled).clamp(
                min=1.0, max=float(self.max_audio_len)
            )
            max_new_frames = int(len_pred.max().round().item())
        if max_new_frames is None:
            max_new_frames = 200

        seq = torch.full((B, 1), self.id_bos, dtype=torch.long, device=device)
        generated = []
        freq = torch.zeros(B, self.v_total, device=device)

        # per-layer caches
        caches: List[Dict[str, torch.Tensor]] = [
            {"k_self": None, "v_self": None, "k_mem": None, "v_mem": None}
            for _ in self.layers
        ]
        seen = 0

        for _ in range(max_new_frames):
            x = self.audio_tok_emb(seq[:, -1:])  # (B,1,D)

            for i, layer in enumerate(self.layers):
                x, caches[i] = layer(
                    x,
                    memory=mem,
                    causal_mask=None,  # causal via KV-cache one-step feed
                    mem_kpm=mem_kpm,
                    tgt_kpm=None,
                    cache=caches[i],
                    pos_offset=seen,
                )

            x = self.final_norm(x)
            logits = self.lm_head(x[:, -1])  # (B,V)

            # repetition penalty
            if repetition_penalty and repetition_penalty > 1.0:
                logits = logits / (
                    1.0 + (freq > 0).float() * (repetition_penalty - 1.0)
                )

            # sampling
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)

            if top_k is not None and top_k > 0:
                k = min(top_k, probs.size(-1))
                vals, idx = torch.topk(probs, k=k, dim=-1)
                pick = torch.multinomial(vals, num_samples=1)
                next_tok = torch.gather(idx, 1, pick)
            elif top_p is not None and 0 < top_p < 1:
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                csum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (csum >= top_p).float().argmax(dim=-1, keepdim=True)
                mask = (
                    torch.arange(probs.size(-1), device=device).unsqueeze(0) <= cutoff
                )
                fp = sorted_probs.masked_fill(~mask, 0)
                fp = fp / fp.sum(dim=-1, keepdim=True)
                pick = torch.multinomial(fp, num_samples=1)
                next_tok = torch.gather(sorted_idx, 1, pick)
            else:
                next_tok = torch.argmax(probs, dim=-1, keepdim=True)

            generated.append(next_tok)
            seq = torch.cat([seq, next_tok], dim=1)
            seen += 1
            freq.scatter_add_(1, next_tok, torch.ones_like(next_tok, dtype=freq.dtype))

            if stop_on_eos and (next_tok == self.id_eos).all():
                break

        out = (
            torch.cat(generated, dim=1)
            if generated
            else torch.empty(B, 0, dtype=torch.long, device=device)
        )

        if stop_on_eos:
            for b in range(B):
                eos_pos = (out[b] == self.id_eos).nonzero(as_tuple=True)[0]
                if eos_pos.numel() > 0:
                    end = eos_pos[0].item()
                    out[b, end:] = self.id_pad
        return out
