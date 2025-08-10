import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from torchtune.modules import RotaryPositionalEmbeddings


class CrossSelfAttnLayer(nn.Module):
    """
    One decoder block with:
      - RMSNorm pre-norms
      - Self-attention (+ RoPE on q,k)
      - Cross-attention to text memory
      - SwiGLU FFN
    Uses torch.nn.RMSNorm and torchtune RotaryPositionalEmbeddings.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        rope_base: int = 10_000,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Self-attn projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Cross-attn projections
        self.qx_proj = nn.Linear(d_model, d_model, bias=False)
        self.km_proj = nn.Linear(d_model, d_model, bias=False)
        self.vm_proj = nn.Linear(d_model, d_model, bias=False)
        self.ox_proj = nn.Linear(d_model, d_model, bias=False)

        self.drop = nn.Dropout(dropout)

        # RoPE on self-attn q,k
        self.rope = RotaryPositionalEmbeddings(
            dim=self.d_head, max_seq_len=8192, base=rope_base
        )

        # Norms + FFN (SwiGLU)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.norm3 = nn.RMSNorm(d_model)

        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # SwiGLU
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.ff_drop = nn.Dropout(dropout)

    @staticmethod
    def _reshape_bld_to_blhd(x: torch.Tensor, n_heads: int) -> torch.Tensor:
        B, L, D = x.shape
        Dh = D // n_heads
        return x.view(B, L, n_heads, Dh)

    @staticmethod
    def _blhd_to_bhld(x: torch.Tensor) -> torch.Tensor:
        # (B, L, H, Dh) -> (B, H, L, Dh)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def _merge_bhld(x: torch.Tensor) -> torch.Tensor:
        # (B, H, L, Dh) -> (B, L, H*Dh)
        B, H, L, Dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)

    def _sdp_attn(
        self,
        q_bhld: torch.Tensor,
        k_bhld: torch.Tensor,
        v_bhld: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # q,k,v: (B,H,Lq,Dh)/(B,H,Lk,Dh), mask broadcastable to (B, H, Lq, Lk)
        out = F.scaled_dot_product_attention(
            q_bhld, k_bhld, v_bhld, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        return out  # (B,H,Lq,Dh)

    def forward(
        self,
        x: torch.Tensor,  # (B, L, D) — target states
        memory: torch.Tensor,  # (B, S, D) — text memory
        causal_mask: Optional[torch.Tensor],  # (L, L) float mask (-inf/0)
        mem_kpm: Optional[torch.Tensor],  # (B, S) True=PAD for memory
        tgt_kpm: Optional[torch.Tensor],  # (B, L) True=PAD for target keys (self)
        cache: Optional[Dict[str, torch.Tensor]] = None,
        pos_offset: int = 0,  # start position index for RoPE (KV-cache)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns updated x and cache (stores rotated k_self, v_self and k_mem, v_mem).
        """
        B, L, _ = x.shape
        # ---------- Self-attention ----------
        h = self.norm1(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q_blhd = self._reshape_bld_to_blhd(q, self.n_heads)
        k_blhd = self._reshape_bld_to_blhd(k, self.n_heads)
        v_bhld = self._blhd_to_bhld(
            self._reshape_bld_to_blhd(v, self.n_heads)
        )  # (B,H,L,Dh)

        # Apply RoPE to q,k (expects [B, L, H, Dh])
        pos = torch.arange(
            pos_offset, pos_offset + L, device=x.device, dtype=torch.long
        )
        q_blhd = self.rope(q_blhd, input_pos=pos)  # (B,L,H,Dh)
        k_blhd = self.rope(k_blhd, input_pos=pos)

        q_bhld = self._blhd_to_bhld(q_blhd)  # (B,H,L,Dh)
        k_bhld = self._blhd_to_bhld(k_blhd)  # (B,H,L,Dh)

        # KV cache for self
        if cache is not None and cache.get("k_self") is not None:
            k_bhld = torch.cat([cache["k_self"], k_bhld], dim=2)
            v_bhld = torch.cat([cache["v_self"], v_bhld], dim=2)

        # Build combined mask: causal + key padding (PAD keys masked)
        attn_mask = None
        if causal_mask is not None:
            # expand to (B,1,Lq,Lk_total)
            attn_mask = (
                causal_mask.unsqueeze(0).unsqueeze(1).expand(B, 1, L, k_bhld.size(2))
            )
        if tgt_kpm is not None:
            key_pad = tgt_kpm.unsqueeze(1).unsqueeze(
                2
            )  # (B,1,1,Lk_total) broadcast to (B,1,Lq,Lk_total)
            pad_mask = (
                torch.zeros_like(attn_mask)
                if attn_mask is not None
                else torch.zeros(B, 1, L, k_bhld.size(2), device=x.device)
            )
            pad_mask = pad_mask.masked_fill(key_pad, float("-inf"))
            attn_mask = pad_mask if attn_mask is None else attn_mask + pad_mask

        x_attn = self._sdp_attn(
            q_bhld, k_bhld, v_bhld, attn_mask=attn_mask
        )  # (B,H,L,Dh)
        x = x + self.drop(self.o_proj(self._merge_bhld(x_attn)))

        # ---------- Cross-attention ----------
        h = self.norm2(x)
        qx = self.qx_proj(h)

        # cache memory K,V once
        if cache is not None and cache.get("k_mem") is not None:
            km_bhld, vm_bhld = cache["k_mem"], cache["v_mem"]
        else:
            km = self.km_proj(memory)  # (B,S,D)
            vm = self.vm_proj(memory)
            km_bhld = self._blhd_to_bhld(self._reshape_bld_to_blhd(km, self.n_heads))
            vm_bhld = self._blhd_to_bhld(self._reshape_bld_to_blhd(vm, self.n_heads))
            if cache is not None:
                cache["k_mem"] = km_bhld
                cache["v_mem"] = vm_bhld

        qx_bhld = self._blhd_to_bhld(self._reshape_bld_to_blhd(qx, self.n_heads))

        cross_mask = None
        if mem_kpm is not None:
            key_pad = mem_kpm.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
            cross_mask = torch.zeros(B, 1, L, km_bhld.size(2), device=x.device)
            cross_mask = cross_mask.masked_fill(key_pad, float("-inf"))

        x_cross = self._sdp_attn(qx_bhld, km_bhld, vm_bhld, attn_mask=cross_mask)
        x = x + self.drop(self.ox_proj(self._merge_bhld(x_cross)))

        # ---------- FFN ----------
        h = self.norm3(x)
        x = x + self.ff_drop(F.silu(self.w1(h)) * self.w2(h) @ self.w3.weight.T)

        if cache is not None:
            cache["k_self"] = k_bhld
            cache["v_self"] = v_bhld

        return x, (cache or {})


def salt_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "text_input_ids": torch.stack(
            [b["text_input_ids"] for b in batch], dim=0
        ),  # (B, S)
        "text_attention_mask": torch.stack(
            [b["text_attention_mask"] for b in batch], dim=0
        ),  # (B, S)
        "audio_targets": torch.stack(
            [b["audio_targets"] for b in batch], dim=0
        ),  # (B, L)
        "audio_pad_mask": torch.stack(
            [b["audio_pad_mask"] for b in batch], dim=0
        ),  # (B, L)
        "target_lengths": torch.stack(
            [b["target_lengths"] for b in batch], dim=0
        ),  # (B,)
    }
