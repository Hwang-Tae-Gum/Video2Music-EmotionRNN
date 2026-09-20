"""
tea_adapter.py
──────────────
Temporal Signal Adapter (구: Temporal Emotion Adapter)

입력: feature  (batch, seq_len, input_dim)
출력: embedding (batch, seq_len, d_model)

구성
  [1] 학습 가능한 signal_weight  (input_dim,)  + temperature scaling
       → 추론 시 weight_override 로 사용자가 직접 조절 가능 (emotion 전용)
  [2] apply_temporal_window  – torch.unfold 완전 벡터화, 지수 감쇠 가중치
  [3] RNN  (단방향 or 양방향, lstm or gru) | Self-Attention (attention)
  [4] Residual + LayerNorm
  [5] output_proj  Linear(d_model, d_model)
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from utilities.constants_TEA import (
    EMO_DIM, TEA_HIDDEN_SIZE, TEA_NUM_LAYERS,
    TEA_WINDOW_SIZE, TEA_OVERLAP,
)


class TemporalEmotionAdapter(nn.Module):
    """
    Parameters
    ----------
    d_model      : int   – Transformer d_model (기본 512)
    input_dim    : int   – 입력 신호 차원 (emotion=6, scene/motion=1)
    direction    : str   – "uni" (Decoder용) | "bi" (Encoder용)
    cell_type    : str   – "lstm" | "gru"
    num_layers   : int   – RNN stacked layer 수
    window_size  : int   – 시간 윈도우 크기
    overlap      : float – 윈도우 오버랩 비율
    """

    def __init__(
        self,
        d_model: int = TEA_HIDDEN_SIZE,
        input_dim: int = EMO_DIM,
        direction: str = "uni",
        cell_type: str = "gru",
        num_layers: int = TEA_NUM_LAYERS,
        window_size: int = TEA_WINDOW_SIZE,
        overlap: float = TEA_OVERLAP,
    ):
        super().__init__()

        self.d_model     = d_model
        self.input_dim   = input_dim
        self.direction   = direction          # "uni" | "bi"
        self.cell_type   = cell_type.lower()  # "lstm" | "gru" | "attention"
        self.num_layers  = num_layers
        self.window_size = window_size
        self.overlap     = overlap
        self.bidirectional = (direction == "bi")

        # ── [0] 입력 dropout ─────────────────────────────────────────────────
        self.input_dropout = nn.Dropout(p=0.2)

        # ── [1] 학습 가능한 신호 가중치 ──────────────────────────────────────
        self.signal_weight = nn.Parameter(torch.ones(input_dim))
        self.temperature   = nn.Parameter(torch.tensor(1.0))

        # ── [2] 시간 윈도우의 지수 감쇠 가중치 (고정) ────────────────────────
        w = torch.exp(torch.linspace(0.0, 1.0, window_size))
        w = w / w.sum()
        self.register_buffer("decay_weight", w)   # (window_size,)

        # ── [4] Residual projection + LayerNorm (공통) ───────────────────────
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # ── [3] Temporal encoder: RNN or Self-Attention ───────────────────────
        if self.cell_type == "attention":
            # Transformer encoder layer 구조: MHA + FFN
            self.attn = nn.MultiheadAttention(
                embed_dim   = d_model,
                num_heads   = 8,
                dropout     = 0.1,
                batch_first = True,
            )
            self.attn_ff = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
            )
            self.attn_ff_norm = nn.LayerNorm(d_model)
        else:
            hidden  = d_model // 2 if self.bidirectional else d_model
            rnn_cls = nn.LSTM if self.cell_type == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size    = input_dim,
                hidden_size   = hidden,
                num_layers    = num_layers,
                batch_first   = True,
                bidirectional = self.bidirectional,
                dropout       = 0.1 if num_layers > 1 else 0.0,
            )

        # ── [5] Output projection ─────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, d_model)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # 구버전 체크포인트(멀티시그널 이전) 호환: emotion_weight → signal_weight 개명
        legacy_key = prefix + "emotion_weight"
        new_key = prefix + "signal_weight"
        if legacy_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(legacy_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _apply_signal_weight(
        self,
        x: torch.Tensor,
        override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x        : (batch, seq, input_dim)
        override : (input_dim,) or None
        반환     : (batch, seq, input_dim)
        """
        if override is not None:
            w = torch.softmax(override / self.temperature.abs().clamp(min=1e-6), dim=-1)
        else:
            w = torch.softmax(self.signal_weight / self.temperature.abs().clamp(min=1e-6), dim=-1)
        return x * w.unsqueeze(0).unsqueeze(0)

    def _apply_temporal_window(self, x: torch.Tensor) -> torch.Tensor:
        """
        슬라이딩 윈도우 + 지수 감쇠 가중치 평균 (완전 벡터화).
        x : (batch, seq, input_dim)
        반환 : (batch, seq, input_dim)
        """
        B, T, C  = x.shape
        W        = self.window_size
        pad_size = W - 1

        x_pad = torch.nn.functional.pad(x, (0, 0, pad_size, 0))
        x_unf = x_pad.unfold(dimension=1, size=W, step=1)   # (B, T, C, W)
        x_unf = x_unf.permute(0, 1, 3, 2)                   # (B, T, W, C)

        w = self.decay_weight.view(1, 1, W, 1)
        return (x_unf * w).sum(dim=2)                        # (B, T, C)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        feature: torch.Tensor,
        weight_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        feature         : (batch, seq_len, input_dim)
        weight_override : (input_dim,) or None  — emotion override 용
        반환            : (batch, seq_len, d_model)
        """
        # [1] 신호 가중치
        x = self._apply_signal_weight(feature, weight_override)

        x = self.input_dropout(x)

        # [2] 시간 윈도우
        x = self._apply_temporal_window(x)

        # [4] Residual projection (공통)
        residual = self.input_proj(x)      # (B, T, d_model)

        if self.cell_type == "attention":
            # [3-A] Self-Attention sub-layer
            T = residual.size(1)
            attn_mask = None
            if self.direction == "uni":
                # causal mask: uni-directional adapter용
                attn_mask = torch.triu(
                    torch.ones(T, T, device=residual.device, dtype=torch.bool),
                    diagonal=1,
                )
            attn_out, _ = self.attn(residual, residual, residual, attn_mask=attn_mask)
            h = self.layer_norm(residual + attn_out)   # post-attn residual+norm

            # [3-B] FFN sub-layer
            ff_out = self.attn_ff(h)
            out = self.attn_ff_norm(h + ff_out)
        else:
            # [3] RNN
            rnn_out, _ = self.rnn(x)       # (B, T, d_model)

            # [4] Residual + LayerNorm
            out = self.layer_norm(residual + rnn_out)

        # [5] Output projection
        return self.output_proj(out)
