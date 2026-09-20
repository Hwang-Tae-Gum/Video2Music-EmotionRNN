"""
contrastive_loss.py
────────────────────
Supervised Contrastive Loss (SupCon) for emotion embeddings.
Khosla et al. (2020) "Supervised Contrastive Learning"

사용법:
  loss_fn = SupConLoss(temperature=0.07)
  emb     = model.get_emotion_emb(feature_emotion)  # (B, proj_dim), L2-norm
  labels  = feature_emotion.mean(dim=1).argmax(dim=-1)  # (B,)
  loss    = loss_fn(emb, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Parameters
    ----------
    temperature : float
        InfoNCE temperature τ. 낮을수록 경계 선명, 높을수록 부드러움.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,   # (N, D) — L2-normalized 임베딩
        labels:   torch.Tensor,   # (N,)   — 감정 클래스 인덱스 (int64)
    ) -> torch.Tensor:
        """
        Returns scalar loss. 배치 내 같은 감정 클래스 = positive pair.
        배치 크기 < 2 이거나 모든 샘플이 동일 클래스면 0 반환.
        """
        N = features.shape[0]
        if N < 2:
            return features.sum() * 0.0  # autograd 그래프 유지

        # ── pairwise cosine similarity (이미 L2-norm) ────────────────────────
        sim = torch.matmul(features, features.T) / self.temperature  # (N, N)

        # ── positive / self 마스크 ───────────────────────────────────────────
        labels_col = labels.view(-1, 1)
        pos_mask   = (labels_col == labels_col.T).float()             # (N, N)
        self_mask  = torch.eye(N, dtype=torch.bool, device=features.device)
        pos_mask.masked_fill_(self_mask, 0.0)  # 자기 자신 제외

        n_pos = pos_mask.sum(dim=1)  # (N,)  각 앵커의 positive 수
        valid = n_pos > 0            # positive 가 있는 앵커만 손실 계산

        if valid.sum() == 0:
            return features.sum() * 0.0

        # ── log-sum-exp 분모: self 제외, 별도 텐서로 ─────────────────────────
        # NOTE: sim 을 in-place 로 바꾸면 분자 계산에서 0 * (-inf) = NaN 발생 →
        #       masked_fill (non-in-place) 로 새 텐서 생성
        log_denom = torch.logsumexp(
            sim.masked_fill(self_mask, float("-inf")), dim=1
        )  # (N,)

        # ── SupCon 손실 ─────────────────────────────────────────────────────
        # loss_i = -1/|P(i)| Σ_{j∈P(i)} [sim(i,j) - log_denom_i]
        # pos_mask 의 대각선은 0 이므로 sim 의 self-sim 값은 곱해서 사라짐
        loss_per = -(pos_mask * (sim - log_denom.unsqueeze(1))).sum(dim=1)
        loss_per = loss_per / n_pos.clamp(min=1)   # valid 아닌 행은 /1 로 무해

        return loss_per[valid].mean()
