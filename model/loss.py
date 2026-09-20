import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

# chord quality id → emotional valence scalar
_QUALITY_VALENCE = {
    0: 0.0, 1: 1.0, 2: -1.0, 3: 0.0, 4: -0.5, 5: -1.0, 6: 0.0,
    7: 0.5, 8: -1.0, 9: 1.0, 10: -0.7, 11: 0.3, 12: -0.5, 13: 1.0,
}

# 6c_l14p emotion order: exciting(0), fearful(1), tense(2), sad(3), relaxing(4), neutral(5)
_EMO_VALENCE_WEIGHTS = torch.tensor([1.0, -1.0, -1.0, -1.0, 1.0, 0.0])


def build_valence_vector(chord_size, chord_end_idx, chord_inv, chord_attr):
    """
    chord index → emotional valence scalar 매핑 벡터 [chord_size].
    chord_inv: {str(idx): "root:quality"}, chord_attr: {quality_str: quality_id}
    """
    vec = torch.zeros(chord_size)
    for idx in range(chord_size):
        if idx == 0 or idx >= chord_end_idx:
            continue
        chord_str = chord_inv.get(str(idx))
        if chord_str is None:
            continue
        parts = chord_str.split(":")
        q = 1 if len(parts) == 1 else chord_attr.get(parts[1], 1)
        vec[idx] = _QUALITY_VALENCE.get(q, 0.0)
    return vec


class EmotionValenceAlignLoss(nn.Module):
    """
    Emotion-Valence Alignment Loss.

    모델이 예측하는 chord 분포의 기대 감정 valence와
    video emotion 벡터의 valence 방향이 일치하도록 MSE 최적화.

    L_align = MSE(softmax(logits) @ valence_vec,  emotion @ emo_weight)

    Args:
        valence_vec:   [CHORD_SIZE]  — build_valence_vector() 로 사전 계산
        ignore_index:  CHORD_PAD 토큰 (마스킹)
    """
    def __init__(self, valence_vec: torch.Tensor, ignore_index: int):
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer("valence_vec", valence_vec)       # [CHORD_SIZE]
        self.register_buffer("emo_weight", _EMO_VALENCE_WEIGHTS)  # [6]

    def forward(self, chord_logits: torch.Tensor,
                feature_emotion: torch.Tensor,
                tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chord_logits:    [B, T, CHORD_SIZE]
            feature_emotion: [B, T, 6]
            tgt:             [B, T]  (before flatten)
        """
        probs          = F.softmax(chord_logits, dim=-1)              # [B, T, V]
        soft_chord_val = (probs * self.valence_vec).sum(-1)           # [B, T]
        video_val      = (feature_emotion * self.emo_weight).sum(-1)  # [B, T]

        mask = (tgt != self.ignore_index)
        if mask.sum() == 0:
            return chord_logits.sum() * 0.0

        return F.mse_loss(soft_chord_val[mask], video_val[mask].detach())


class MarginAlignLoss(nn.Module):
    """
    Hinge loss on emotion-chord valence sign agreement.
    AR metric(agree/disagree)을 직접 최적화: 방향 불일치에만 penalty.

    L = mean(relu(margin - pred_val * video_val))

    agreement = pred_val * video_val:
      > margin  → loss=0 (agree 또는 무관심)
      < margin  → loss = margin - agreement  (disagree penalty)
    """
    def __init__(self, valence_vec: torch.Tensor, ignore_index: int, margin: float = 0.1):
        super().__init__()
        self.ignore_index = ignore_index
        self.margin = margin
        self.register_buffer("valence_vec", valence_vec)
        self.register_buffer("emo_weight", _EMO_VALENCE_WEIGHTS)

    def forward(self, chord_logits: torch.Tensor,
                feature_emotion: torch.Tensor,
                tgt: torch.Tensor) -> torch.Tensor:
        probs          = F.softmax(chord_logits, dim=-1)
        soft_chord_val = (probs * self.valence_vec).sum(-1)           # [B, T]
        video_val      = (feature_emotion * self.emo_weight).sum(-1)  # [B, T]

        mask = (tgt != self.ignore_index)
        if mask.sum() == 0:
            return chord_logits.sum() * 0.0

        agreement = soft_chord_val[mask] * video_val[mask].detach()
        return F.relu(self.margin - agreement).mean()


# Borrowed from https://github.com/jason9693/MusicTransformer-pytorch/blob/5f183374833ff6b7e17f3a24e3594dedd93a5fe5/custom/criterion.py#L28
class SmoothCrossEntropyLoss(_Loss):
    """
    https://arxiv.org/abs/1512.00567
    """
    __constants__ = ['label_smoothing', 'vocab_size', 'ignore_index', 'reduction']

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, reduction='mean', is_logits=True):
        assert 0.0 <= label_smoothing <= 1.0
        super().__init__(reduction=reduction)

        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.input_is_logits = is_logits

    def forward(self, input, target):
        """
        Args:
            input: [B * T, V]
            target: [B * T]
        Returns:
            cross entropy: [1]
        """
        mask = (target == self.ignore_index).unsqueeze(-1)
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        u = 1.0 / self.vocab_size
        q_prime = (1.0 - self.label_smoothing) * q + self.label_smoothing * u
        q_prime = q_prime.masked_fill(mask, 0)

        ce = self.cross_entropy_with_logits(q_prime, input)
        if self.reduction == 'mean':
            lengths = torch.sum(target != self.ignore_index)
            return ce.sum() / lengths
        elif self.reduction == 'sum':
            return ce.sum()
        else:
            raise NotImplementedError

    def cross_entropy_with_logits(self, p, q):
        return -torch.sum(p * (q - q.logsumexp(dim=-1, keepdim=True)), dim=-1)


def build_chord_similarity_matrix(vocab_size, chord_end_idx, device=None):
    """
    Build a (vocab_size × vocab_size) similarity matrix based on shared pitch classes.

    Chord indices 1..(chord_end_idx-1) map to (root, quality) pairs:
      root   = (idx - 1) // 13 + 1   → 1=C ... 12=B
      quality= (idx - 1) % 13  + 1   → 1=maj .. 13=maj7

    Index 0          = N (no chord) — similarity only with itself
    Index chord_end_idx     = END token
    Index chord_end_idx + 1 = PAD token
    END/PAD are treated like N (only self-similar).
    """
    # Semitone intervals above root for each quality (1-indexed)
    QUALITY_INTERVALS = {
        1:  [0, 4, 7],            # maj
        2:  [0, 3, 6],            # dim
        3:  [0, 5, 7],            # sus4
        4:  [0, 3, 7, 10],        # min7
        5:  [0, 3, 7],            # min
        6:  [0, 2, 7],            # sus2
        7:  [0, 4, 8],            # aug
        8:  [0, 3, 6, 9],         # dim7
        9:  [0, 4, 7, 9],         # maj6
        10: [0, 3, 6, 10],        # hdim7
        11: [0, 4, 7, 10],        # 7
        12: [0, 3, 7, 9],         # min6
        13: [0, 4, 7, 11],        # maj7
    }

    def pitch_classes(chord_idx):
        """Return frozenset of pitch classes (0-11) for chord_idx."""
        if chord_idx == 0 or chord_idx >= chord_end_idx:
            return frozenset()
        root   = (chord_idx - 1) // 13      # 0-11 (semitone offset, C=0)
        q_idx  = (chord_idx - 1) % 13 + 1   # 1-13
        ivs    = QUALITY_INTERVALS.get(q_idx, [0])
        return frozenset((root + iv) % 12 for iv in ivs)

    sim = torch.zeros(vocab_size, vocab_size)
    for i in range(vocab_size):
        pc_i = pitch_classes(i)
        for j in range(vocab_size):
            if i == j:
                sim[i, j] = 1.0
                continue
            if not pc_i:
                continue
            pc_j = pitch_classes(j)
            if not pc_j:
                continue
            # Jaccard similarity
            inter = len(pc_i & pc_j)
            union = len(pc_i | pc_j)
            sim[i, j] = inter / union if union > 0 else 0.0

    if device is not None:
        sim = sim.to(device)
    return sim


class SoftCorrLoss(_Loss):
    """
    Soft-correlation cross-entropy loss.

    Instead of a one-hot target, each ground-truth chord i gets a soft target
    distribution proportional to harmonic similarity:
        t[j] = sim[i, j] / sum_k sim[i, k]

    This gives partial credit to harmonically close chords so the model is no
    longer penalised equally for predicting G:7 vs. D#:min when the truth is G.

    Args:
        sim_matrix: FloatTensor [V × V] pre-built by build_chord_similarity_matrix()
        ignore_index: token index to ignore (CHORD_PAD)
    """

    def __init__(self, sim_matrix: torch.Tensor, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.ignore_index = ignore_index
        # Pre-normalise rows once at construction time
        row_sums = sim_matrix.sum(dim=1, keepdim=True).clamp(min=1e-9)
        self.register_buffer = None  # not an nn.Module buffer; store directly
        self._soft_targets = sim_matrix / row_sums  # [V, V], rows sum to 1

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input:  [N, V] logits
            target: [N]    hard integer targets
        Returns:
            scalar loss
        """
        device = input.device
        soft = self._soft_targets.to(device)  # [V, V]

        valid_mask = (target != self.ignore_index)   # [N]
        if valid_mask.sum() == 0:
            return input.sum() * 0.0

        valid_input  = input[valid_mask]               # [M, V]
        valid_target = target[valid_mask].long()       # [M]

        # Soft target for each position: look up row from sim matrix
        q = soft[valid_target]                         # [M, V]

        # CE = -sum(q * log_softmax(logits))
        log_probs = F.log_softmax(valid_input, dim=-1) # [M, V]
        ce = -(q * log_probs).sum(dim=-1)              # [M]

        if self.reduction == 'mean':
            return ce.mean()
        elif self.reduction == 'sum':
            return ce.sum()
        else:
            raise NotImplementedError
