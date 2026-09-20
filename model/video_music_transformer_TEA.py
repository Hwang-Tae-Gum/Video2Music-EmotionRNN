"""
video_music_transformer_TEA.py
───────────────────────────────
VideoMusicTransformer 에 TEA(Temporal Emotion Adapter)를 주입한 모델.

TEA_WHERE 설정에 따라:
  "encoder" – Encoder 쪽 어댑터만
  "decoder" – Decoder 쪽 어댑터만 (cross-attention)
  "both"    – 둘 다 + enc_dec_fusion

기존 VideoMusicTransformer 의 forward / generate 시그니처를 그대로 유지하고
emotion_weight_override 파라미터만 추가한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
import random
import json

from utilities.constants import *
from utilities.constants_TEA import (
    TEA_WHERE, TEA_ENCODER_CELL, TEA_DECODER_CELL,
    TEA_NUM_LAYERS, TEA_HIDDEN_SIZE, TEA_NUM_HEADS, EMO_DIM,
    CONTRASTIVE_PROJ_DIM,
)
from utilities.device import get_device
from .positional_encoding import PositionalEncoding
from .rpr import TransformerDecoderRPR, TransformerDecoderLayerRPR
from .tea_adapter import TemporalEmotionAdapter


class VideoMusicTransformerTEA(nn.Module):

    def __init__(
        self,
        n_layers        = 6,
        num_heads       = 8,
        d_model         = 512,
        dim_feedforward = 1024,
        dropout         = 0.1,
        max_sequence_midi  = 2048,
        max_sequence_video = 300,
        max_sequence_chord = 300,
        total_vf_dim    = 0,
        rpr             = False,
        # TEA params
        tea_where        = TEA_WHERE,
        tea_num_layers   = TEA_NUM_LAYERS,
        use_multisignal  = False,   # True: scene/motion TEA 추가 (MS-TEA)
    ):
        super().__init__()

        # ── 기본 하이퍼파라미터 ──────────────────────────────────────────────
        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq_midi   = max_sequence_midi
        self.max_seq_video  = max_sequence_video
        self.max_seq_chord  = max_sequence_chord
        self.rpr             = rpr
        self.tea_where       = tea_where.lower()
        self.use_multisignal = use_multisignal

        # ── Input embeddings ────────────────────────────────────────────────
        self.embedding      = nn.Embedding(CHORD_SIZE, d_model)
        self.embedding_root = nn.Embedding(CHORD_ROOT_SIZE, d_model)
        self.embedding_attr = nn.Embedding(CHORD_ATTR_SIZE, d_model)

        self.total_vf_dim   = total_vf_dim
        self.Linear_vis     = nn.Linear(total_vf_dim, d_model)
        self.Linear_emo     = nn.Linear(6, d_model)   # dedicated emotion pathway
        self.Linear_chord   = nn.Linear(d_model + 1, d_model)

        # ── Positional encoding ─────────────────────────────────────────────
        self.positional_encoding       = PositionalEncoding(d_model, dropout, max_sequence_chord)
        self.positional_encoding_video = PositionalEncoding(d_model, dropout, max_sequence_video)

        # ── Transformer (base) ──────────────────────────────────────────────
        if not rpr:
            self.transformer = nn.Transformer(
                d_model         = d_model,
                nhead           = num_heads,
                num_encoder_layers = n_layers,
                num_decoder_layers = n_layers,
                dropout         = dropout,
                dim_feedforward = dim_feedforward,
            )
        else:
            decoder_norm  = LayerNorm(d_model)
            decoder_layer = TransformerDecoderLayerRPR(
                d_model, num_heads, dim_feedforward, dropout,
                er_len = max_sequence_chord,
            )
            decoder = TransformerDecoderRPR(decoder_layer, n_layers, decoder_norm)
            self.transformer = nn.Transformer(
                d_model         = d_model,
                nhead           = num_heads,
                num_encoder_layers = n_layers,
                num_decoder_layers = n_layers,
                dropout         = dropout,
                dim_feedforward = dim_feedforward,
                custom_decoder  = decoder,
            )

        # ── Output heads ────────────────────────────────────────────────────
        self.Wout      = nn.Linear(d_model, CHORD_SIZE)
        self.Wout_root = nn.Linear(d_model, CHORD_ROOT_SIZE)
        self.Wout_attr = nn.Linear(d_model, CHORD_ATTR_SIZE)
        self.softmax   = nn.Softmax(dim=-1)

        # ════════════════════════════════════════════════════════════════════
        #  TEA modules
        # ════════════════════════════════════════════════════════════════════

        # ── Encoder 어댑터 (양방향) ────────────────────────────────────────
        if self.tea_where in ("encoder", "both"):
            self.tea_encoder = TemporalEmotionAdapter(
                d_model    = d_model,
                input_dim  = EMO_DIM,
                direction  = "bi",
                cell_type  = TEA_ENCODER_CELL,
                num_layers = tea_num_layers,
            )
            self.encoder_emotion_proj = nn.Linear(d_model, d_model)
            self.encoder_emotion_norm = nn.LayerNorm(d_model)

            # ── Scene offset TEA + Motion TEA (MS-TEA 전용) ───────────────
            if use_multisignal:
                self.tea_scene = TemporalEmotionAdapter(
                    d_model    = d_model,
                    input_dim  = 1,
                    direction  = "bi",
                    cell_type  = TEA_ENCODER_CELL,
                    num_layers = tea_num_layers,
                )
                self.encoder_scene_proj = nn.Linear(d_model, d_model)
                self.encoder_scene_norm = nn.LayerNorm(d_model)

                self.tea_motion = TemporalEmotionAdapter(
                    d_model    = d_model,
                    input_dim  = 1,
                    direction  = "bi",
                    cell_type  = TEA_ENCODER_CELL,
                    num_layers = tea_num_layers,
                )
                self.encoder_motion_proj = nn.Linear(d_model, d_model)
                self.encoder_motion_norm = nn.LayerNorm(d_model)

        # ── Decoder 어댑터 (양방향) ────────────────────────────────────────
        # emotion 시퀀스는 비디오에서 사전 계산된 전체 sequence이므로
        # causal 제약 없음 → 양방향이 전체 감정 흐름을 더 잘 포착
        if self.tea_where in ("decoder", "both"):
            self.tea_decoder = TemporalEmotionAdapter(
                d_model    = d_model,
                input_dim  = EMO_DIM,
                direction  = "bi",
                cell_type  = TEA_DECODER_CELL,
                num_layers = tea_num_layers,
            )
            # Decoder: cross-attention (Q=transformer out, K/V=emotion emb)
            self.decoder_emotion_cross_attn = nn.MultiheadAttention(
                embed_dim   = d_model,
                num_heads   = TEA_NUM_HEADS,
                dropout     = dropout,
                batch_first = True,
            )
            self.decoder_emotion_norm = nn.LayerNorm(d_model)

            # 감정이 음악 생성을 방해하지 않도록 초기엔 작게 시작
            self.emotion_scale = nn.Parameter(torch.tensor(0.1))

        # ── enc → dec fusion  (both 일 때만) ──────────────────────────────
        if self.tea_where == "both":
            self.enc_dec_fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )
            self.fusion_scale = nn.Parameter(torch.tensor(0.1))

        # ── Emotion projection + classification head (학습 시에만 사용) ──────
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, CONTRASTIVE_PROJ_DIM),
        )
        self.emotion_clf = nn.Linear(CONTRASTIVE_PROJ_DIM, EMO_DIM)

    # ──────────────────────────────────────────────────────────────────────────
    #  Contrastive embedding (학습 전용)
    # ──────────────────────────────────────────────────────────────────────────

    def get_emotion_emb(
        self,
        feature_emotion:         torch.Tensor,         # (B, T, EMO_DIM)
        emotion_weight_override: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        TEA 어댑터 출력을 시간 축 mean-pool → projection → L2 정규화.
        반환: (B, CONTRASTIVE_PROJ_DIM) — SupConLoss 입력용.

        encoder 또는 both 설정이면 tea_encoder 사용;
        decoder only 면 tea_decoder 사용.
        """
        if self.tea_where in ("encoder", "both"):
            raw = self.tea_encoder(feature_emotion, emotion_weight_override)
        else:  # "decoder"
            raw = self.tea_decoder(feature_emotion, emotion_weight_override)

        pooled = raw.mean(dim=1)                            # (B, d_model)
        proj   = self.contrastive_proj(pooled)              # (B, proj_dim)
        return F.normalize(proj, dim=-1)

    # ──────────────────────────────────────────────────────────────────────────
    #  forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x, x_root, x_attr,
        feature_semantic_list,
        feature_key,
        feature_scene_offset,
        feature_motion,
        feature_emotion,
        mask                    = True,
        emotion_weight_override = None,   # (EMO_DIM,) or None
    ):
        # ── 기존 마스크 / 임베딩 처리 ────────────────────────────────────────
        if mask is True:
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None

        x_root = self.embedding_root(x_root)
        x_attr = self.embedding_attr(x_attr)
        x      = x_root + x_attr

        # feature_key: (B,) or (B,1) — expand to (B, T, 1)
        feature_key_padded = feature_key.float().view(x.shape[0], 1, 1).expand(x.shape[0], x.shape[1], 1).to(get_device())
        x  = torch.cat([x, feature_key_padded], dim=-1)
        xf = self.Linear_chord(x)

        # ── 영상 feature 연결 ─────────────────────────────────────────────
        vf_concat = feature_semantic_list[0].float()
        for i in range(1, len(feature_semantic_list)):
            vf_concat = torch.cat((vf_concat, feature_semantic_list[i].float()), dim=2)

        vf_concat = torch.cat([vf_concat, feature_scene_offset.unsqueeze(-1).float()], dim=-1)
        vf_concat = torch.cat([vf_concat, feature_motion.unsqueeze(-1).float()], dim=-1)
        vf        = self.Linear_vis(vf_concat) + self.Linear_emo(feature_emotion.float())  # (B, T_v, d_model)

        # ── TEA: Encoder 쪽 ───────────────────────────────────────────────
        enc_emotion_emb = None
        if self.tea_where in ("encoder", "both"):
            # Emotion TEA
            enc_emotion_emb = self.tea_encoder(
                feature_emotion, emotion_weight_override
            )                                            # (B, T_v, d_model)
            vf = self.encoder_emotion_norm(
                vf + self.encoder_emotion_proj(enc_emotion_emb)
            )

            # Scene / Motion TEA (MS-TEA 전용)
            if self.use_multisignal:
                scene_in = feature_scene_offset.unsqueeze(-1).float()
                scene_emb = self.tea_scene(scene_in)
                vf = self.encoder_scene_norm(vf + self.encoder_scene_proj(scene_emb))

                motion_in = feature_motion.unsqueeze(-1).float()
                motion_emb = self.tea_motion(motion_in)
                vf = self.encoder_motion_norm(vf + self.encoder_motion_proj(motion_emb))

        # ── Positional encoding & Transformer ────────────────────────────
        xf = xf.permute(1, 0, 2)   # (T_c, B, d_model)
        vf = vf.permute(1, 0, 2)   # (T_v, B, d_model)
        xf = self.positional_encoding(xf)
        vf = self.positional_encoding_video(vf)

        x_out = self.transformer(src=vf, tgt=xf, tgt_mask=mask)
        x_out = x_out.permute(1, 0, 2)   # (B, T_c, d_model)

        # ── TEA: Decoder 쪽 ───────────────────────────────────────────────
        if self.tea_where in ("decoder", "both"):
            dec_emotion_emb = self.tea_decoder(
                feature_emotion, emotion_weight_override
            )                                            # (B, T_v, d_model)

            # "both" 일 때: Encoder 감정 맥락을 Decoder 어댑터에 융합
            if self.tea_where == "both" and enc_emotion_emb is not None:
                # enc / dec 길이 동일 (T_v)
                fused = self.enc_dec_fusion(
                    torch.cat([enc_emotion_emb, dec_emotion_emb], dim=-1)
                )
                fs = self.fusion_scale.clamp(0.0, 1.0)
                dec_emotion_emb = dec_emotion_emb + fs * fused

            # x_out: (B, T_c, d_model)  /  dec_emotion_emb: (B, T_v, d_model)
            # T_c 와 T_v 가 다를 수 있으므로 T_c 에 맞게 자르거나 패딩
            T_c = x_out.shape[1]
            T_v = dec_emotion_emb.shape[1]
            if T_v >= T_c:
                dec_ctx = dec_emotion_emb[:, :T_c, :]
            else:
                pad = dec_emotion_emb[:, -1:, :].expand(-1, T_c - T_v, -1)
                dec_ctx = torch.cat([dec_emotion_emb, pad], dim=1)

            attn_out, _ = self.decoder_emotion_cross_attn(
                query = x_out,
                key   = dec_ctx,
                value = dec_ctx,
            )
            es    = self.emotion_scale.clamp(0.0, 1.0)
            x_out = self.decoder_emotion_norm(x_out + es * attn_out)

        # ── Output ───────────────────────────────────────────────────────
        del mask
        if IS_SEPERATED:
            return self.Wout_root(x_out), self.Wout_attr(x_out)
        else:
            return self.Wout(x_out)

    # ──────────────────────────────────────────────────────────────────────────
    #  generate
    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        feature_semantic_list   = [],
        feature_key             = None,
        feature_scene_offset    = None,
        feature_motion          = None,
        feature_emotion         = None,
        primer                  = None,
        primer_root             = None,
        primer_attr             = None,
        target_seq_length       = 300,
        beam                    = 0,
        beam_chance             = 1.0,
        max_conseq_N            = 0,
        max_conseq_chord        = 2,
        temperature             = 1.0,    # > 1 = 다양성 증가, < 1 = 확정적
        emotion_weight_override = None,   # (EMO_DIM,) or None  ← 사용자 제어
    ):
        assert not self.training, "Cannot generate while in training mode"
        print("Generating sequence of max length:", target_seq_length)

        with open("dataset/vevo_meta/chord_inv.json")  as f: chordInvDic  = json.load(f)
        with open("dataset/vevo_meta/chord_root.json") as f: chordRootDic = json.load(f)
        with open("dataset/vevo_meta/chord_attr.json") as f: chordAttrDic = json.load(f)

        gen_seq      = torch.full((1, target_seq_length), CHORD_PAD,      dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_root = torch.full((1, target_seq_length), CHORD_ROOT_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_attr = torch.full((1, target_seq_length), CHORD_ATTR_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer]      = primer.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_root[..., :num_primer] = primer_root.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_attr[..., :num_primer] = primer_attr.type(TORCH_LABEL_TYPE).to(get_device())

        cur_i = num_primer
        while cur_i < target_seq_length:
            y = self.softmax(
                self.forward(
                    gen_seq[..., :cur_i],
                    gen_seq_root[..., :cur_i],
                    gen_seq_attr[..., :cur_i],
                    feature_semantic_list,
                    feature_key,
                    feature_scene_offset,
                    feature_motion,
                    feature_emotion,
                    emotion_weight_override = emotion_weight_override,
                )
            )[..., :CHORD_END]

            token_probs = y[:, cur_i - 1, :]

            if beam == 0:
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0, 1)

            if beam_ran <= beam_chance:
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)
                beam_rows = top_i // CHORD_SIZE
                beam_cols = top_i %  CHORD_SIZE
                gen_seq   = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols
            else:
                if max_conseq_N == 0:
                    token_probs[0][0] = 0.0

                isMaxChord = False
                if cur_i >= max_conseq_chord:
                    preChord   = gen_seq[0][cur_i - 1].item()
                    isMaxChord = all(
                        preChord == gen_seq[0][cur_i - 1 - k].item()
                        for k in range(1, max_conseq_chord)
                    )

                if isMaxChord:
                    token_probs[0][gen_seq[0][cur_i - 1].item()] = 0.0

                if temperature != 1.0:
                    token_probs = token_probs ** (1.0 / temperature)
                    token_probs = token_probs / token_probs.sum()

                distrib    = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                gen_seq[:, cur_i] = next_token

                gen_chord = chordInvDic[str(next_token.item())]
                chord_arr = gen_chord.split(":")
                if len(chord_arr) == 1:
                    chordRootID = torch.tensor([chordRootDic[chord_arr[0]]]).to(get_device())
                    chordAttrID = torch.tensor([1]).to(get_device())
                elif len(chord_arr) == 2:
                    chordRootID = torch.tensor([chordRootDic[chord_arr[0]]]).to(get_device())
                    chordAttrID = torch.tensor([chordAttrDic[chord_arr[1]]]).to(get_device())

                gen_seq_root[:, cur_i] = chordRootID
                gen_seq_attr[:, cur_i] = chordAttrID

                if next_token == CHORD_END:
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break

            cur_i += 1
            if cur_i % 50 == 0:
                print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]