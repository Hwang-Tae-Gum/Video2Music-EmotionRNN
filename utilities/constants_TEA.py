from utilities.constants import *

# ─────────────────────────────────────────────
#  TEA (Temporal Emotion Adapter) hyper-params
# ─────────────────────────────────────────────

# Where to inject the adapter
#   "encoder"  – Encoder 어댑터만
#   "decoder"  – Decoder 어댑터만
#   "both"     – 둘 다 + enc_dec_fusion
TEA_WHERE = "encoder"

# RNN cell type for each side
TEA_ENCODER_CELL = "gru"   # "lstm" | "gru"
TEA_DECODER_CELL = "gru"   # "lstm" | "gru"

# Temporal sliding-window params  (Ghaleb 2019)
TEA_WINDOW_SIZE = 2          # window size in seconds
TEA_OVERLAP     = 0.5        # overlap ratio

# Model dimensions (must match d_model in the base transformer)
TEA_HIDDEN_SIZE = 512        # d_model
TEA_NUM_LAYERS  = 2          # RNN stacked layers
TEA_NUM_HEADS   = 8          # cross-attention heads (decoder side)

# Emotion feature dimension
EMO_DIM = 6                  # exciting / fearful / tense / sad / relaxing / neutral

# ─────────────────────────────────────────────
#  Emotion-Valence Alignment Loss hyper-params
# ─────────────────────────────────────────────
LAMBDA_ALIGN = 0.1   # L_total += λ_align * MSE(soft_chord_valence, video_valence)

# ─────────────────────────────────────────────
#  Contrastive Emotion Loss hyper-params
# ─────────────────────────────────────────────
CONTRASTIVE_LAMBDA   = 0.1   # total loss weight: λ_con
CONTRASTIVE_PROJ_DIM = 128   # projection head output dimension
CONTRASTIVE_TEMP     = 0.07  # InfoNCE temperature τ