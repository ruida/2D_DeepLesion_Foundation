"""
lesion_encoder.py
=================
Three building blocks for injecting DeepLesion auxiliary information
(bounding boxes + anatomy concept IDs) into R2Gen-Mamba.

  BoundingBoxEncoder  – MLP: [cx,cy,w,h] → d_model token
  AnatomyEmbedder     – Embedding table: anatomy_id → d_model token
  LesionFusion        – Concatenates visual / bbox / anatomy token seqs
                        and returns the fused feature + attention mask.

Design notes
------------
* All three modules work in d_model space so their outputs can be
  directly concatenated with the already-projected visual tokens.
* A learnable padding/mask token handles missing bbox slots so the
  sequence length is always constant (easier batching).
* padding_idx=0 in AnatomyEmbedder matches the convention in
  datasets_deeplesion.py (0 = no anatomy label).
* LesionFusion also builds the matching boolean attention mask so the
  Mamba encoder knows which positions to ignore.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1.  Bounding-box encoder
# ---------------------------------------------------------------------------

class BoundingBoxEncoder(nn.Module):
    """
    Maps each normalised [cx, cy, w, h] box to a d_model-dimensional token.

    Padding slots (bbox_mask == False) are replaced by a learnable mask
    token so gradient can still flow and the encoder sees a constant-length
    sequence.

    Args
    ----
    d_model    : output / model dimension
    hidden_dim : intermediate size of the 2-layer MLP (default 128)
    """

    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, d_model),
        )
        # Learnable token used for empty (padded) box slots
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        bboxes: torch.Tensor,      # (B, max_boxes, 4)  float32, normalised
        bbox_mask: torch.Tensor,   # (B, max_boxes)      bool
    ) -> torch.Tensor:             # (B, max_boxes, d_model)
        tokens = self.mlp(bboxes)                              # (B, K, d_model)
        valid  = bbox_mask.unsqueeze(-1).float()               # (B, K, 1)
        tokens = tokens * valid + self.mask_token * (1.0 - valid)
        return tokens


# ---------------------------------------------------------------------------
# 2.  Anatomy / lesion concept embedder
# ---------------------------------------------------------------------------

class AnatomyEmbedder(nn.Module):
    """
    Standard embedding table for integer anatomy concept IDs.

    The DeepLesion IDs range from 2 to 172.  Index 0 is reserved as the
    padding index; it is always mapped to the zero vector.

    Args
    ----
    num_anatomy : vocabulary size including padding (use max_id + 2 = 174)
    d_model     : embedding dimension
    """

    def __init__(self, num_anatomy: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(num_anatomy, d_model, padding_idx=0)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight[0].zero_()   # guarantee padding row is 0

    def forward(self, anatomy_ids: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        anatomy_ids : (B, max_anatomy)  long, 0 = padding

        Returns
        -------
        tokens      : (B, max_anatomy, d_model)
        """
        return self.embed(anatomy_ids)



class AnatomyTextEmbedder(nn.Module):
    """
    Embedding table for anatomy-name text token IDs.

    These token IDs usually come from the report tokenizer. This lets the
    conditioning sequence use text-like inputs such as:
        "lesion lung"
        "lesion lymph node"

    Padding is still 0.
    """

    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight[pad_idx].zero_()

    def forward(self, anatomy_text_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(anatomy_text_ids)


# ---------------------------------------------------------------------------
# 3.  Fusion module
# ---------------------------------------------------------------------------

class LesionFusion(nn.Module):
    """
    Fuses projected visual tokens with bbox tokens and anatomy tokens by
    simple sequence concatenation.

    Input visual tokens are expected to have already been projected from
    d_vf → d_model by the model's att_embed layer BEFORE calling this
    module.

    Sequence layout after fusion:
        [ visual_tokens (S) | bbox_tokens (max_boxes) | anatomy_tokens (max_anatomy) ]

    A learnable scalar gate (initialised to zero → sigmoid = 0.5) blends
    the fused representation with the original visual-only representation
    for the auxiliary positions.  This lets the model start training in a
    regime close to the original R2Gen-Mamba and gradually opens the gate.

    Args
    ----
    d_model      : model hidden dimension (must match Mamba d_model = 512)
    num_anatomy  : anatomy vocab size (≥ max anatomy_id + 2)
    max_boxes    : number of bbox slots per sample
    max_anatomy  : number of anatomy slots per sample
    bbox_hidden  : hidden dim inside BoundingBoxEncoder MLP
    """

    def __init__(
        self,
        d_model:     int = 512,
        num_anatomy: int = 174,
        max_boxes:   int = 2,
        max_anatomy: int = 20,
        bbox_hidden: int = 128,
        anatomy_encoding: str = "id",
        anatomy_text_vocab_size: int = None,
    ):
        super().__init__()
        self.max_boxes   = max_boxes
        self.max_anatomy = max_anatomy
        self.anatomy_encoding = anatomy_encoding

        self.bbox_encoder = BoundingBoxEncoder(d_model, bbox_hidden)

        if anatomy_encoding == "id":
            self.anatomy_embedder = AnatomyEmbedder(num_anatomy, d_model)
        elif anatomy_encoding == "text":
            if anatomy_text_vocab_size is None:
                raise ValueError("anatomy_text_vocab_size is required when anatomy_encoding='text'")
            self.anatomy_embedder = AnatomyTextEmbedder(anatomy_text_vocab_size, d_model)
        else:
            raise ValueError(f"Unknown anatomy_encoding={anatomy_encoding}. Use 'id' or 'text'.")

        # Layer-norms keep scales compatible with visual features
        self.bbox_norm    = nn.LayerNorm(d_model)
        self.anatomy_norm = nn.LayerNorm(d_model)

        # Scalar gate – sigmoid(0) = 0.5; starts half-open
        self.gate = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def forward(
        self,
        visual_feats: torch.Tensor,    # (B, S, d_model) – projected visual
        bboxes:       torch.Tensor,    # (B, max_boxes, 4)
        bbox_mask:    torch.Tensor,    # (B, max_boxes)  bool
        anatomy_ids:  torch.Tensor,    # (B, max_anatomy) long; ID or text-token IDs
    ):
        """
        Returns
        -------
        fused_feats : (B, S + max_boxes + max_anatomy, d_model)
        fused_mask  : (B, S + max_boxes + max_anatomy)  long  1=valid 0=pad
        """
        B, S, D = visual_feats.shape

        # ---- encode auxiliary tokens ------------------------------------
        bbox_tok    = self.bbox_norm(
            self.bbox_encoder(bboxes, bbox_mask))           # (B, K, D)
        anatomy_tok = self.anatomy_norm(
            self.anatomy_embedder(anatomy_ids))             # (B, A, D)

        # ---- gated blend for auxiliary positions -----------------------
        g           = torch.sigmoid(self.gate)              # scalar
        bbox_tok    = g * bbox_tok                          # blend with 0-baseline
        anatomy_tok = g * anatomy_tok

        # ---- concatenate -----------------------------------------------
        fused_feats = torch.cat([visual_feats, bbox_tok, anatomy_tok], dim=1)
        # (B, S + K + A, D)

        # ---- attention mask  -------------------------------------------
        visual_mask  = visual_feats.new_ones(B, S,              dtype=torch.long)
        bbox_attn    = bbox_mask.long()                          # (B, K)
        anatomy_attn = (anatomy_ids > 0).long()                  # (B, A)
        fused_mask   = torch.cat([visual_mask, bbox_attn, anatomy_attn], dim=1)
        # (B, S + K + A)

        return fused_feats, fused_mask
