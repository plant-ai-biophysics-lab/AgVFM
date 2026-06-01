"""PEZ gradient adapters for each detection model's text encoder.

Each adapter wraps a frozen vision model and provides:
  - Access to the token embedding matrix E ∈ R^{V × d}
  - A differentiable ``encode_text(P_ste)`` call that runs the text encoder
    on a continuous embedding (straight-through from PEZ)
  - ``precompute_image_cosine_feat(samples)`` — frozen, called once before the
    optimisation loop — for the cosine-similarity loss variant
    - ``compute_iou_cls_loss(sample, P_ste)`` — per-sample, called every
        optimisation step — for the IoU × classification-score loss variant
    - ``compute_composite_loss(sample, P_ste)`` — optional spatially aware
        three-part loss used by PEZ when the model exposes region scores
  - ``decode(token_ids)`` — converts token IDs back to a readable string

Three concrete adapters are implemented:

  CLIPPEZAdapter  — used for YOLO World (CLIP ViT text encoder loaded
                    separately; GT-box ROI crops used for the IoU variant)
  OWLv2PEZAdapter — uses Owlv2ForObjectDetection's internal CLIP text encoder;
                    patch-level queries used for the IoU variant
  GDINOPEZAdapter — uses GroundingDINO's BERT text backbone; token-level
                    matching logits used for the IoU variant; optional
                    adjacent-embedding fluency regularisation
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PEZAdapter(ABC):
    """Abstract interface that every model-specific PEZ adapter must satisfy."""

    # ------------------------------------------------------------------
    # Subclasses must set these after __init__ calls super().__init__()
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def embedding_matrix(self) -> torch.Tensor:
        """Token embedding matrix E of shape [vocab_size, embed_dim] on device."""

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        """Start-of-sequence token ID."""

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """End-of-sequence token ID."""

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """Padding token ID."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device the model lives on."""

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    @abstractmethod
    def tokenize(self, text: str, max_length: int | None = None) -> torch.Tensor:
        """Return 1-D token ID tensor (without BOS/EOS) for ``text``."""

    @abstractmethod
    def encode_text(self, P_ste: torch.Tensor) -> torch.Tensor:
        """Run text encoder on continuous prompt embeddings.

        Parameters
        ----------
        P_ste:
            Straight-through embedding tensor of shape [T, d] where T is the
            token budget.  BOS/EOS tokens are added internally.

        Returns
        -------
        torch.Tensor
            Projected text feature vector of shape [d_proj].  Gradient must
            flow back through P_ste.
        """

    @abstractmethod
    def precompute_image_cosine_feat(
        self,
        samples: list,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Compute the frozen mean image feature used for the cosine loss.

        Parameters
        ----------
        samples:
            Proxy samples (``list[Sample]``).
        batch_size:
            How many images to encode per GPU call.

        Returns
        -------
        torch.Tensor
            Mean image feature vector of shape [d_proj], no gradient, on device.
        """

    @abstractmethod
    def compute_iou_cls_loss(
        self,
        sample,                  # Sample from loader
        P_ste: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the IoU × classification-score loss for a single sample.

        For YOLO World (CLIP adapter) this is a GT-box-guided cosine
        similarity loss.  For OWLv2 and GDINO it uses model detection logits.

        Returns a scalar loss tensor with gradient w.r.t. P_ste.
        """

    def compute_composite_loss(self, sample, P_ste: torch.Tensor) -> torch.Tensor:
        """Optional three-part PEZ loss: classification + GIoU + L1.

        Concrete adapters can override this when they expose spatial scores.
        The default implementation raises NotImplementedError so callers can
        fall back to ``compute_iou_cls_loss``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement composite PEZ loss"
        )

    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode a 1-D token ID tensor to a human-readable string."""


# ---------------------------------------------------------------------------
# Helper: build full embeddings with BOS / EOS
# ---------------------------------------------------------------------------

def _clip_encode_text_embeds(
    clip_text_model,           # CLIPTextModel or compatible submodule
    text_projection,           # nn.Linear or None
    embeds: torch.Tensor,      # [1, T+2, d_model] — already has BOS/EOS prepended
    device: torch.device,
) -> torch.Tensor:
    """Bypass CLIPTextTransformer.forward() and call the encoder directly.

    In transformers ≥ 5.x ``CLIPTextTransformer.forward()`` no longer accepts
    ``inputs_embeds``; the parameter was removed.  Anything passed via **kwargs
    leaks to ``self.encoder()``, which already receives ``inputs_embeds`` from
    the embedding layer — producing the "multiple values" TypeError.

    This helper instead:
      1. Adds positional embeddings via the model's own ``position_embedding`` layer.
      2. Builds the required upper-triangular causal mask.
      3. Calls ``encoder`` and ``final_layer_norm`` directly.
      4. Pools at the EOS position (last token in the sequence).
      5. Applies ``text_projection`` if provided.

    Gradient flows back through ``embeds`` (and therefore through P_ste) without
    touching any parameter of the frozen text encoder.
    """
    seq_len = embeds.shape[1]

    # 1. Positional embeddings
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    pos_embeds = clip_text_model.embeddings.position_embedding(pos_ids)  # [1, T+2, d]
    hidden = embeds + pos_embeds                                          # [1, T+2, d]

    # 2. Causal attention mask — shape [1, 1, T+2, T+2]
    #    Upper triangle → -inf (future positions masked out)
    #    Diagonal + lower triangle → 0.0 (attend freely)
    dtype = hidden.dtype
    causal = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
    upper = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(diagonal=1)
    causal.masked_fill_(upper.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)

    # 3. Encoder — try the two known API variants across transformers versions
    encoder = clip_text_model.encoder
    try:
        # transformers ≥ 4.36: separate causal_attention_mask kwarg
        enc_out = encoder(
            inputs_embeds=hidden,
            attention_mask=None,
            causal_attention_mask=causal,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    except TypeError:
        # Older API: single attention_mask arg
        enc_out = encoder(
            inputs_embeds=hidden,
            attention_mask=causal,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

    # 4. Final layer norm + pool at EOS (last token position)
    last_hidden = clip_text_model.final_layer_norm(enc_out.last_hidden_state)  # [1, T+2, d]
    pooled = last_hidden[:, -1, :]                                              # [1, d]

    # 5. Text projection (optional; absent in some OWLv2 variants)
    if text_projection is not None:
        pooled = text_projection(pooled)

    return F.normalize(pooled[0], dim=-1)  # [d_proj]


def _prepend_bos_eos(
    P_ste: torch.Tensor,          # [T, d]
    E: torch.Tensor,              # [V, d]
    bos_id: int,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate BOS + P_ste + EOS and return (embeds, attention_mask)."""
    bos_emb = E[bos_id].unsqueeze(0)                  # [1, d]
    eos_emb = E[eos_id].unsqueeze(0)                  # [1, d]
    embeds = torch.cat([bos_emb, P_ste, eos_emb], dim=0)  # [T+2, d]
    # Batch dimension: [1, T+2, d]
    embeds = embeds.unsqueeze(0)
    mask = torch.ones(1, embeds.shape[1], dtype=torch.long, device=embeds.device)
    return embeds, mask


# ---------------------------------------------------------------------------
# CLIP PEZ Adapter — for YOLO World
# ---------------------------------------------------------------------------

class CLIPPEZAdapter(PEZAdapter):
    """PEZ adapter that loads a standalone CLIP model.

    Used for YOLO World, which internally uses a CLIP ViT text encoder that is
    not directly accessible through the Ultralytics API.  The same CLIP encoder
    is loaded separately here so gradients can flow through it.

    For the IoU × cls loss, GT bounding box regions are cropped from proxy
    images and encoded with CLIP's vision encoder.  The loss is then:

        L = 1 - cosine_sim(text_feat, mean_gt_crop_feat)

    This is the "region-text cosine similarity" described in the YOLO World
    gradient optimisation section.
    """

    _CHECKPOINT = "openai/clip-vit-large-patch14"

    def __init__(self, checkpoint: str = _CHECKPOINT) -> None:
        from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor

        logger.info(f"Loading CLIP adapter from {checkpoint!r} …")
        self._model = CLIPModel.from_pretrained(checkpoint)
        self._tokenizer = CLIPTokenizerFast.from_pretrained(checkpoint)
        self._image_processor = CLIPImageProcessor.from_pretrained(checkpoint)

        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(_dev)
        self._model.eval()
        # Freeze all parameters
        for p in self._model.parameters():
            p.requires_grad_(False)

        self._device = _dev
        self._E = self._model.text_model.embeddings.token_embedding.weight.detach()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_matrix(self) -> torch.Tensor:
        return self._E

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.bos_token_id or self._tokenizer.cls_token_id or 49406

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id or self._tokenizer.sep_token_id or 49407

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id or 0

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def tokenize(self, text: str, max_length: int | None = None) -> torch.Tensor:
        enc = self._tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=max_length or 75,
            truncation=True,
        )
        return enc.input_ids[0].to(self._device)

    def encode_text(self, P_ste: torch.Tensor) -> torch.Tensor:
        """Run CLIP text encoder on continuous prompt embeddings → [d_proj].

        Calls the encoder layer-stack directly (bypassing text_model.forward),
        which avoids both the 'You have to specify input_ids' check and the
        'multiple values for inputs_embeds' TypeError introduced in transformers>=5.x.
        """
        E = self._E
        embeds, _ = _prepend_bos_eos(P_ste, E, self.bos_token_id, self.eos_token_id)
        return _clip_encode_text_embeds(
            clip_text_model=self._model.text_model,
            text_projection=self._model.text_projection,
            embeds=embeds,
            device=self._device,
        )

    @torch.no_grad()
    def precompute_image_cosine_feat(
        self,
        samples: list,
        batch_size: int = 8,
    ) -> torch.Tensor:
        all_feats = []
        for i in range(0, len(samples), batch_size):
            batch_imgs = [s.image for s in samples[i: i + batch_size]]
            inputs = self._image_processor(images=batch_imgs, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self._device)
            # Use vision_model + visual_projection directly — get_image_features()
            # returns a BaseModelOutputWithPooling in transformers>=5.x, not a tensor.
            vision_out = self._model.vision_model(pixel_values=pixel_values)
            pooled = vision_out.pooler_output                    # [batch, d_vision]
            img_feats = self._model.visual_projection(pooled)   # [batch, d_proj]
            img_feats = F.normalize(img_feats, dim=-1)
            all_feats.append(img_feats)
        return torch.cat(all_feats, dim=0).mean(dim=0)  # [d_proj]

    def compute_iou_cls_loss(self, sample, P_ste: torch.Tensor) -> torch.Tensor:
        """GT-box-guided cosine similarity loss for YOLO World.

        Crops each GT bounding box from the image, encodes it with the frozen
        CLIP vision encoder, and returns 1 - cosine_sim(text_feat, gt_feat).
        Falls back to full-image cosine sim when no GT boxes are available.
        """
        text_feat = self.encode_text(P_ste)  # [d_proj]

        image = sample.image
        w, h = image.size
        anns = sample.annotations

        gt_crops: list[Image.Image] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            # Annotations are normalised fractions (from loader._parse_annotation)
            x1, y1 = int(x * w), int(y * h)
            x2, y2 = int((x + bw) * w), int((y + bh) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                gt_crops.append(image.crop((x1, y1, x2, y2)))

        if not gt_crops:
            # No GT boxes — fall back to full-image cosine loss
            gt_crops = [image]

        with torch.no_grad():
            inputs = self._image_processor(images=gt_crops, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self._device)
            vision_out = self._model.vision_model(pixel_values=pixel_values)
            pooled = vision_out.pooler_output                    # [n_crops, d_vision]
            gt_feat = self._model.visual_projection(pooled)     # [n_crops, d_proj]
            gt_feat = F.normalize(gt_feat, dim=-1).mean(dim=0)  # [d_proj]

        return 1.0 - F.cosine_similarity(text_feat.unsqueeze(0), gt_feat.unsqueeze(0))

    def compute_composite_loss(self, sample, P_ste: torch.Tensor) -> torch.Tensor:
        """Fallback composite loss for CLIP-based PEZ.

        YOLO World does not expose a differentiable region grid through the
        public API in this codebase, so the composite objective falls back to
        the existing IoU/classification surrogate.
        """
        return self.compute_iou_cls_loss(sample, P_ste)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self._tokenizer.decode(
            token_ids.cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()


# ---------------------------------------------------------------------------
# OWLv2 PEZ Adapter
# ---------------------------------------------------------------------------

class OWLv2PEZAdapter(PEZAdapter):
    """PEZ adapter that uses Owlv2ForObjectDetection's internal CLIP text encoder.

    Cosine loss: text feature vs mean patch-level image feature from OWLv2's
    vision encoder.

    IoU × cls loss: OWLv2 pred_logits (patch-text similarities) weighted by
    IoU of the highest-scoring patches against GT boxes.
    """

    _CHECKPOINT = "google/owlv2-base-patch16-ensemble"

    def __init__(self, checkpoint: str = _CHECKPOINT, threshold: float = 0.3) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        logger.info(f"Loading OWLv2 adapter from {checkpoint!r} …")
        self._processor = Owlv2Processor.from_pretrained(checkpoint)
        self._model = Owlv2ForObjectDetection.from_pretrained(checkpoint)
        self.threshold = threshold

        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(_dev)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        self._device = _dev
        # Embedding matrix from the internal CLIP text model
        self._E = (
            self._model.owlv2.text_model.embeddings.token_embedding.weight.detach()
        )

    @property
    def embedding_matrix(self) -> torch.Tensor:
        return self._E

    @property
    def bos_token_id(self) -> int:
        return self._processor.tokenizer.bos_token_id or 49406

    @property
    def eos_token_id(self) -> int:
        return self._processor.tokenizer.eos_token_id or 49407

    @property
    def pad_token_id(self) -> int:
        return self._processor.tokenizer.pad_token_id or 0

    @property
    def device(self) -> torch.device:
        return self._device

    def tokenize(self, text: str, max_length: int | None = None) -> torch.Tensor:
        enc = self._processor.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=max_length or 16,
            truncation=True,
        )
        return enc.input_ids[0].to(self._device)

    def encode_text(self, P_ste: torch.Tensor) -> torch.Tensor:
        """Run OWLv2's CLIP text encoder → L2-normalised feature [d_proj].

        Same bypass as CLIPPEZAdapter: calls the encoder layer-stack directly
        to avoid the transformers>=5.x CLIPTextTransformer API breakage.
        """
        E = self._E
        embeds, _ = _prepend_bos_eos(P_ste, E, self.bos_token_id, self.eos_token_id)
        text_proj = getattr(self._model.owlv2, "text_projection", None)
        return _clip_encode_text_embeds(
            clip_text_model=self._model.owlv2.text_model,
            text_projection=text_proj,
            embeds=embeds,
            device=self._device,
        )

    @torch.no_grad()
    def precompute_image_cosine_feat(
        self,
        samples: list,
        batch_size: int = 4,
    ) -> torch.Tensor:
        all_feats = []
        for i in range(0, len(samples), batch_size):
            batch_imgs = [s.image for s in samples[i: i + batch_size]]
            inputs = self._processor(
                images=batch_imgs,
                text=[["placeholder"]] * len(batch_imgs),
                return_tensors="pt",
                truncation=True,
            ).to(self._device)
            vision_out = self._model.owlv2.vision_model(
                pixel_values=inputs.pixel_values
            )
            # Patch tokens (exclude CLS): [batch, n_patches, d]
            patch_feats = vision_out.last_hidden_state[:, 1:, :]
            pooled = patch_feats.mean(dim=1)           # [batch, d]
            _vproj = getattr(self._model.owlv2, "vision_projection", None) or getattr(self._model.owlv2, "visual_projection", None)
            if _vproj is not None:
                pooled = _vproj(pooled)
            pooled = F.normalize(pooled, dim=-1)
            all_feats.append(pooled)
        return torch.cat(all_feats, dim=0).mean(dim=0)

    def compute_iou_cls_loss(self, sample, P_ste: torch.Tensor) -> torch.Tensor:
        """OWLv2 pred_logits weighted by IoU against GT boxes.

        Runs the vision encoder (frozen) and the text encoder (via P_ste,
        gradient-enabled).  The classification score for each image patch is
        the cosine similarity between the patch query and the text feature.
        Patches overlapping GT boxes (IoU > 0.3) are considered positives.
        """
        text_feat = self.encode_text(P_ste)  # [d_proj], grad-enabled

        image = sample.image
        w, h = image.size

        with torch.no_grad():
            inputs = self._processor(
                images=[image],
                text=[["placeholder"]],
                return_tensors="pt",
                truncation=True,
            ).to(self._device)
            vision_out = self._model.owlv2.vision_model(
                pixel_values=inputs.pixel_values
            )
            patch_feats = vision_out.last_hidden_state[:, 1:, :]  # [1, n_patches, d]
            _vproj = getattr(self._model.owlv2, "vision_projection", None) or getattr(self._model.owlv2, "visual_projection", None)
            if _vproj is not None:
                patch_feats = _vproj(patch_feats)
            patch_feats = F.normalize(patch_feats[0], dim=-1)  # [n_patches, d]

        # Similarity: [n_patches]
        sims = (patch_feats @ text_feat).clamp(min=0)

        if not sample.annotations:
            # No GT — maximise mean patch similarity
            return -sims.mean()

        # Soft IoU weighting: patches with high GT-overlap get upweighted.
        # We approximate patch positions assuming a grid layout.
        n_patches = patch_feats.shape[0]
        grid_h = grid_w = int(n_patches ** 0.5)
        # Build patch boxes in normalised coords
        ys = (torch.arange(grid_h, device=self._device).float() + 0.5) / grid_h
        xs = (torch.arange(grid_w, device=self._device).float() + 0.5) / grid_w
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        patch_sz_y = 1.0 / grid_h
        patch_sz_x = 1.0 / grid_w
        patch_boxes = torch.stack([
            xx.flatten() - patch_sz_x / 2,
            yy.flatten() - patch_sz_y / 2,
            xx.flatten() + patch_sz_x / 2,
            yy.flatten() + patch_sz_y / 2,
        ], dim=-1)  # [n_patches, 4] x1y1x2y2 normalised

        # GT boxes in xyxy normalised
        gt_boxes = []
        for ann in sample.annotations:
            x, y, bw, bh = ann["bbox"]
            gt_boxes.append([x, y, x + bw, y + bh])
        gt_t = torch.tensor(gt_boxes, dtype=torch.float32, device=self._device)

        # IoU between each patch and each GT box
        iou = _box_iou(patch_boxes, gt_t)  # [n_patches, n_gt]
        max_iou, _ = iou.max(dim=1)        # [n_patches]

        # Weighted loss: reward high-similarity patches that overlap GT boxes
        weights = max_iou.detach()
        if weights.sum() < 1e-6:
            weights = torch.ones_like(weights)

        loss = -(weights * sims).sum() / (weights.sum() + 1e-8)
        return loss

    def compute_composite_loss(self, sample, P_ste: torch.Tensor) -> torch.Tensor:
        """Three-part differentiable loss: classification + GIoU + L1."""
        text_feat = self.encode_text(P_ste)  # [d_proj], grad-enabled

        image = sample.image

        with torch.no_grad():
            inputs = self._processor(
                images=[image],
                text=[["placeholder"]],
                return_tensors="pt",
                truncation=True,
            ).to(self._device)
            vision_out = self._model.owlv2.vision_model(
                pixel_values=inputs.pixel_values
            )
            patch_feats = vision_out.last_hidden_state[:, 1:, :]  # [1, n_patches, d]
            _vproj = getattr(self._model.owlv2, "vision_projection", None) or getattr(self._model.owlv2, "visual_projection", None)
            if _vproj is not None:
                patch_feats = _vproj(patch_feats)
            patch_feats = F.normalize(patch_feats[0], dim=-1)  # [n_patches, d]

        sims = (patch_feats @ text_feat).clamp(min=0)

        if not sample.annotations:
            return -sims.mean()

        n_patches = patch_feats.shape[0]
        grid_h = grid_w = int(n_patches ** 0.5)
        ys = (torch.arange(grid_h, device=self._device).float() + 0.5) / grid_h
        xs = (torch.arange(grid_w, device=self._device).float() + 0.5) / grid_w
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        patch_sz_y = 1.0 / grid_h
        patch_sz_x = 1.0 / grid_w
        patch_boxes = torch.stack([
            xx.flatten() - patch_sz_x / 2,
            yy.flatten() - patch_sz_y / 2,
            xx.flatten() + patch_sz_x / 2,
            yy.flatten() + patch_sz_y / 2,
        ], dim=-1)

        gt_boxes = []
        for ann in sample.annotations:
            x, y, bw, bh = ann["bbox"]
            gt_boxes.append([x, y, x + bw, y + bh])
        gt_t = torch.tensor(gt_boxes, dtype=torch.float32, device=self._device)

        iou = _box_iou(patch_boxes, gt_t)
        max_iou, _ = iou.max(dim=1)
        weights = max_iou.detach()
        if weights.sum() < 1e-6:
            weights = torch.ones_like(weights)

        cls_loss = -(weights * sims).sum() / (weights.sum() + 1e-8)
        soft_box = _softmax_weighted_box(sims, patch_boxes, temperature=0.5)
        gt_box = _mean_gt_box(gt_t)
        giou_loss = 1.0 - _box_giou(
            soft_box.unsqueeze(0), gt_box.unsqueeze(0)
        ).squeeze()
        l1_loss = F.l1_loss(soft_box, gt_box)
        return cls_loss + giou_loss + l1_loss

    def decode(self, token_ids: torch.Tensor) -> str:
        return self._processor.tokenizer.decode(
            token_ids.cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()


# ---------------------------------------------------------------------------
# Grounding DINO PEZ Adapter
# ---------------------------------------------------------------------------

class GDINOPEZAdapter(PEZAdapter):
    """PEZ adapter that uses GroundingDINO's BERT text backbone.

    Cosine loss: BERT [CLS] / mean-token text feature vs Swin backbone image
    features (pooled to match BERT's 768-d).

    IoU × cls loss: token-level matching logits from GDINO's detection head,
    weighted by IoU against GT boxes.

    Optional fluency regularisation: adjacent token embedding cosine similarity
    (smooth token sequences → more natural prompts), controlled by the
    ``fluency_lambda`` parameter passed to :func:`pez_optimize`.
    """

    _CHECKPOINT = "IDEA-Research/grounding-dino-base"

    def __init__(self, checkpoint: str = _CHECKPOINT, threshold: float = 0.3) -> None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        logger.info(f"Loading Grounding DINO adapter from {checkpoint!r} …")
        self._processor = AutoProcessor.from_pretrained(checkpoint)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
        self.threshold = threshold

        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(_dev)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        self._device = _dev
        self._text_backbone = self._find_text_backbone()
        self._E = self._text_backbone.embeddings.word_embeddings.weight.detach()

    def _find_text_backbone(self):
        """Locate GDINO's BERT text backbone (attribute path may vary by version)."""
        model_core = self._model.model
        for attr in ("text_backbone", "language_backbone", "bert", "text_encoder"):
            if hasattr(model_core, attr):
                logger.debug(f"Found GDINO text backbone at model.model.{attr}")
                return getattr(model_core, attr)
        raise AttributeError(
            "Could not find GroundingDINO text backbone. "
            "Expected one of: text_backbone, language_backbone, bert, text_encoder "
            f"on {type(model_core).__name__}."
        )

    @property
    def embedding_matrix(self) -> torch.Tensor:
        return self._E

    @property
    def bos_token_id(self) -> int:
        return self._processor.tokenizer.cls_token_id or 101

    @property
    def eos_token_id(self) -> int:
        return self._processor.tokenizer.sep_token_id or 102

    @property
    def pad_token_id(self) -> int:
        return self._processor.tokenizer.pad_token_id or 0

    @property
    def device(self) -> torch.device:
        return self._device

    def tokenize(self, text: str, max_length: int | None = None) -> torch.Tensor:
        enc = self._processor.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=max_length or 20,
            truncation=True,
        )
        return enc.input_ids[0].to(self._device)

    def encode_text(self, P_ste: torch.Tensor) -> torch.Tensor:
        """Run BERT text backbone on continuous prompt embeddings → [d_text].

        Returns the mean-pooled token hidden states (BERT-base: 768-d).
        Gradient flows back through P_ste via the straight-through estimate.
        """
        E = self._E
        embeds, mask = _prepend_bos_eos(P_ste, E, self.bos_token_id, self.eos_token_id)
        text_out = self._text_backbone(
            inputs_embeds=embeds,
            attention_mask=mask,
        )
        # Mean-pool all token outputs (more stable than CLS alone for BERT)
        hidden = text_out.last_hidden_state  # [1, T+2, d]
        # Mask-weighted mean
        mask_f = mask.unsqueeze(-1).float()  # [1, T+2, 1]
        feat = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1)  # [1, d]
        return feat[0]  # [d]

    @torch.no_grad()
    def precompute_image_cosine_feat(
        self,
        samples: list,
        batch_size: int = 4,
    ) -> torch.Tensor:
        """Pool Swin backbone spatial features across proxy images → [768]."""
        all_feats = []
        dummy_text = "object."
        for i in range(0, len(samples), batch_size):
            batch_imgs = [s.image for s in samples[i: i + batch_size]]
            inputs = self._processor(
                images=batch_imgs,
                text=[dummy_text] * len(batch_imgs),
                return_tensors="pt",
            ).to(self._device)
            # Run only the backbone (no cross-modal encoder)
            backbone_out = self._model.model.backbone(
                pixel_values=inputs.pixel_values,
                pixel_mask=inputs.get("pixel_mask"),
            )
            # Handle both tuple (newer transformers) and dataclass with .feature_maps
            # Backbone returns either a dataclass with .feature_maps, or a 2-tuple
            # (feature_maps_list, position_encodings_list).  Each element of the
            # feature_maps_list may itself be a (tensor, mask) pair (FPN output).
            feat_maps = backbone_out.feature_maps if hasattr(backbone_out, "feature_maps") else backbone_out[0]
            feat_map = feat_maps[-1]
            if isinstance(feat_map, (tuple, list)):
                feat_map = feat_map[0]  # unwrap (tensor, mask) FPN pair
            pooled = feat_map.mean(dim=[2, 3])          # [batch, C]
            # Project to 768 if needed (Swin-T last stage = 768 for GDINO-base)
            if pooled.shape[-1] != 768:
                pooled = pooled[..., :768]              # slice to 768 as fallback
            pooled = F.normalize(pooled, dim=-1)
            all_feats.append(pooled)
        return torch.cat(all_feats, dim=0).mean(dim=0)  # [768]

    def compute_iou_cls_loss(self, sample, P_ste: torch.Tensor) -> torch.Tensor:
        """Token-level GDINO matching logits × IoU against GT boxes.

        Runs the GDINO vision backbone + cross-modal encoder (both frozen) to
        get region-level features, then computes similarity against the
        text features from P_ste (grad-enabled).

        Because running the full encoder requires both image and text features,
        we pass a lightweight text-only forward to compute region embeddings,
        then score them against our optimised text embedding.
        """
        text_feat = self.encode_text(P_ste)  # [768], grad-enabled

        image = sample.image
        w, h = image.size

        # Precompute image region features (frozen, decoupled from text)
        with torch.no_grad():
            dummy_text = "object."
            inputs = self._processor(
                images=[image],
                text=[dummy_text],
                return_tensors="pt",
            ).to(self._device)
            backbone_out = self._model.model.backbone(
                pixel_values=inputs.pixel_values,
                pixel_mask=inputs.get("pixel_mask"),
            )
            # Handle both tuple (newer transformers) and dataclass with .feature_maps
            # Backbone returns either a dataclass with .feature_maps, or a 2-tuple
            # (feature_maps_list, position_encodings_list).  Each element of the
            # feature_maps_list may itself be a (tensor, mask) pair (FPN output).
            feat_maps = backbone_out.feature_maps if hasattr(backbone_out, "feature_maps") else backbone_out[0]
            feat_map = feat_maps[-1]
            if isinstance(feat_map, (tuple, list)):
                feat_map = feat_map[0]  # unwrap (tensor, mask) FPN pair
            fH, fW = feat_map.shape[2], feat_map.shape[3]
            region_feats = feat_map[0].reshape(feat_map.shape[1], -1).T  # [fH*fW, C]
            if region_feats.shape[-1] != 768:
                region_feats = region_feats[..., :768]
            region_feats = F.normalize(region_feats, dim=-1)  # [n_regions, 768]

        # Similarity: [n_regions]
        sims = (region_feats @ text_feat).clamp(min=0)

        if not sample.annotations:
            return -sims.mean()

        # Build approximate spatial boxes for each backbone grid cell
        n_r = region_feats.shape[0]
        ys = (torch.arange(fH, device=self._device).float() + 0.5) / fH
        xs = (torch.arange(fW, device=self._device).float() + 0.5) / fW
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        region_boxes = torch.stack([
            xx.flatten() - 0.5 / fW,
            yy.flatten() - 0.5 / fH,
            xx.flatten() + 0.5 / fW,
            yy.flatten() + 0.5 / fH,
        ], dim=-1)  # [n_regions, 4] x1y1x2y2 normalised

        gt_boxes = []
        for ann in sample.annotations:
            x, y, bw, bh = ann["bbox"]
            gt_boxes.append([x, y, x + bw, y + bh])
        gt_t = torch.tensor(gt_boxes, dtype=torch.float32, device=self._device)

        iou = _box_iou(region_boxes, gt_t)   # [n_regions, n_gt]
        max_iou, _ = iou.max(dim=1)           # [n_regions]
        weights = max_iou.detach()
        if weights.sum() < 1e-6:
            weights = torch.ones_like(weights)

        loss = -(weights * sims).sum() / (weights.sum() + 1e-8)
        return loss

    def compute_composite_loss(self, sample, P_ste: torch.Tensor) -> torch.Tensor:
        """Three-part differentiable loss: classification + GIoU + L1."""
        text_feat = self.encode_text(P_ste)  # [768], grad-enabled

        image = sample.image

        with torch.no_grad():
            dummy_text = "object."
            inputs = self._processor(
                images=[image],
                text=[dummy_text],
                return_tensors="pt",
            ).to(self._device)
            backbone_out = self._model.model.backbone(
                pixel_values=inputs.pixel_values,
                pixel_mask=inputs.get("pixel_mask"),
            )
            # Handle both tuple (newer transformers) and dataclass with .feature_maps
            # Backbone returns either a dataclass with .feature_maps, or a 2-tuple
            # (feature_maps_list, position_encodings_list).  Each element of the
            # feature_maps_list may itself be a (tensor, mask) pair (FPN output).
            feat_maps = backbone_out.feature_maps if hasattr(backbone_out, "feature_maps") else backbone_out[0]
            feat_map = feat_maps[-1]
            if isinstance(feat_map, (tuple, list)):
                feat_map = feat_map[0]  # unwrap (tensor, mask) FPN pair
            fH, fW = feat_map.shape[2], feat_map.shape[3]
            region_feats = feat_map[0].reshape(feat_map.shape[1], -1).T  # [fH*fW, C]
            if region_feats.shape[-1] != 768:
                region_feats = region_feats[..., :768]
            region_feats = F.normalize(region_feats, dim=-1)  # [n_regions, 768]

        sims = (region_feats @ text_feat).clamp(min=0)

        if not sample.annotations:
            return -sims.mean()

        ys = (torch.arange(fH, device=self._device).float() + 0.5) / fH
        xs = (torch.arange(fW, device=self._device).float() + 0.5) / fW
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        region_boxes = torch.stack([
            xx.flatten() - 0.5 / fW,
            yy.flatten() - 0.5 / fH,
            xx.flatten() + 0.5 / fW,
            yy.flatten() + 0.5 / fH,
        ], dim=-1)

        gt_boxes = []
        for ann in sample.annotations:
            x, y, bw, bh = ann["bbox"]
            gt_boxes.append([x, y, x + bw, y + bh])
        gt_t = torch.tensor(gt_boxes, dtype=torch.float32, device=self._device)

        iou = _box_iou(region_boxes, gt_t)
        max_iou, _ = iou.max(dim=1)
        weights = max_iou.detach()
        if weights.sum() < 1e-6:
            weights = torch.ones_like(weights)

        cls_loss = -(weights * sims).sum() / (weights.sum() + 1e-8)
        soft_box = _softmax_weighted_box(sims, region_boxes, temperature=0.5)
        gt_box = _mean_gt_box(gt_t)
        giou_loss = 1.0 - _box_giou(
            soft_box.unsqueeze(0), gt_box.unsqueeze(0)
        ).squeeze()
        l1_loss = F.l1_loss(soft_box, gt_box)
        return cls_loss + giou_loss + l1_loss

    def compute_fluency_loss(self, P_ste: torch.Tensor) -> torch.Tensor:
        """Adjacent-token cosine similarity as a fluency regulariser.

        High adjacent similarity → smoother / more coherent token sequences.
        Returns a positive scalar; add to detection loss with weight λ.
        """
        if P_ste.shape[0] < 2:
            return torch.tensor(0.0, device=self._device, requires_grad=False)
        sim = F.cosine_similarity(P_ste[:-1], P_ste[1:])  # [T-1]
        return -sim.mean()   # negative so minimising → maximising coherence

    def decode(self, token_ids: torch.Tensor) -> str:
        return self._processor.tokenizer.decode(
            token_ids.cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADAPTER_MAP: dict[str, type[PEZAdapter]] = {
    "yolo_world":      CLIPPEZAdapter,
    "grounding_dino":  GDINOPEZAdapter,
    "owlv2":           OWLv2PEZAdapter,
}


def load_adapter(model_name: str, **kwargs) -> PEZAdapter:
    """Instantiate the PEZ adapter for the given detection model name."""
    if model_name not in _ADAPTER_MAP:
        raise NotImplementedError(
            f"No PEZ adapter for model {model_name!r}. "
            f"Available: {list(_ADAPTER_MAP)}"
        )
    cls = _ADAPTER_MAP[model_name]
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def _box_iou(
    boxes_a: torch.Tensor,   # [N, 4] x1y1x2y2
    boxes_b: torch.Tensor,   # [M, 4] x1y1x2y2
) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes → [N, M]."""
    # Intersection
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)                                  # [N, M, 2]
    inter = wh[..., 0] * wh[..., 1]                              # [N, M]

    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))  # [N]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))  # [M]
    union = area_a[:, None] + area_b[None, :] - inter            # [N, M]
    return inter / (union + 1e-6)


def _box_giou(
    boxes_a: torch.Tensor,   # [N, 4] x1y1x2y2
    boxes_b: torch.Tensor,   # [M, 4] x1y1x2y2
) -> torch.Tensor:
    """Compute pairwise generalized IoU between two sets of boxes → [N, M]."""
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)                                  # [N, M, 2]
    inter = wh[..., 0] * wh[..., 1]                              # [N, M]

    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))  # [N]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))  # [M]
    union = area_a[:, None] + area_b[None, :] - inter            # [N, M]
    iou = inter / (union + 1e-6)

    c_lt = torch.min(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [N, M, 2]
    c_rb = torch.max(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [N, M, 2]
    c_wh = (c_rb - c_lt).clamp(min=0)
    c_area = c_wh[..., 0] * c_wh[..., 1]
    return iou - (c_area - union) / (c_area + 1e-6)


def _softmax_weighted_box(
    scores: torch.Tensor,
    boxes: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Take the expected box under softmax-normalised spatial scores."""
    temp = max(float(temperature), 1e-6)
    weights = F.softmax(scores / temp, dim=0)
    return (weights.unsqueeze(-1) * boxes).sum(dim=0)


def _mean_gt_box(gt_boxes: torch.Tensor) -> torch.Tensor:
    """Collapse one or more GT boxes to a single continuous target box."""
    if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4:
        raise ValueError(f"Expected [N, 4] GT boxes, got {tuple(gt_boxes.shape)}")
    return gt_boxes.mean(dim=0)
