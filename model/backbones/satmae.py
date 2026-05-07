"""SatMAE RGB ViT-Large dense feature backbone wrapper for EPBA."""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_satmae_layers(layers: Union[str, Sequence[int], None]) -> List[int]:
    """Parse SatMAE block indices from CLI string or sequence."""
    if layers is None:
        return [5, 11, 17, 23]
    if isinstance(layers, str):
        parsed = [int(item.strip()) for item in layers.split(",") if item.strip()]
    else:
        parsed = [int(item) for item in layers]
    if not parsed:
        raise ValueError("satmae_layers must contain at least one block index.")
    return parsed


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_satmae_models():
    satmae_root = _repo_root() / "third_party" / "SatMAE"
    if not satmae_root.is_dir():
        raise FileNotFoundError(f"SatMAE source directory not found: {satmae_root}")
    satmae_root_str = str(satmae_root)
    if satmae_root_str not in sys.path:
        sys.path.insert(0, satmae_root_str)
    try:
        import models_vit  # type: ignore
    except ModuleNotFoundError as e:
        if e.name == "timm":
            raise ModuleNotFoundError(
                "SatMAE requires the 'timm' package. Install timm or use another backbone."
            ) from e
        raise
    return models_vit


def _strip_prefix(key: str, prefixes: Iterable[str]) -> str:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _select_checkpoint_state(checkpoint, ckpt_key: Optional[str] = None):
    if ckpt_key:
        if not isinstance(checkpoint, dict) or ckpt_key not in checkpoint:
            raise KeyError(f"SatMAE checkpoint key '{ckpt_key}' was not found.")
        return checkpoint[ckpt_key]
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _is_state_dict(value) -> bool:
    return isinstance(value, dict) and all(isinstance(k, str) for k in value.keys())


def _infer_grid(num_patch_tokens: int, fallback_hw: Optional[Sequence[int]] = None) -> Sequence[int]:
    if fallback_hw is not None and fallback_hw[0] * fallback_hw[1] == num_patch_tokens:
        return int(fallback_hw[0]), int(fallback_hw[1])
    side = int(math.sqrt(num_patch_tokens))
    if side * side != num_patch_tokens:
        raise ValueError(
            f"Cannot infer a 2D grid from {num_patch_tokens} position tokens. "
            "Provide an RGB non-temporal vanilla ViT checkpoint with square position embeddings."
        )
    return side, side


def interpolate_pos_embed(pos_embed: torch.Tensor, target_num_tokens: int, target_hw: Sequence[int]) -> torch.Tensor:
    """Interpolate cls+patch position embeddings to target patch grid."""
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ValueError(f"Expected pos_embed shape [1, N, C], got {tuple(pos_embed.shape)}")
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if pos_embed.shape[1] == target_num_tokens:
        return pos_embed

    try:
        _infer_grid(pos_embed.shape[1] - 1)
        has_cls = True
    except ValueError:
        has_cls = False
    if has_cls:
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
    else:
        cls_pos = pos_embed.new_zeros((1, 1, pos_embed.shape[-1]))
        patch_pos = pos_embed

    src_h, src_w = _infer_grid(patch_pos.shape[1])
    patch_pos = patch_pos.reshape(1, src_h, src_w, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(target_h, target_w), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_h * target_w, -1)
    return torch.cat([cls_pos, patch_pos], dim=1)


class SatMAEBackboneWrapper(nn.Module):
    """RGB SatMAE ViT-Large Patch16 wrapper that returns 1/16 dense features."""

    _ignored_prefixes = (
        "head.", "decoder_", "fc_norm.", "mask_token", "norm_pix_loss",
        "channel_embed", "patch_embed.0", "patch_embed.1",
    )

    def __init__(
        self,
        weight_path: Optional[str] = None,
        layers: Union[str, Sequence[int], None] = None,
        img_size: Union[int, Sequence[int]] = 512,
        patch_size: int = 16,
        model_name: str = "vit_large_patch16",
        ckpt_key: Optional[str] = None,
        apply_norm: bool = True,
        verbose: int = 1,
    ):
        super().__init__()
        if patch_size != 16:
            raise ValueError("SatMAEBackboneWrapper currently supports only patch_size=16.")
        if model_name != "vit_large_patch16":
            raise ValueError("SatMAEBackboneWrapper currently supports only model_name='vit_large_patch16'.")

        self.layers = parse_satmae_layers(layers)
        self.patch_size = patch_size
        self.apply_norm = apply_norm
        self.verbose = verbose

        models_vit = _import_satmae_models()
        self.model = models_vit.vit_large_patch16(img_size=img_size, patch_size=patch_size, in_chans=3, num_classes=1000)
        self._relax_patch_embed_size_checks()

        self.embed_dim = int(self.model.embed_dim)
        self.num_blocks = len(self.model.blocks)
        invalid = [idx for idx in self.layers if idx < 0 or idx >= self.num_blocks]
        if invalid:
            raise ValueError(f"Invalid SatMAE layer indices {invalid}; valid range is [0, {self.num_blocks - 1}].")
        self.output_dim = self.embed_dim * len(self.layers)

        if weight_path:
            self.load_checkpoint(weight_path, ckpt_key=ckpt_key)
        elif self.verbose:
            warnings.warn("SatMAE weight_path is not set; using random initialization for smoke tests only.")

    def _relax_patch_embed_size_checks(self) -> None:
        patch_embed = self.model.patch_embed
        if hasattr(patch_embed, "strict_img_size"):
            patch_embed.strict_img_size = False
        if hasattr(patch_embed, "dynamic_img_pad"):
            patch_embed.dynamic_img_pad = False

    @property
    def pos_embed(self) -> torch.Tensor:
        return self.model.pos_embed

    def _runtime_pos_embed(self, grid_hw: Sequence[int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        target_tokens = 1 + int(grid_hw[0]) * int(grid_hw[1])
        pos_embed = interpolate_pos_embed(self.model.pos_embed.to(device=device, dtype=dtype), target_tokens, grid_hw)
        return pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"SatMAEBackboneWrapper expects RGB input [B, 3, H, W], got {tuple(x.shape)}")
        b, _, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"SatMAE input H/W must be divisible by {self.patch_size}, got H={h}, W={w}.")
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        tokens = self.model.patch_embed(x)
        cls_tokens = self.model.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self._runtime_pos_embed((grid_h, grid_w), tokens.dtype, tokens.device)
        tokens = self.model.pos_drop(tokens)

        outputs = []
        wanted = set(self.layers)
        for idx, block in enumerate(self.model.blocks):
            tokens = block(tokens)
            if idx in wanted:
                patch_tokens = tokens[:, 1:, :]
                if self.apply_norm and hasattr(self.model, "norm"):
                    patch_tokens = self.model.norm(patch_tokens)
                feat = patch_tokens.reshape(b, grid_h, grid_w, self.embed_dim).permute(0, 3, 1, 2).contiguous()
                outputs.append(feat)

        if len(outputs) != len(self.layers):
            raise RuntimeError(f"Collected {len(outputs)} SatMAE layers, expected {len(self.layers)}.")
        return torch.cat(outputs, dim=1)

    def _looks_like_non_rgb_checkpoint(self, state_dict) -> bool:
        keys = set(state_dict.keys())
        if any(k.startswith("channel_embed") for k in keys):
            return True
        if any(k.startswith("patch_embed.0") or k.startswith("patch_embed.1") for k in keys):
            return True
        proj = state_dict.get("patch_embed.proj.weight")
        return isinstance(proj, torch.Tensor) and proj.ndim == 4 and proj.shape[1] != 3

    def _clean_state_dict(self, state_dict):
        if not _is_state_dict(state_dict):
            raise ValueError("SatMAE checkpoint payload is not a PyTorch state_dict.")
        cleaned = {}
        for key, value in state_dict.items():
            key = _strip_prefix(key, ("module.", "model.", "backbone.", "encoder."))
            if any(key.startswith(prefix) for prefix in self._ignored_prefixes):
                continue
            if not isinstance(value, torch.Tensor):
                continue
            cleaned[key] = value
        return cleaned

    def load_checkpoint(self, weight_path: str, ckpt_key: Optional[str] = None) -> None:
        checkpoint = torch.load(weight_path, map_location="cpu")
        raw_state = _select_checkpoint_state(checkpoint, ckpt_key=ckpt_key)
        if not _is_state_dict(raw_state):
            raise ValueError("SatMAE checkpoint payload is not a PyTorch state_dict.")
        raw_for_detection = {
            _strip_prefix(k, ("module.", "model.", "backbone.", "encoder.")): v
            for k, v in raw_state.items()
            if isinstance(k, str)
        }
        if self._looks_like_non_rgb_checkpoint(raw_for_detection):
            raise ValueError(
                "The SatMAE checkpoint appears to be group-channel/Sentinel/multispectral rather than RGB vanilla ViT. "
                "This EPBA wrapper supports only RGB non-temporal vit_large_patch16 checkpoints."
            )
        state = self._clean_state_dict(raw_state)

        model_state = self.model.state_dict()
        filtered = {}
        skipped = []
        target_hw = self.model.patch_embed.grid_size
        if isinstance(target_hw, int):
            target_hw = (target_hw, target_hw)

        for key, value in state.items():
            if key not in model_state:
                continue
            if key == "pos_embed" and value.shape != model_state[key].shape:
                try:
                    value = interpolate_pos_embed(value, model_state[key].shape[1], target_hw)
                    if self.verbose:
                        print(f"[SatMAE] Interpolated checkpoint pos_embed to {tuple(value.shape)}.")
                except Exception as exc:
                    skipped.append((key, tuple(value.shape), tuple(model_state[key].shape), str(exc)))
                    continue
            if value.shape != model_state[key].shape:
                skipped.append((key, tuple(value.shape), tuple(model_state[key].shape), "shape mismatch"))
                continue
            filtered[key] = value

        incompatible = self.model.load_state_dict(filtered, strict=False)
        if self.verbose:
            print(f"[SatMAE] Loaded {len(filtered)} tensors from checkpoint: {weight_path}")
            if skipped:
                print("[SatMAE] Skipped checkpoint tensors with incompatible shapes:")
                for key, src_shape, dst_shape, reason in skipped:
                    print(f"  - {key}: checkpoint {src_shape} vs model {dst_shape} ({reason})")
            print(f"[SatMAE] Missing keys: {list(incompatible.missing_keys)}")
            print(f"[SatMAE] Unexpected keys: {list(incompatible.unexpected_keys)}")
