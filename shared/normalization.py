"""Backbone-aware image normalization presets."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SATMAE_FMOW_RGB_MEAN = (0.4182007312774658, 0.4214799106121063, 0.3991275727748871)
SATMAE_FMOW_RGB_STD = (0.28774282336235046, 0.27541765570640564, 0.2764017581939697)


def get_normalization_coefs(backbone="dinov3", normalization="auto"):
    """Return {'mean': ..., 'std': ...} for a backbone and requested normalization preset."""
    preset = (normalization or "auto").lower()
    backbone = (backbone or "dinov3").lower()
    if preset == "auto":
        preset = "satmae_fmow_rgb" if backbone == "satmae" else "imagenet"
    if preset in ("imagenet", "dino", "dinov3", "resnet50"):
        return {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    if preset in ("satmae", "satmae_fmow_rgb", "fmow_rgb"):
        return {"mean": SATMAE_FMOW_RGB_MEAN, "std": SATMAE_FMOW_RGB_STD}
    raise ValueError("Unsupported normalization preset '{}'. Use 'auto', 'imagenet', or 'satmae_fmow_rgb'.".format(normalization))
