"""
Generate transformed images for the input CSV files, saving 3 outputs per image:
  1. preprocessed_original: "img_original" with only preprocessing_settings applied.
  2. augmented: "img" run through the full train pipeline (preprocessing ->
     data_augmentation -> post_augmentation) via a MONAI ``Dataset``/``DataLoader``.
  3. augmented_original_replayed: the *same* preprocessed original from (1), with the
     exact randomized augmentation operations that were actually sampled for "img" in
     (2) replayed onto it. (2) and (3) should end up identical; they are just produced
     via two different code paths (dataloader vs. manual replay).

Input CSV must contain a 'filename' column.
"""

import argparse
import copy
import json
import pathlib

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import monai.transforms as monai_transforms
from monai.transforms import LoadImaged
from monai.data import Dataset, DataLoader

from segment_tb_cxr.auxiliary.ensemble_nnunet_yolov8m import _read_image


class CustomLoadImaged(LoadImaged):
    """
    Custom image loader that reads a Chest X-Ray image into a channel-first
    numpy array of shape (1, H, W) with attached metadata.

    Keys in `image_keys` are read with `_read_image` (handles DICOM fallback
    and sRGB-to-grayscale conversion); any other key falls back to a plain
    `sitk.ReadImage`.
    """

    def __init__(
        self,
        keys,
        image_keys=("img", "img_original"),
        allow_missing_keys=False,
        **kwargs,
    ):
        super().__init__(
            keys, image_only=True, allow_missing_keys=allow_missing_keys, **kwargs
        )
        self.image_keys = set(image_keys)

    def _to_array(self, sitk_img):
        img_array = np.expand_dims(
            np.transpose(sitk.GetArrayFromImage(sitk_img), (1, 0)), axis=0
        ).astype(np.float32)
        meta_dict = {
            "spatial_shape": img_array.shape,
            "original_channel_dim": "no_channel",
        }
        return img_array, meta_dict

    def _load_img(self, path, key):
        sitk_img = (
            _read_image(str(path))
            if key in self.image_keys
            else sitk.ReadImage(str(path))
        )
        return self._to_array(sitk_img)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img, meta = self._load_img(d[key], key)
            d[key] = img
            d[key + "_meta_dict"] = meta
        return d


def _build_transform(cls_name, keys, kwargs):
    """Instantiate a MONAI (or locally-defined) dict transform from a config entry."""
    cls = getattr(monai_transforms, cls_name, None) or globals()[cls_name]
    kwargs = {
        k: (tuple(v) if k == "mode" and isinstance(v, list) else v)
        for k, v in (kwargs or {}).items()
    }
    return cls(keys=keys, allow_missing_keys=True, **kwargs)


def _kwargs_for_single_key(kwargs, keys, key):
    """Slice any per-key list-valued kwargs (e.g. 'mode') down to the value for `key`."""
    idx = keys.index(key)
    return {
        k: (v[idx] if k == "mode" and isinstance(v, list) else v)
        for k, v in (kwargs or {}).items()
    }


def _summarize_array(value):
    """Full arrays (e.g. RandGaussianNoise's sampled noise, same shape as the image) can't be
    printed element-by-element; summarize the actually-used values via shape/mean/std instead.
    """
    arr = value if isinstance(value, np.ndarray) else value.detach().cpu().numpy()
    if arr.size <= 8:
        return arr.tolist()
    return {
        "shape": tuple(arr.shape),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _sampled_values(transform, depth=2, _seen=None):
    """Recursively collect simple (numeric/tuple/bool) attribute values from a transform and
    its nested sub-transforms (many Rand*d transforms wrap an inner array-transform, e.g.
    RandRotated.rand_rotate, that actually holds the sampled angle/zoom/gamma/etc.)."""
    _seen = _seen if _seen is not None else set()
    if depth < 0 or id(transform) in _seen:
        return {}
    _seen.add(id(transform))

    values = {}
    for name, value in vars(transform).items():
        if name in (
            "R",
            "keys",
            "allow_missing_keys",
            "backend",
            "lazy",
            "_lazy",
            "prob",
            "_do_transform",
        ):
            continue
        if isinstance(value, (int, float, bool, str)):
            values[name] = value
        elif isinstance(value, (tuple, list)) and all(
            isinstance(x, (int, float, bool)) for x in value
        ):
            values[name] = value
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            values[name] = _summarize_array(value)
        elif hasattr(value, "__dict__"):
            for sub_name, sub_value in _sampled_values(value, depth - 1, _seen).items():
                values[f"{name}.{sub_name}"] = sub_value
    return values


def _apply_deterministic(entries, d, extra_keys=()):
    """Apply a list of (non-randomized) config transform entries to `d`, also targeting
    `extra_keys` wherever an entry already targets 'img' (e.g. to keep the replayed-original
    key in sync for type-casting transforms like EnsureTyped)."""
    for e in entries:
        keys = e.get("keys", ["img"])
        if "img" in keys:
            keys = list(keys) + [k for k in extra_keys if k not in keys]
        d = _build_transform(e["class"], keys, e.get("kwargs"))(d)
    return d


def transform_image(config, d):
    """
    Applies every randomized transform from data_augmentation_settings to "img" (and "seg")
    , each with a freshly drawn seed.

    Returns:
        d: with "img"/"seg" now actually augmented.
        recipe: ordered list of {"class", "kwargs", "seed", "fired", "sampled_values"} - exactly
            what was done to "img", so it can be reproduced elsewhere via apply_transform().
    """
    entries = (config.get("data_augmentation_settings") or {}).get("transforms") or []
    recipe = []

    for entry in entries:
        keys = entry.get("keys", ["img"])
        kwargs = entry.get("kwargs") or {}
        seed = int(np.random.randint(0, 2**31 - 1))

        transform = _build_transform(entry["class"], keys, kwargs)
        transform.set_random_state(seed=seed)
        d = transform(d)

        if "img" in keys:
            recipe.append(
                {
                    "class": entry["class"],
                    "kwargs": _kwargs_for_single_key(kwargs, keys, "img"),
                    "seed": seed,
                    "fired": getattr(transform, "_do_transform", True),
                    "sampled_values": _sampled_values(transform),
                }
            )

    return d, recipe


def apply_transform(recipe, d, key):
    """Re-applies the exact recipe recorded by transform_image() onto `key` in `d`,
    reproducing the identical sampled operations."""
    for step in recipe:
        cls = getattr(monai_transforms, step["class"], None) or globals()[step["class"]]
        d = cls(keys=[key], allow_missing_keys=True, **step["kwargs"]).set_random_state(
            seed=step["seed"]
        )(d)
    return d


def build_pipeline(config):
    """
    Returns a callable that takes {"img": file, "img_original": file} and produces a dict with:
      - "img_original": preprocessing only (output 1).
      - "img": full pipeline, i.e. what the dataloader actually produces (output 2).
      - "img_original_replayed": preprocessing + a replay of img's exact sampled
        augmentation ops (output 3).
    """
    preprocessing_entries = config["preprocessing_settings"]["transforms"]
    post_entries = config["post_augmentation_settings"]["transforms"]

    def _call(data):
        d = _apply_deterministic(preprocessing_entries, dict(data))
        d["img_original_replayed"] = copy.deepcopy(d["img_original"])

        d, recipe = transform_image(config, d)
        d = apply_transform(recipe, d, key="img_original_replayed")
        d = _apply_deterministic(
            post_entries, d, extra_keys=["img_original", "img_original_replayed"]
        )

        filename = pathlib.Path(data["img"]).name
        print(f"{filename}: applied augmentation transforms ->")
        for step in recipe:
            if step["fired"]:
                print(f"  {step['class']}: {step['sampled_values']}")

        return d

    return _call


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate transformations for the input CSV files."
    )
    parser.add_argument(
        "input_csv_path",
        type=pathlib.Path,
        help="CSV with 'filename' and 'ref_seg_file' columns",
    )
    parser.add_argument(
        "config_path",
        type=pathlib.Path,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="Output directory for the transformed images",
    )
    args = parser.parse_args(argv)

    with open(str(args.config_path)) as f:
        config_info = json.load(f)

    pipeline = build_pipeline(config_info["train"])

    df = pd.read_csv(args.input_csv_path)
    data = [
        {"img": row.filename, "img_original": row.filename, "seg": row.ref_seg_file}
        for row in df.itertuples()
    ]

    dataset = Dataset(data=data, transform=pipeline)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_suffixes = {
        "img_original": "preprocessed_original",
        "img": "augmented",
        "img_original_replayed": "augmented_original_replayed",
        "seg": "augmented_seg",
    }
    for batch, filename in zip(loader, df["filename"]):
        stem = pathlib.Path(filename).stem
        for key, suffix in output_suffixes.items():
            # CustomLoadImaged transposes (H, W) -> (W, H) on load; undo it here so

            array = np.transpose(np.asarray(batch[key][0, 0]), (1, 0))
            sitk.WriteImage(
                sitk.GetImageFromArray(array),
                str(args.output_dir / f"{stem}_{suffix}.nrrd"),
                useCompression=True,
            )


if __name__ == "__main__":
    main()
