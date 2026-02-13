from __future__ import annotations

import pathlib
import re

import numpy as np
import skimage.exposure
import skimage.util
import tifffile

from .sampler import WsiPatchSampler
from .segment import (
    adjust_intensity,
    difference_mask_from_file,
    dilate_slide_mask,
    make_qc_image,
    mask_to_contour,
    segment_slide,
    segment_tile,
)


def sample_and_test(
    img_path: str | pathlib.Path,
    name: str,
    out_dir: str | pathlib.Path,
    param_sets: list[dict],
) -> list[pathlib.Path]:
    out_dir = pathlib.Path(out_dir) / name / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group param sets by sampling params to reuse extracted patches
    sampling_groups: dict[tuple, list[dict]] = {}
    for ps in param_sets:
        key = (
            ps.get("channel", 0),
            ps.get("patch_size", 1024),
            ps.get("n_bins", 8),
            ps.get("n_patches", 8),
        )
        sampling_groups.setdefault(key, []).append(ps)

    output_paths = []

    for (channel, patch_size, n_bins, n_patches), group in sampling_groups.items():
        sampler = WsiPatchSampler(
            img_path=str(img_path),
            channel=channel,
            n_bins=n_bins,
            patch_size=patch_size,
            n_patches=n_patches,
        )
        patches = sampler.extract_patches(channel=channel)

        for ps in group:
            intensity_p0 = ps.get("intensity_p0", 0.1)
            intensity_p1 = ps.get("intensity_p1", 99.95)
            intensity_gamma = ps.get("intensity_gamma", 1.0)
            diameter = ps.get("diameter", 15.0)
            flow_threshold = ps.get("flow_threshold", 0.4)
            min_size = ps.get("min_size")

            # Compute intensity range from all patches
            all_pixels = np.concatenate([p.ravel() for p in patches])
            p0, p1 = np.percentile(all_pixels, [intensity_p0, intensity_p1])
            in_range = (p0, p1)

            seg_kwargs = dict(
                diameter=diameter,
                flow_threshold=flow_threshold,
            )
            if min_size is not None:
                seg_kwargs["min_size"] = min_size

            pairs = []
            for patch in patches:
                adjusted = adjust_intensity(
                    patch.copy(), intensity_in_range=in_range, intensity_gamma=intensity_gamma
                )
                mask = segment_tile(adjusted, **seg_kwargs)
                contour = mask_to_contour(mask.astype("int32"))

                # Normalize raw patch for display
                raw_display = skimage.exposure.rescale_intensity(
                    adjusted, out_range="float"
                )

                # Create overlay: raw patch with contour in white
                overlay = raw_display.copy()
                overlay[contour] = 1.0

                # Side-by-side: raw | overlay
                pair = np.concatenate([raw_display, overlay], axis=1)
                pairs.append(pair)

            # Arrange into montage: rows = bins, cols = patches
            grid_shape = (n_bins, n_patches)
            montage = skimage.util.montage(pairs, grid_shape=grid_shape)

            out_path = out_dir / f"{ps['name']}.tif"
            tifffile.imwrite(
                out_path,
                (montage * 255).clip(0, 255).astype("uint8"),
                compression="zlib",
            )
            print(f"  saved: {out_path}")
            output_paths.append(out_path)

    return output_paths


def run_slide(
    img_path: str | pathlib.Path,
    name: str,
    out_dir: str | pathlib.Path,
    params: dict,
) -> dict[str, pathlib.Path]:
    img_path = pathlib.Path(img_path)
    _out_dir = pathlib.Path(out_dir)
    img_name = re.sub(r"\.ome\.tif{1,2}$", "", img_path.name, flags=re.IGNORECASE)
    seg_dir = _out_dir / name / "segmentation" / img_name
    seg_dir.mkdir(parents=True, exist_ok=True)

    seg_kwargs = {}
    if params.get("min_size") is not None:
        seg_kwargs["min_size"] = params["min_size"]

    mask_path = segment_slide(
        img_path,
        channel=params.get("channel", 0),
        out_dir=seg_dir,
        intensity_p0=params.get("intensity_p0", 0.1),
        intensity_p1=params.get("intensity_p1", 99.95),
        intensity_gamma=params.get("intensity_gamma", 1.0),
        diameter=params.get("diameter", 15.0),
        flow_threshold=params.get("flow_threshold", 0.4),
        **seg_kwargs,
    )

    qc_path = make_qc_image(
        img_path,
        mask_path,
        seg_dir,
        channel=params.get("channel", 0),
        intensity_p0=params.get("intensity_p0", 0.1),
        intensity_p1=params.get("intensity_p1", 99.95),
    )

    cell_mask_path = dilate_slide_mask(
        mask_path,
        radius=params.get("dilation_radius", 3),
        out_dir=seg_dir,
    )

    difference_mask_from_file(cell_mask_path, mask_path, seg_dir)

    return {
        "nucleus": mask_path,
        "qc": qc_path,
        "cell": cell_mask_path,
    }
