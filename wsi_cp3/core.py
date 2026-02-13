from __future__ import annotations

import pathlib
import re

import dask.array as da
import numpy as np
import skimage.util
import tifffile

from .sampler import WsiPatchSampler
from .segment import (
    adjust_intensity,
    da_to_zarr,
    difference_mask_from_file,
    dilate_slide_mask,
    make_qc_image,
    mask_to_contour,
    percentile_intensity,
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

        # Montage patches into a single 2D image (n_bins rows × n_patches cols)
        grid_shape = (n_bins, n_patches)
        montage_img = skimage.util.montage(list(patches), grid_shape=grid_shape)

        for ps in group:
            intensity_p0 = ps.get("intensity_p0", 0.1)
            intensity_p1 = ps.get("intensity_p1", 99.95)
            intensity_gamma = ps.get("intensity_gamma", 1.0)
            diameter = ps.get("diameter", 15.0)
            flow_threshold = ps.get("flow_threshold", 0.4)
            min_size = ps.get("min_size")

            # Compute intensity range from the whole slide
            p0, p1 = percentile_intensity(
                sampler.reader.pyramid[0][channel], [intensity_p0, intensity_p1]
            )
            in_range = (p0, p1)
            print(f"  {ps['name']} intensity range: {np.round(in_range, decimals=2)}")

            seg_kwargs = dict(
                diameter=diameter,
                flow_threshold=flow_threshold,
            )
            if min_size is not None:
                seg_kwargs["min_size"] = min_size

            # Convert montage to dask array chunked by patch_size
            da_montage = da.from_array(montage_img, chunks=patch_size)

            # Adjust intensity via map_blocks (same as segment_slide)
            da_adjusted = da_montage.map_blocks(
                adjust_intensity,
                intensity_in_range=in_range,
                intensity_gamma=intensity_gamma,
                dtype="float32",
            )
            za_adjusted = da_to_zarr(da_adjusted)

            # Segment via map_overlap (same as segment_slide)
            da_adjusted_zarr = da.from_zarr(za_adjusted, name=False)
            da_mask = da_adjusted_zarr.map_overlap(
                segment_tile,
                depth={0: 128, 1: 128},
                boundary="none",
                dtype=bool,
                **seg_kwargs,
            )
            print(f"  run cellpose; number of chunks: {da_mask.numblocks}")
            za_mask = da_to_zarr(da_mask, num_workers=2)

            # Create QC overlay: contours on adjusted montage
            da_binary = da.from_zarr(za_mask, name=False)
            contour = da_binary.astype("int32").map_blocks(
                mask_to_contour, dtype=bool
            )

            adjusted_np = np.array(za_adjusted)
            contour_np = np.array(da_to_zarr(contour))

            overlay = adjusted_np.copy()
            overlay[contour_np > 0] = 1.0

            # Side-by-side: adjusted | overlay
            qc_img = np.concatenate([adjusted_np, overlay], axis=1)

            out_path = out_dir / f"{ps['name']}.tif"
            tifffile.imwrite(
                out_path,
                (qc_img * 255).clip(0, 255).astype("uint8"),
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
