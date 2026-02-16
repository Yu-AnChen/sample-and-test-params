from __future__ import annotations

import csv
import math
import pathlib

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PARAM_DEFAULTS: dict[str, int | float | str] = {
    "channel": 0,
    "patch_size": 1024,
    "n_bins": 8,
    "n_patches": 8,
    "intensity_p0": 0.1,
    "intensity_p1": 99.95,
    "intensity_gamma": 1.0,
    "diameter": 15.0,
    "flow_threshold": 0.4,
    "dilation_radius": 3,
    "model_type": "cyto3",
    "restore_type": "deblur_cyto3",
    "model_backend": "denoise",
    "pretrained_model": "cpsam",
    # min_size intentionally omitted — auto-computed from diameter
}

_INT_FIELDS = {"channel", "patch_size", "n_bins", "n_patches", "dilation_radius"}
_FLOAT_FIELDS = {
    "intensity_p0",
    "intensity_p1",
    "intensity_gamma",
    "diameter",
    "flow_threshold",
    "min_size",
}


def load_slides(csv_path: str | pathlib.Path) -> list[dict]:
    csv_path = pathlib.Path(csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"name", "img_path", "out_dir"}
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return rows


def load_params(path: str | pathlib.Path) -> list[dict]:
    path = pathlib.Path(path)
    ext = path.suffix.lower()
    if ext == ".toml":
        return _load_params_toml(path)
    elif ext == ".csv":
        return _load_params_csv(path)
    else:
        raise ValueError(f"Unsupported params file extension: {ext} (use .toml or .csv)")


def _load_params_csv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    result = []
    for row in rows:
        params = dict(PARAM_DEFAULTS)
        params["name"] = row["name"]
        for key in row:
            if key == "name":
                continue
            val = row[key].strip() if row[key] else ""
            if not val:
                continue
            if key in _INT_FIELDS:
                params[key] = int(val)
            elif key in _FLOAT_FIELDS:
                params[key] = float(val)
            else:
                params[key] = val
        result.append(resolve_params(params))
    return result


def _load_params_toml(path: pathlib.Path) -> list[dict]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    defaults = {**PARAM_DEFAULTS, **data.get("defaults", {})}
    param_sets = data.get("param_sets", {})
    if not param_sets:
        raise ValueError(f"No [param_sets.*] entries found in {path}")
    result = []
    for name, overrides in param_sets.items():
        params = {**defaults, "name": name, **overrides}
        result.append(resolve_params(params))
    return result


def resolve_params(row: dict) -> dict:
    if "min_size" not in row or row["min_size"] is None:
        d = row["diameter"]
        row["min_size"] = 0.5 * math.pi * (d / 2) ** 2
    return row
