from __future__ import annotations

import argparse
from importlib.metadata import version

from .config import load_params, load_slides
from .core import run_slide, sample_and_test


def main():
    parser = argparse.ArgumentParser(
        prog="wsi-cp3",
        description="WSI sampling & Cellpose segmentation tool",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('wsi-cp3')}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- test subcommand ---
    test_parser = subparsers.add_parser(
        "test", help="Sample patches and generate segmentation montages"
    )
    test_parser.add_argument("--slides", required=True, help="Path to slides CSV")
    test_parser.add_argument("--params", required=True, help="Path to params file (CSV or TOML)")
    test_parser.add_argument("--slide-name", help="Process only this slide name")

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run", help="Run full-slide segmentation"
    )
    run_parser.add_argument("--slides", required=True, help="Path to slides CSV")
    run_parser.add_argument("--params", required=True, help="Path to params file (CSV or TOML)")
    run_parser.add_argument("--param-set", required=True, help="Name of the parameter set to use")
    run_parser.add_argument("--slide-name", help="Process only this slide name")

    args = parser.parse_args()

    slides = load_slides(args.slides)
    param_sets = load_params(args.params)

    if args.slide_name:
        slides = [s for s in slides if s["name"] == args.slide_name]
        if not slides:
            parser.error(f"Slide name '{args.slide_name}' not found in slides CSV")

    if args.command == "test":
        for slide in slides:
            print(f"\nTesting slide: {slide['name']}")
            sample_and_test(
                img_path=slide["img_path"],
                name=slide["name"],
                out_dir=slide["out_dir"],
                param_sets=param_sets,
            )

    elif args.command == "run":
        matched = [ps for ps in param_sets if ps["name"] == args.param_set]
        if not matched:
            available = [ps["name"] for ps in param_sets]
            parser.error(
                f"Parameter set '{args.param_set}' not found. "
                f"Available: {available}"
            )
        params = matched[0]
        for slide in slides:
            print(f"\nProcessing slide: {slide['name']}")
            results = run_slide(
                img_path=slide["img_path"],
                name=slide["name"],
                out_dir=slide["out_dir"],
                params=params,
            )
            print(f"  Outputs: {results}")


if __name__ == "__main__":
    main()
