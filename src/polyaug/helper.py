"""CLI argument parsing and runtime configuration helpers for PolyAug."""

import argparse
from importlib import metadata
from pathlib import Path
from typing import Sequence


def _get_installed_version() -> str:
    # Resolve installed package version for `polyaug --version`.
    try:
        return metadata.version("polyaug")
    except metadata.PackageNotFoundError:
        # Fallback when running from source without an installed distribution.
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    # Create the root CLI parser for the polyaug command.
    parser = argparse.ArgumentParser(
        prog="polyaug",
        description="Run PolyAug polygon augmentation on LabelMe image/json datasets.",
    )
    # Support quick CLI version inspection without requiring data arguments.
    parser.add_argument("--version", action="version", version=f"%(prog)s {_get_installed_version()}")

    # Required dataset input directories.
    parser.add_argument("--img", "--img-dir", dest="img_dir", required=True, help="Directory containing source images.")
    parser.add_argument("--json", "--json-dir", dest="json_dir", required=True, help="Directory containing source LabelMe JSON files.")

    # Output directory configuration.
    parser.add_argument(
        "--save",
        "--save-dir",
        dest="save_dir",
        default="./aug_result",
        help="Output root directory. Defaults to ./aug_result",
    )
    parser.add_argument(
        "--index-json-dir",
        default="",
        help="Optional indexed JSON output directory. Disabled by default.",
    )

    # Augmentation count control.
    parser.add_argument("--num-per-image", type=int, default=2, help="Number of augmentations per source image.")

    # Numeric range controls for geometric and photometric transforms.
    parser.add_argument("--crop", nargs=2, type=float, metavar=("MIN", "MAX"), default=[0.8, 0.9])
    parser.add_argument("--rotation", nargs=2, type=float, metavar=("MIN", "MAX"), default=[-30.0, 30.0])
    parser.add_argument("--scale", nargs=2, type=float, metavar=("MIN", "MAX"), default=[0.7, 1.3])
    parser.add_argument("--translate", nargs=2, type=float, metavar=("MIN", "MAX"), default=[-0.1, 0.1])
    parser.add_argument("--brightness", nargs=2, type=float, metavar=("MIN", "MAX"), default=[-0.1, 0.1])
    parser.add_argument("--contrast", nargs=2, type=float, metavar=("MIN", "MAX"), default=[-0.1, 0.1])

    # Per-transform application probabilities.
    parser.add_argument("--p-rotate", type=float, default=0.9)
    parser.add_argument("--p-flip-h", type=float, default=0.2)
    parser.add_argument("--p-flip-v", type=float, default=0.1)
    parser.add_argument("--p-affine", type=float, default=0.8)
    parser.add_argument("--p-crop", type=float, default=0.7)
    parser.add_argument("--p-brightness", type=float, default=0.4)

    # Polygon/mask repair and filtering thresholds.
    parser.add_argument("--contour-simplify-epsilon", type=float, default=1.5)
    parser.add_argument("--min-component-area", type=float, default=12.0)
    parser.add_argument("--min-mask-pixel-area", type=float, default=12.0)
    parser.add_argument("--random-aug-per-image", type=int, default=3)
    parser.add_argument("--debug", action="store_true", help="Enable debug logs in the augmentor.")

    return parser


def _validate_range(name: str, values: Sequence[float]) -> tuple[float, float]:
    # Normalize a [min, max] input pair and enforce ordering.
    low = float(values[0])
    high = float(values[1])
    # Reject inverted ranges early so downstream transforms stay valid.
    if low > high:
        raise ValueError(f"{name}: minimum cannot be greater than maximum ({low} > {high})")
    # Return an immutable pair used by augmentation parameter mapping.
    return (low, high)


def build_runtime_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    try:
        # Validate every numeric range argument before execution.
        crop = _validate_range("crop", args.crop)
        rotation = _validate_range("rotation", args.rotation)
        scale = _validate_range("scale", args.scale)
        translate = _validate_range("translate", args.translate)
        brightness = _validate_range("brightness", args.brightness)
        contrast = _validate_range("contrast", args.contrast)
    except ValueError as err:
        # Surface range validation failures through argparse's standard error path.
        parser.error(str(err))

    # Convert CLI path strings to Path objects for safe path composition.
    img_dir = Path(args.img_dir)
    json_dir = Path(args.json_dir)
    save_root = Path(args.save_dir)

    # Resolve derived output directories from the chosen save root.
    save_img_dir = save_root / "images"
    save_json_dir = save_root / "json"
    save_index_json_dir = Path(args.index_json_dir) if args.index_json_dir else None

    # Build the parameter bag consumed by the augmentation engine.
    augmentation_params = {
        # Random crop scale interval.
        "crop_scale_range": crop,
        # Rotation angle interval in degrees.
        "angle_limit": rotation,
        # Probability of applying rotation.
        "p_rotate": args.p_rotate,
        # Probability of horizontal flip.
        "p_flip_h": args.p_flip_h,
        # Probability of vertical flip.
        "p_flip_v": args.p_flip_v,
        # Probability of affine transform.
        "p_affine": args.p_affine,
        # Affine scale interval.
        "scale_limit": scale,
        # Affine translation interval (fraction of image size).
        "translate_limit": translate,
        # Probability of random crop.
        "p_crop": args.p_crop,
        # Brightness delta interval.
        "brightness_limit": brightness,
        # Contrast delta interval.
        "contrast_limit": contrast,
        # Probability of brightness/contrast transform.
        "p_brightness": args.p_brightness,
        # Contour simplification epsilon after mask extraction.
        "contour_simplify_epsilon": args.contour_simplify_epsilon,
        # Minimum connected component area retained during repair.
        "min_component_area": args.min_component_area,
        # Minimum mask pixel area retained during repair.
        "min_mask_pixel_area": args.min_mask_pixel_area,
        # Upper bound when num_augmentations is random.
        "random_aug_per_image": args.random_aug_per_image,
    }

    # Return a single runtime dictionary for CLI and library consistency.
    return {
        # Input image directory.
        "img_dir": img_dir,
        # Input LabelMe JSON directory.
        "json_dir": json_dir,
        # Root output directory.
        "save_root": save_root,
        # Output folder for augmented images.
        "save_img_dir": save_img_dir,
        # Output folder for augmented JSON annotations.
        "save_json_dir": save_json_dir,
        # Optional output folder for index-projection debug JSON.
        "save_index_json_dir": save_index_json_dir,
        # Number of augmentations to generate per source image.
        "num_per_image": args.num_per_image,
        # Enable verbose debug output in augmentor internals.
        "debug": args.debug,
        # Full augmentation and repair settings.
        "augmentation_params": augmentation_params,
    }


def print_run_summary(runtime: dict) -> None:
    # Print the resolved configuration once before processing begins.
    params = runtime["augmentation_params"]
    # Input/output directory summary.
    print("Image Dir:", runtime["img_dir"])
    print("Json Dir:", runtime["json_dir"])
    print("Chosen Augmentation Parameters:")
    # Geometric value ranges.
    print("crop:", list(params["crop_scale_range"]))
    print("rotation:", list(params["angle_limit"]))
    print("scale:", list(params["scale_limit"]))
    print("translate:", list(params["translate_limit"]))
    # Photometric value ranges.
    print("brightness:", list(params["brightness_limit"]))
    print("contrast:", list(params["contrast_limit"]))
    # Per-transform probabilities.
    print("p_rotate:", params["p_rotate"])
    print("p_flip_h:", params["p_flip_h"])
    print("p_flip_v:", params["p_flip_v"])
    print("p_affine:", params["p_affine"])
    print("p_crop:", params["p_crop"])
    print("p_brightness:", params["p_brightness"])
    # Repair/filter thresholds.
    print("contour_simplify_epsilon:", params["contour_simplify_epsilon"])
    print("min_component_area:", params["min_component_area"])
    print("min_mask_pixel_area:", params["min_mask_pixel_area"])
    print("random_aug_per_image:", params["random_aug_per_image"])
    # Final run/output metadata.
    print("num_per_image:", runtime["num_per_image"])
    print("Save_dir:", runtime["save_root"])
    print("Output Images:", runtime["save_img_dir"])
    print("Output Json:", runtime["save_json_dir"])
    print("Output Indexed Json:", runtime["save_index_json_dir"] if runtime["save_index_json_dir"] else "disabled")
