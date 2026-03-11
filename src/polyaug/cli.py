"""Command-line entrypoint wiring parser config to the augmentation engine."""

from polyaug.helper import build_parser, build_runtime_config, print_run_summary


def main() -> None:
    # Build and parse CLI arguments.
    parser = build_parser()
    # Parse terminal inputs into a typed argparse namespace.
    args = parser.parse_args()
    # Translate parsed args into the runtime dictionary used across the app.
    runtime = build_runtime_config(args, parser)

    # Show the final resolved run configuration to the user.
    print_run_summary(runtime)

    # Delay heavy import until after argument validation.
    from ringaug.augmentor import IndexPreservingPolygonAugmentor

    # Instantiate augmentor and execute dataset augmentation.
    # The debug flag is forwarded from CLI for verbose internals when needed.
    augmentor = IndexPreservingPolygonAugmentor(debug=runtime["debug"])
    augmentor.augment_dataset(
        # Source image directory.
        data_dir=runtime["img_dir"],
        # Source LabelMe JSON directory.
        json_dir=runtime["json_dir"],
        # Destination for augmented images.
        save_img_dir=runtime["save_img_dir"],
        # Destination for augmented LabelMe JSON.
        save_json_dir=runtime["save_json_dir"],
        # Destination for index-projection diagnostic JSON.
        save_index_json_dir=runtime["save_index_json_dir"],
        # Number of augmented samples to generate for each source image.
        num_augmentations=runtime["num_per_image"],
        # Transform and repair parameter map consumed by augmentor.
        augmentation_params=runtime["augmentation_params"],
    )


if __name__ == "__main__":
    # Allow running as `python -m ringaug.cli`.
    main()
