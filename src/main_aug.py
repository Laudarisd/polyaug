"""Compatibility wrapper for older imports.

Use `src.augmentor` for the implementation.
"""

from src.augmentor import IndexPreservingPolygonAugmentor

__all__ = ["IndexPreservingPolygonAugmentor"]


if __name__ == "__main__":
    # Keep previous direct execution behavior for local manual runs.
    params = {
        "crop_scale_range": (0.8, 0.9),
        "angle_limit": (-30, 30),
        "p_rotate": 0.9,
        "p_flip_h": 0.2,
        "p_flip_v": 0.1,
        "p_affine": 0.8,
        "scale_limit": (0.7, 1.3),
        "translate_limit": (-0.1, 0.1),
        "p_crop": 0.7,
        "brightness_limit": (-0.1, 0.1),
        "contrast_limit": (-0.1, 0.1),
        "p_brightness": 0.4,
        "random_aug_per_image": 3,
        "contour_simplify_epsilon": 1.5,
        "min_component_area": 12.0,
        "min_mask_pixel_area": 32,
        "min_repair_polygon_area": 1.0,
        "repair_dedupe_eps": 0.5,
        "source_overlap_eps": 0.5,
        "max_projection_distance_for_repair": 4.0,
        "min_retained_vertex_ratio_for_repair": 0.7,
    }

    augmentor = IndexPreservingPolygonAugmentor(debug=False)
    augmentor.augment_dataset(
        data_dir="seg-topo-augment/images",
        json_dir="seg-topo-augment/json",
        save_img_dir="seg-topo-augment/augmented/images",
        save_json_dir="seg-topo-augment/augmented/json",
        save_index_json_dir="seg-topo-augment/augmented/augmented_index_json",
        num_augmentations=2,
        augmentation_params=params,
    )
