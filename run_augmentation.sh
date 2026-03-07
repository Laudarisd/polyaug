#!/usr/bin/env bash
set -euo pipefail

# ================ User Settings ================
# Input paths
ORIGINAL_IMG_DIR="./seg-topo-augment/images"
ORIGINAL_JSON_DIR="./seg-topo-augment/json"

# Output paths
OUTPUT_DIR="./seg-topo-augment/augmented"

# Number of augmented images per source image
NUM_PER_IMAGE=3

# ----------- Augmentation Ranges -----------
# crop scale range: 0.1 to 1.0
CROP_SCALE_MIN=0.80
CROP_SCALE_MAX=0.90

# rotation angle range in degrees: -180 to 180
ANGLE_MIN=-30
ANGLE_MAX=30

# affine scale range: >0
SCALE_MIN=0.70
SCALE_MAX=1.30

# translation percent range: -1.0 to 1.0
TRANSLATE_MIN=-0.10
TRANSLATE_MAX=0.10

# brightness/contrast delta range: usually -1.0 to 1.0
BRIGHTNESS_MIN=-0.10
BRIGHTNESS_MAX=0.10
CONTRAST_MIN=-0.10
CONTRAST_MAX=0.10

# ----------- Probabilities (0.0 to 1.0) -----------
P_ROTATE=0.90
P_FLIP_H=0.20
P_FLIP_V=0.10
P_AFFINE=0.80
P_CROP=0.70
P_BRIGHTNESS=0.40

# ----------- Topology/Contour Controls -----------
CONTOUR_SIMPLIFY_EPSILON=1.5
MIN_COMPONENT_AREA=12.0
MIN_MASK_PIXEL_AREA=12.0
RANDOM_AUG_PER_IMAGE=3

# Debug mode: set to true to enable
DEBUG=false
# ================ End Settings ================

echo "========================================"
echo "Running Seg-TOPO augmentation"
echo "Input images: ${ORIGINAL_IMG_DIR}"
echo "Input json:   ${ORIGINAL_JSON_DIR}"
echo "Output root:  ${OUTPUT_DIR}"
echo "Num/image:    ${NUM_PER_IMAGE}"
echo "========================================"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed or not in PATH."
  echo "Install uv, then re-run this script."
  exit 1
fi

CMD=(
  uv run python main.py
  --original-img-dir "${ORIGINAL_IMG_DIR}"
  --original-json-dir "${ORIGINAL_JSON_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --num-per-image "${NUM_PER_IMAGE}"
  --crop-scale-min "${CROP_SCALE_MIN}"
  --crop-scale-max "${CROP_SCALE_MAX}"
  --angle-min "${ANGLE_MIN}"
  --angle-max "${ANGLE_MAX}"
  --scale-min "${SCALE_MIN}"
  --scale-max "${SCALE_MAX}"
  --translate-min "${TRANSLATE_MIN}"
  --translate-max "${TRANSLATE_MAX}"
  --brightness-min "${BRIGHTNESS_MIN}"
  --brightness-max "${BRIGHTNESS_MAX}"
  --contrast-min "${CONTRAST_MIN}"
  --contrast-max "${CONTRAST_MAX}"
  --p-rotate "${P_ROTATE}"
  --p-flip-h "${P_FLIP_H}"
  --p-flip-v "${P_FLIP_V}"
  --p-affine "${P_AFFINE}"
  --p-crop "${P_CROP}"
  --p-brightness "${P_BRIGHTNESS}"
  --contour-simplify-epsilon "${CONTOUR_SIMPLIFY_EPSILON}"
  --min-component-area "${MIN_COMPONENT_AREA}"
  --min-mask-pixel-area "${MIN_MASK_PIXEL_AREA}"
  --random-aug-per-image "${RANDOM_AUG_PER_IMAGE}"
)

if [[ "${DEBUG}" == "true" ]]; then
  CMD+=(--debug)
fi

"${CMD[@]}"

echo "========================================"
echo "Done."
echo "========================================"
