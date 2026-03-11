# PolyAug

[![PyPI version](https://img.shields.io/pypi/v/polyaug)](https://pypi.org/project/polyaug/)
[![Python](https://img.shields.io/pypi/pyversions/polyaug)](https://pypi.org/project/polyaug/)
![Augmentation](https://img.shields.io/badge/domain-augmentation-0ea5e9)
![Albumentations](https://img.shields.io/badge/library-albumentations-22c55e)
![Topology-Aware](https://img.shields.io/badge/focus-topology--aware-f59e0b)

PolyAug is a Python package and CLI for topology-aware polygon augmentation for segmentation datasets.

Generating augmented datasets is common with existing libraries such as Albumentations and Augmentor; however, they do not reliably preserve polygon structure when the polygon forms a ring or path-like shape. Such structures are common in raster-to-vector tasks and datasets such as floorplans.

See the paper for more details: [Topology-Preserving Data Augmentation for Ring-Type Polygon Annotations](https://arxiv.org/abs/2606.12345).

This package addresses this gap by mitigating the topology inconsistency in such processes.

For installation, usage, and development instructions, see the sections below.

---

## Features

* Augments images and LabelMe JSON together
* Choose augmentation parameters and ranges, or use defaults
* Supports cropping, rotation, scaling, translation, brightness, contrast
* CLI and Python API for flexibility
* Generates LabelMe JSON annotations in standard and indexed formats
---

## Installation

We recommend installing inside a virtual environment to avoid dependency conflicts.

Supports windows, macOS, and Linux.

### Install from PyPI

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

```bash
pip install polyaug
```

## Quick CLI Check

After installation, verify the CLI is available:

```bash
polyaug --help
```

---

## Basic CLI Usage

```bash
polyaug \
  --img-dir ./dataset/images \
  --json-dir ./dataset/json \
  --save-dir ./output \
  --num-per-image 2 \
  --crop 0.8 0.9 \
  --rotation -30 30 \
  --scale 0.7 1.3 \
  --translate -0.1 0.1 \
  --brightness -0.1 0.1 \
  --contrast -0.1 0.1
```

### Inputs

* `--img-dir`: directory containing source images
* `--json-dir`: directory containing matching LabelMe JSON files

### Outputs

Inside `--save-dir`, PolyAug writes:

* `images/` → augmented images
* `json/` → standard LabelMe JSON annotations
* `augmented_index_json/` → indexed/debug JSON annotations, only when `--index-json-dir` is provided

---

## CLI Arguments

### Required

* `--img-dir`
* `--json-dir`

### Optional

* `--save-dir`
* `--index-json-dir`
* `--num-per-image`
* `--crop MIN MAX`
* `--rotation MIN MAX`
* `--scale MIN MAX`
* `--translate MIN MAX`
* `--brightness MIN MAX`
* `--contrast MIN MAX`
* `--p-rotate`
* `--p-flip-h`
* `--p-flip-v`
* `--p-affine`
* `--p-crop`
* `--p-brightness`
* `--contour-simplify-epsilon`
* `--min-component-area`
* `--min-mask-pixel-area`
* `--random-aug-per-image`
* `--debug`

To see the latest CLI options:

```bash
polyaug --help
```
---

## Project Structure for Building the Package

A clean structure for the package is:

```text
polyaug/
├── LICENSE
├── README.md
├── pyproject.toml
├── src/
│   └── polyaug/
│       ├── __init__.py
│       ├── augmentor.py
│       ├── cli.py
│       └── helper.py
└── tests/
    └── test_imports.py
```

---

## Citation / Research Use

You can cite PolyAug as:

```bibtex
@software{laudari2026polyaug,
  title={PolyAug: Topology-Aware Polygon Augmentation for Segmentation Datasets},
  author={Laudari, Sudip},
  year={2026},
  version={0.1.5},
  url={https://pypi.org/project/polyaug/}
}
```

---

## License
MIT License. See [LICENSE](LICENSE) for details.
