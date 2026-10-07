# CALE

**Collaborative adjustment based on lacunarity and entropy for long-tailed object detection**

CALE extends [FRACAL](https://github.com/kostas1515/FRACAL) with entropy-weighted frequency adjustment and a lacunarity prior. It adjusts the predictions of a frozen detector without retraining. This repository contains the implementation, inference configurations, precomputed priors, and scripts for reproducing the main results on LVIS v1.0 and V3Det.

## Installation

Use Linux, Python 3.10, and a CUDA GPU. Clone this repository and install its dependencies:

```bash
git clone https://github.com/tracycwl94-commits/CALE.git
cd CALE
conda create -n cale python=3.10 -y
conda activate cale
pip install "numpy<2"
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install -r requirements.txt
pip install -e . --no-deps
```

The commands above use the CUDA 11.8 / PyTorch 2.1 MMCV wheels. For GPUs requiring a newer CUDA/PyTorch version, install a compatible PyTorch build and build MMCV 2.1.0 against that environment following the [MMCV installation instructions](https://mmcv.readthedocs.io/en/v2.1.0/get_started/installation.html). The bundled `mmdet/` contains the detector implementations used here; a separate MMDetection installation is unnecessary.

## Datasets

- **LVIS v1.0:** download the COCO 2017 [training images](http://images.cocodataset.org/zips/train2017.zip) and [validation images](http://images.cocodataset.org/zips/val2017.zip), together with the LVIS [training annotations](https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip) and [validation annotations](https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip). LVIS validation uses images from both COCO image folders.
- **V3Det:** download the images and v1 annotations from the [official dataset repository](https://github.com/V3Det/V3Det).

Extract the files as follows (or pass your existing dataset directory with `--data-root`):

```text
data/
├── lvis/
│   ├── train2017/
│   ├── val2017/
│   ├── lvis_v1_train.json
│   └── lvis_v1_val.json
└── v3det/
    ├── images/
    └── annotations/
        ├── v3det_2023_v1_train.json
        └── v3det_2023_v1_val.json
```

Training annotations are needed only when regenerating priors. The priors used for evaluation are already provided in `stat_files/`.

## Checkpoints

Download the checkpoint for the desired configuration and save it under `checkpoints/` using the filename below. The baseline and CALE use the **same checkpoint**.

| Dataset / detector | CALE configuration | Checkpoint filename | Download |
| --- | --- | --- | --- |
| LVIS / Mask R-CNN R50 | [lvis_r50.py](configs/lvis_r50.py) | `lvis_r50.pth` | [FRACAL](https://drive.usercontent.google.com/download?id=1W5SAFxGrygPISX-DI905PzbjBahU1Lgk&export=download) |
| LVIS / Mask R-CNN R101 | [lvis_r101.py](configs/lvis_r101.py) | `lvis_r101.pth` | [FRACAL](https://drive.usercontent.google.com/download?id=1wWYsxIZYnqMvUYz2whRMN5KfFMC5ihF5&export=download) |
| LVIS / Mask R-CNN X101-32x4d | [lvis_x101_32.py](configs/lvis_x101_32.py) | `lvis_x101_32.pth` | [MMDetection](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-ebbc5c81.pth) |
| LVIS / Mask R-CNN X101-64x4d | [lvis_x101_64.py](configs/lvis_x101_64.py) | `lvis_x101_64.pth` | [MMDetection](https://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-43d9edfe.pth) |
| V3Det / Faster R-CNN + APA | [v3det_faster.py](configs/v3det_faster.py) | `v3det_faster.pth` | [APA/AGLU](https://drive.usercontent.google.com/download?id=1kSZkewkLNvpRcIE9f2fHVDvvn4xNCiI-&export=download) |
| V3Det / Cascade R-CNN + APA | [v3det_cascade.py](configs/v3det_cascade.py) | `v3det_cascade.pth` | [APA/AGLU](https://drive.usercontent.google.com/download?id=1dhF2N-4ndpjFt46MA5hsnXx1zoUGks6u&export=download) |
| LVIS / ATSS R50 | [lvis_atss.py](configs/lvis_atss.py) | `lvis_atss.pth` | [Download](https://drive.google.com/file/d/1gcKVckaYHxjTxm3OMp7g2H7-76h5SFj1/view?usp=drive_link) |
The ATSS checkpoint was trained by us on LVIS v1.0 for 12 epochs.
The V3Det configurations retain the APA/AGLU architecture and its learned parameters. Use the linked APA checkpoints. File hashes for the evaluated checkpoints are listed in [checkpoints.json](checkpoints.json).

## Evaluation

Run commands from the repository root. For Mask R-CNN R50 on LVIS:

```bash
# Original detector
python tools/test.py configs/lvis_r50_baseline.py checkpoints/lvis_r50.pth

# CALE
python tools/test.py configs/lvis_r50.py checkpoints/lvis_r50.pth
```

For V3Det:

```bash
python tools/test.py configs/v3det_faster_baseline.py checkpoints/v3det_faster.pth
python tools/test.py configs/v3det_faster.py checkpoints/v3det_faster.pth
```

For the other models, substitute the configuration and checkpoint from the table. Every CALE configuration has a corresponding `*_baseline.py` configuration. Results and predictions are written to `work_dirs/<configuration>/`.

To reuse a dataset elsewhere or perform a quick installation check:

```bash
python tools/test.py configs/lvis_r50.py checkpoints/lvis_r50.pth \
    --data-root /path/to/lvis --limit 2 --work-dir work_dirs/smoke
```

Remove `--limit 2` for full evaluation. `--limit` only checks inference and does not report AP. Use `--batch-size 1` if GPU memory is insufficient; use the configuration's default batch size when reproducing the reported settings. Small numerical differences can occur across hardware and software versions.

LVIS evaluation reports official box/mask AP, including rare/common/frequent groups. V3Det evaluation reproduces the paper's category-wise COCO box-AP protocol (`maxDets=300`, without hierarchical parent-category ignore rules), followed by the supplied training-frequency groups. These groups are diagnostic splits, not official V3Det splits. The configuration files specify all inference thresholds; the original R50/R101 baseline settings are retained for baseline evaluation.

## Prior generation (optional)

The supplied priors use variance exponent 1.25 and an adaptive scale bound capped at 64. To regenerate them from training annotations:

```bash
python tools/generate_priors.py --dataset lvis \
    --annotations data/lvis/lvis_v1_train.json --output-dir work_dirs/priors_lvis

python tools/generate_priors.py --dataset v3det \
    --annotations data/v3det/annotations/v3det_2023_v1_train.json --output-dir work_dirs/priors_v3det
```

`*_frequency.csv` stores category image counts and the base-10 frequency term. `*_lacunarity.csv` stores the final structural factor `lambda` in classifier category order. The provided files preserve the values used in the experiments; regeneration can differ slightly through floating-point rounding. Generation does not require detector weights or image pixels.

## Code structure

```text
cale/          CALE adapters for ROI heads and ATSS
configs/       Baseline and CALE inference configurations
stat_files/    Precomputed frequency and lacunarity priors
tools/         Evaluation and prior-generation scripts
mmdet/         Bundled MMDetection / FRACAL detector code
```

## Acknowledgements and license

This implementation builds on [FRACAL](https://github.com/kostas1515/FRACAL), [MMDetection](https://github.com/open-mmlab/mmdetection), and the [APA/AGLU implementation](https://github.com/kostas1515/AGLU). We thank the authors for releasing their code and checkpoints. The source code is distributed under the [Apache License 2.0](LICENSE). Dataset and checkpoint usage follows the terms of their original providers.
