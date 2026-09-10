# Satellite Image Change Detection — Siamese ResNet-34 + FPN

Binary change detection on satellite imagery using a Siamese network with a shared ResNet-34 backbone and FPN-style decoder. Trained on LEVIR-CD, with scripts for Sentinel-2 inference on Lahore urban growth and the 2022 Indus River floods.

## Overview

Change detection from remote sensing imagery is a core task in urban planning, disaster response, and environmental monitoring. The goal is to take two co-registered satellite images of the same location captured at different times and produce a pixel-wise binary mask indicating where meaningful change has occurred — new construction, building demolition, land use shifts, or flood damage. This project approaches the problem as a binary segmentation task over image pairs, using a weight-sharing Siamese architecture to process both images through the same encoder and then compare the extracted features at multiple scales:

- A **Siamese encoder** (shared-weight ResNet backbone) extracts multi-scale features independently from the "before" (A) and "after" (B) images.
- Features at each scale are **compared** (via subtraction or concatenation) to isolate change signal while cancelling out shared, unchanged content.
- A lightweight **FPN-style decoder** progressively upsamples and fuses these difference features using skip connections, producing a full-resolution change probability map.
- The network is trained end-to-end with a combined **BCE + Dice loss** to handle the severe class imbalance inherent to change detection (most pixels don't change).

Beyond training and evaluation on LEVIR-CD, the repo includes a standalone Sentinel-2 downloader and a tiled inference script, so the trained model can be pointed at real-world imagery outside the benchmark dataset.

Beyond training and evaluation on LEVIR-CD, the repo includes a standalone Sentinel-2 downloader and a tiled inference script, so the trained model can be pointed at real-world imagery outside the benchmark dataset.

## Model Architecture

**Siamese ResNet-34 encoder + FPN-style decoder** (implemented in [`models/siamese.py`](models/siamese.py) via [`SiameseChangeDetector`](models/siamese.py)):

1. **Shared encoder** — A single ResNet backbone (ResNet-34 by default, ResNet-50 supported) processes image A and image B independently, with weights tied between the two passes. Backbones are loaded through [`segmentation-models-pytorch`](https://github.com/qubvel/segmentation_models.pytorch) with optional ImageNet pretraining.
2. **Feature differencing** — At every encoder stage, the two feature maps are combined via one of two configurable modes:
   - `subtract` — element-wise difference (fewer parameters, faster).
   - `concatenate` — channel-wise concatenation (more expressive, larger decoder).
3. **FPN decoder** ([`FPNDecoder`](models/siamese.py)) — Starting from the deepest difference feature map, each [`DecoderBlock`](models/siamese.py) upsamples by 2×, concatenates the corresponding shallower skip connection, and applies two Conv-BN-ReLU layers. This repeats across all decoder stages (default channel widths `[256, 128, 64, 32]`).
4. **Segmentation head** — A final 1×1 convolution projects to a single-channel logit map, which is bilinearly upsampled back to the input resolution.

The result is a binary logit mask the same size as the input; applying `sigmoid` and thresholding at 0.5 gives the predicted change mask.

## Dataset — LEVIR-CD

The model is trained on **[LEVIR-CD](https://justchenhao.github.io/LEVIR/)**, a large-scale building change detection benchmark consisting of 1024×1024 co-registered image pairs with pixel-level binary change annotations (445 train / 64 val / 128 test image pairs).

[`LEVIRDataset`](data/dataset.py) handles loading and preparation:

- **On-the-fly tiling** — full 1024×1024 images are cropped into overlapping `tile_size × tile_size` tiles (default 256, stride 192) at runtime rather than pre-cropped to disk.
- **Synchronized spatial augmentation** (train only) — horizontal/vertical flips, 90° rotations, shift-scale-rotate, and grid distortion, applied identically to image A, image B, and the mask via `albumentations`' `additional_targets`.
- **Independent color jitter** — brightness/contrast/saturation/hue jitter applied *separately* to A and B, simulating real-world differences in acquisition conditions between the two capture dates.
- **ImageNet normalization** for compatibility with the pretrained backbone.

Expected directory layout:

```
images/{train,val,test}/{A,B}/   # before/after image pairs
labels/{train,val,test}/         # binary masks (0 = no change, 255 = change)
```

## Loss Functions

Implemented in [`models/losses.py`](models/losses.py) as [`CombinedLoss`](models/losses.py):

- **Binary Cross-Entropy** (`BCEWithLogitsLoss`) with `pos_weight=8.0` to counteract the heavy class imbalance — change pixels are a small minority in most tiles.
- **Dice Loss** ([`DiceLoss`](models/losses.py)) to directly optimize region-level mask overlap, which BCE alone tends to under-weight for small/thin change regions.
- Combined as a **50/50 weighted sum**: `loss = 0.5 * BCE + 0.5 * Dice`.

## Training Details

Handled by [`train.py`](train.py):

- **Optimizer**: AdamW (`lr=1e-4`, `weight_decay=1e-4`)
- **Scheduler**: Cosine annealing over the full training run
- **Mixed precision (AMP)**: enabled automatically on CUDA via `torch.cuda.amp`
- **Early stopping**: stops if validation F1 doesn't improve for `patience` epochs (default 20)
- **Checkpointing**: the best model (by validation F1) is saved to `checkpoints/best.pth`, including model/optimizer state, epoch, best F1, and the full config used
- **Reproducibility**: fixed random seed (42) across `random`, `numpy`, and `torch`
- **Metrics**: computed dataset-wide via [`MetricTracker`](utils/metrics.py), which accumulates true/false positives and negatives across *all* batches before computing Precision/Recall/F1/IoU — this avoids the bias introduced by naively averaging per-batch metrics, which skews heavily negative on tiles with no change pixels

| Metric   | Target  |
|----------|---------|
| Test IoU | > 0.65  |
| Test F1  | > 0.78  |

## Repository Structure

```
change_detection_satellite/
├── configs/
│   └── default.yaml              # Data, model, training, and logging hyperparameters
├── data/
│   ├── __init__.py
│   └── dataset.py                 # LEVIRDataset — tiling, augmentation, mask loading
├── models/
│   ├── __init__.py
│   ├── siamese.py                 # SiameseChangeDetector: shared encoder + FPN decoder
│   └── losses.py                  # CombinedLoss (BCE + Dice)
├── utils/
│   ├── __init__.py
│   ├── metrics.py                 # MetricTracker, AverageMeter — dataset-wide metrics
│   └── visualize.py               # Sample panels + training curve plots
├── checkpoints/                   # Saved weights (best.pth) & training curve plots
├── test_lahore/                   # Sample before/after Lahore image pair for ad-hoc inference
├── train.py                       # Training loop entry point
├── evaluate.py                    # Test-set evaluation + qualitative sample generation
├── infer.py                       # Tiled inference on a single image pair
├── download_sentinel.py           # Sentinel-2 L2A downloader (Copernicus Data Space)
├── requirements.txt
├── Change_Detection_Research_Paper.pdf   # Accompanying research write-up
└── README.md
```

## Getting Started

### Installation

```bash
git clone <repo-url>
cd change_detection_satellite
pip install -r requirements.txt
```

### 1. Prepare the dataset

Download **LEVIR-CD** (e.g. from Kaggle) and arrange it as:

```
images/{train,val,test}/{A,B}/   — satellite image pairs
labels/{train,val,test}/         — binary masks (0/255)
```

### 2. Train

```bash
python train.py --config configs/default.yaml
```

Trains using the settings in `configs/default.yaml`, saving the best checkpoint to `checkpoints/best.pth` and a training curve plot to `checkpoints/training_curves.png`.

### 3. Evaluate

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pth
```

Runs the trained model over the test split, prints dataset-wide IoU/F1/Precision/Recall, saves them to `eval_results/test_metrics.json`, and writes qualitative before/after/ground-truth/prediction panels to `eval_results/`.

### 4. Run inference on a single image pair

```bash
python infer.py --img_a path/to/before.png --img_b path/to/after.png \
    --checkpoint checkpoints/best.pth --output change_mask.png
```

Automatically tiles large images, runs inference per tile, and stitches the result back into a full-resolution mask (plus a 3-panel visualization saved alongside it). A sample Lahore image pair is provided in [`test_lahore/`](test_lahore/) for a quick smoke test.

### 5. Download Sentinel-2 imagery (optional)

Requires a free [Copernicus Data Space](https://dataspace.copernicus.eu/) account (instructions printed on first run):

```bash
python download_sentinel.py --region lahore --year 2019
python download_sentinel.py --region lahore --year 2024
python download_sentinel.py --region floods --year 2022 --month 8
```

Pre-configured regions cover Lahore's urban periphery and the 2022 Indus River flood extent (Sindh / South Punjab), useful for testing the trained model on real-world change beyond LEVIR-CD.

## Configuration Reference

Key settings in [`configs/default.yaml`](configs/default.yaml):

| Key                        | Default          | Description                                      |
|-----------------------------|------------------|---------------------------------------------------|
| `data.root`                 | `./images`       | Path to the LEVIR-CD `images/` directory           |
| `data.tile_size`             | `256`            | Crop size for on-the-fly tiling                    |
| `data.stride`                 | `192`            | Stride between tiles (overlap = tile_size - stride) |
| `model.backbone`             | `resnet50`       | Encoder backbone (`resnet34`, `resnet50`, ...)      |
| `model.diff_mode`            | `concatenate`    | `subtract` or `concatenate` feature comparison      |
| `model.decoder_channels`     | `[256,128,64,32]`| Decoder stage channel widths                        |
| `training.epochs`            | `80`             | Max training epochs (early stopping may cut short)  |
| `training.batch_size`        | `8`              | Batch size                                          |
| `training.lr`                | `1e-4`           | AdamW learning rate                                 |
| `training.patience`          | `20`             | Early stopping patience (epochs)                    |

Switch backbone/diff mode by editing `configs/default.yaml` directly — `subtract` mode with a ResNet-34 backbone is a lighter-weight configuration well suited for faster experimentation, while `concatenate` with a deeper backbone trades compute for expressiveness.

## Training Logs
<img width="931" height="730" alt="image" src="https://github.com/user-attachments/assets/fdba6c94-55ab-4bd4-858b-b92ca9ec1a3e" />

## Training Curve
<img width="980" height="474" alt="image" src="https://github.com/user-attachments/assets/346f609e-0611-45f9-a6f4-bf27b5a7a415" />

## Test Results
<img width="394" height="149" alt="image" src="https://github.com/user-attachments/assets/8221f0b9-93f1-47ce-8a87-826139e8386d" />

## Change Mask Visualization
<img width="983" height="342" alt="image" src="https://github.com/user-attachments/assets/dba6f29f-8de1-4593-a823-d81e2f75b447" />

## Documentation

A more detailed write-up of the methodology, experiments, and results is available in [`Change_Detection_Research_Paper.pdf`](Change_Detection_Research_Paper.pdf).

---

## Team & Contributions

Developed collaboratively by Alizay Nasir and Khadija Rashid. Alizay led the research paper, architecture design decisions, and documentation, while Khadija led model training and GPU infrastructure.
