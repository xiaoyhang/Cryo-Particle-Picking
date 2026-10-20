# Cryo-EM Particle Picking with a Lightweight U-Net

A minimal course-project pipeline that represents particle picking as center-point
heatmap prediction. A compact U-Net predicts a probability heatmap; local maxima and
distance-based non-maximum suppression turn it into particle coordinates.

## Pipeline

```text
MRC micrograph + center coordinates
        -> random crops and Gaussian heatmap targets
        -> lightweight U-Net training
        -> sliding-window heatmap prediction
        -> local peaks and distance NMS
        -> CSV/RELION STAR coordinates and red-circle overlays
        -> coordinate-level precision, recall, F1, and localization error
```

The heatmap is an intermediate representation. Evaluation is performed on the final
particle centers, not on background-dominated pixel accuracy.

## Project layout

```text
src/data_processing.py  Data loading, crops, augmentation, heatmap targets
src/unet.py             U-Net model
src/train.py            Losses, training, validation, checkpoints
src/predict.py          Tiled inference, peak/NMS, CSV/STAR, overlays
src/evaluate.py         One-to-one coordinate matching and metrics
tests/test_pipeline.py  Small synthetic end-to-end component tests
cluster/train.slurm     Single-GPU SLURM template
```

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Do not add real CryoPPP data to Git. Place it on cluster project or scratch storage and
create `data/manifest.csv` from `data/manifest.example.csv`. Each micrograph belongs to
exactly one of `train`, `val`, or `test`; the split must happen before patch cropping.

Coordinate CSV files require `x,y` columns. RELION STAR coordinates are also supported.

## Train

```bash
python src/train.py --config config.yaml
```

The best and latest checkpoints and training history are written below the configured
`output.directory`.

## Predict and visualize

Predict the manifest test split:

```bash
python src/predict.py --config config.yaml \
  --checkpoint outputs/unet_baseline/best_model.pt
```

Or predict selected micrographs:

```bash
python src/predict.py --config config.yaml \
  --checkpoint outputs/unet_baseline/best_model.pt \
  --input /path/to/micrograph.mrc
```

Outputs include CSV and RELION STAR coordinates, grayscale heatmaps, and original
micrographs with red circles centered on predicted particles.

## Evaluate

```bash
python src/evaluate.py --config config.yaml \
  --pred-dir outputs/unet_baseline/coordinates
```

Predictions are processed from highest to lowest confidence and matched one-to-one to
the nearest unmatched annotation inside `evaluation.matching_distance`.

## Test without downloading CryoPPP

```bash
pytest -q
```

The tests create small synthetic arrays in temporary storage and check heatmap target
generation, U-Net forward/backward propagation, peak extraction, coordinate metrics,
and red-circle visualization.

