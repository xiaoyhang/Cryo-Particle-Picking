from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_processing import CryoParticleDataset, gaussian_heatmap, normalize_micrograph
from evaluate import match_coordinates
from predict import extract_peaks, predict_heatmap, save_overlay
from train import FocalLoss
from unet import UNet


def test_heatmap_to_coordinates() -> None:
    truth = np.array([[30, 40], [90, 70]], dtype=np.float32)
    heatmap = gaussian_heatmap(128, 128, truth, sigma=3)
    points = extract_peaks(heatmap, threshold=0.5, min_distance=10)
    metrics = match_coordinates(points, truth, max_distance=2)
    assert metrics["tp"] == 2
    assert metrics["f1"] == 1.0


def test_model_forward_and_backward() -> None:
    model = UNet(base_channels=4)
    image = torch.randn(2, 1, 64, 64)
    target = torch.zeros_like(image)
    target[:, :, 32, 32] = 1
    logits = model(image)
    assert logits.shape == target.shape
    loss = FocalLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss)


def test_normalization_and_overlay(tmp_path: Path) -> None:
    yy, xx = np.mgrid[:128, :128]
    raw = np.random.default_rng(0).normal(0, 0.2, (128, 128)).astype(np.float32)
    raw += np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * 8**2))
    normalized = normalize_micrograph(raw)
    assert abs(float(normalized.mean())) < 1e-5
    output = tmp_path / "overlay.png"
    save_overlay(raw, np.array([[64, 64, 0.9]], dtype=np.float32), output, radius=12)
    assert output.exists() and output.stat().st_size > 0


def test_dataset_and_sliding_prediction(tmp_path: Path) -> None:
    yy, xx = np.mgrid[:128, :128]
    image = np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * 8**2)).astype(np.float32)
    image_path = tmp_path / "micrograph.npy"
    coordinate_path = tmp_path / "coordinates.csv"
    np.save(image_path, image)
    coordinate_path.write_text("x,y\n64,64\n", encoding="utf-8")
    dataset = CryoParticleDataset(
        [{"micrograph_path": image_path, "coordinates_path": coordinate_path}],
        patch_size=128,
        patches_per_micrograph=1,
        heatmap_sigma=4,
        positive_crop_ratio=1.0,
        augment=False,
    )
    sample, target = dataset[0]
    assert sample.shape == target.shape == (1, 128, 128)
    assert float(target.max()) == 1.0

    class IdentityLogit(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value * 5

    predicted = predict_heatmap(
        IdentityLogit(), sample.numpy()[0], 64, 16, 2, torch.device("cpu")
    )
    assert predicted.shape == (128, 128)
    assert np.isfinite(predicted).all()

