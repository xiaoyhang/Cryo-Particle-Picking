"""Sliding-window inference, peak extraction, coordinate export, and visualization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from data_processing import load_config, normalize_micrograph, read_manifest, read_micrograph
from train import choose_device
from unet import UNet


def _starts(length: int, patch_size: int, overlap: int) -> list[int]:
    if length <= patch_size:
        return [0]
    stride = patch_size - overlap
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


@torch.no_grad()
def predict_heatmap(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int,
    overlap: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    original_h, original_w = image.shape
    pad_h, pad_w = max(0, patch_size - original_h), max(0, patch_size - original_w)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    height, width = image.shape
    probability_sum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    locations = [(y, x) for y in _starts(height, patch_size, overlap) for x in _starts(width, patch_size, overlap)]
    model.eval()
    for offset in range(0, len(locations), batch_size):
        batch_locations = locations[offset : offset + batch_size]
        patches = np.stack(
            [image[y : y + patch_size, x : x + patch_size] for y, x in batch_locations]
        )
        tensor = torch.from_numpy(patches[:, None]).to(device)
        probabilities = torch.sigmoid(model(tensor)).cpu().numpy()[:, 0]
        for probability, (y, x) in zip(probabilities, batch_locations):
            probability_sum[y : y + patch_size, x : x + patch_size] += probability
            count[y : y + patch_size, x : x + patch_size] += 1
    return (probability_sum / np.maximum(count, 1))[:original_h, :original_w]


def extract_peaks(
    heatmap: np.ndarray, threshold: float, min_distance: int
) -> np.ndarray:
    tensor = torch.from_numpy(heatmap)[None, None]
    kernel = 2 * max(1, int(min_distance)) + 1
    maxima = F.max_pool2d(tensor, kernel, stride=1, padding=kernel // 2)
    mask = (tensor == maxima) & (tensor >= threshold)
    yy, xx = torch.where(mask[0, 0])
    candidates = [
        (float(x), float(y), float(heatmap[y, x])) for y, x in zip(yy.tolist(), xx.tolist())
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)
    kept: list[tuple[float, float, float]] = []
    distance_sq = float(min_distance**2)
    for candidate in candidates:
        if all((candidate[0] - p[0]) ** 2 + (candidate[1] - p[1]) ** 2 >= distance_sq for p in kept):
            kept.append(candidate)
    return np.asarray(kept, dtype=np.float32).reshape(-1, 3)


def save_coordinates(points: np.ndarray, csv_path: Path, star_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "confidence"])
        writer.writerows(points.tolist())
    with star_path.open("w", encoding="utf-8") as handle:
        handle.write("data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnAutopickFigureOfMerit #3\n")
        for x, y, confidence in points:
            handle.write(f"{x:.2f} {y:.2f} {confidence:.6f}\n")


def _display_image(image: np.ndarray) -> Image.Image:
    low, high = np.percentile(image, (0.5, 99.5))
    scaled = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    return Image.fromarray((scaled * 255).astype(np.uint8)).convert("RGB")


def save_overlay(image: np.ndarray, points: np.ndarray, path: Path, radius: float) -> None:
    canvas = _display_image(image)
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, int(round(radius / 15)))
    for x, y, _ in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="red", width=line_width)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_heatmap(heatmap: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8)).save(path)


def load_model(checkpoint: str, config: dict, device: torch.device) -> UNet:
    model = UNet(base_channels=int(config["model"]["base_channels"])).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


def predict_file(path: Path, model: UNet, config: dict, device: torch.device) -> Path:
    raw = read_micrograph(path)
    image = normalize_micrograph(raw)
    pred_cfg = config["prediction"]
    heatmap = predict_heatmap(
        model,
        image,
        int(config["data"]["patch_size"]),
        int(pred_cfg["overlap"]),
        int(pred_cfg["batch_size"]),
        device,
    )
    points = extract_peaks(
        heatmap,
        float(pred_cfg["confidence_threshold"]),
        int(pred_cfg["nms_distance"]),
    )
    output = Path(config["output"]["directory"])
    stem = path.stem
    csv_path = output / "coordinates" / f"{stem}.csv"
    save_coordinates(points, csv_path, csv_path.with_suffix(".star"))
    diameter = float(config["data"]["particle_diameter"])
    if config["output"].get("save_overlay", True):
        save_overlay(raw, points, output / "overlays" / f"{stem}_overlay.png", diameter / 2)
    if config["output"].get("save_heatmap", True):
        save_heatmap(heatmap, output / "heatmaps" / f"{stem}_heatmap.png")
    print(f"{path.name}: {len(points)} particles")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", nargs="*", help="Micrograph paths; omit to use manifest test split")
    args = parser.parse_args()
    config = load_config(args.config)
    device = choose_device(str(config["training"].get("device", "auto")))
    model = load_model(args.checkpoint, config, device)
    paths = [Path(value) for value in args.input] if args.input else [
        Path(row["micrograph_path"]) for row in read_manifest(config["data"]["manifest"], "test")
    ]
    if not paths:
        raise ValueError("No input micrographs were provided or found in the test split")
    for path in paths:
        predict_file(path, model, config, device)


if __name__ == "__main__":
    main()

