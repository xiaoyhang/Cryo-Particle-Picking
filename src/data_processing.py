"""Data loading, cropping, augmentation, and heatmap target creation."""

from __future__ import annotations

import csv
from functools import lru_cache
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML to read the project configuration.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_micrograph(path: str | Path) -> np.ndarray:
    """Read a 2-D MRC, NPY, or common image file as float32."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".mrc", ".mrcs"}:
        try:
            import mrcfile
        except ImportError as exc:
            raise ImportError("Install mrcfile to read MRC data.") from exc
        with mrcfile.open(path, permissive=True) as handle:
            image = np.asarray(handle.data, dtype=np.float32).squeeze().copy()
    elif suffix == ".npy":
        image = np.asarray(np.load(path), dtype=np.float32).squeeze()
    else:
        image = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D micrograph, got {image.shape} from {path}")
    return image


def normalize_micrograph(image: np.ndarray) -> np.ndarray:
    """Robustly clip extreme pixels and return a zero-mean, unit-variance image."""
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("Micrograph has no finite pixels")
    fill = float(np.median(image[finite]))
    image = np.nan_to_num(image, nan=fill, posinf=fill, neginf=fill)
    low, high = np.percentile(image, (0.5, 99.5))
    image = np.clip(image, low, high)
    mean, std = float(image.mean()), float(image.std())
    return (image - mean) / max(std, 1e-6)


@lru_cache(maxsize=2)
def _read_normalized_cached(path: str) -> np.ndarray:
    """Keep only the most recent micrographs in each data-loader worker."""
    return normalize_micrograph(read_micrograph(path))


def _find_xy_columns(fieldnames: list[str]) -> tuple[str, str]:
    aliases_x = {"x", "coord_x", "coordinate_x", "rlncoordinatex"}
    aliases_y = {"y", "coord_y", "coordinate_y", "rlncoordinatey"}
    normalized = {name.lower().lstrip("_"): name for name in fieldnames}
    x_key = next((normalized[key] for key in aliases_x if key in normalized), None)
    y_key = next((normalized[key] for key in aliases_y if key in normalized), None)
    if x_key is None or y_key is None:
        raise ValueError(f"Could not find x/y coordinate columns in {fieldnames}")
    return x_key, y_key


def read_coordinates(path: str | Path) -> np.ndarray:
    """Read center coordinates from CSV or a simple RELION STAR file."""
    path = Path(path)
    if path.suffix.lower() == ".star":
        return _read_star_coordinates(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return np.empty((0, 2), dtype=np.float32)
        x_key, y_key = _find_xy_columns(reader.fieldnames)
        rows = [[float(row[x_key]), float(row[y_key])] for row in reader]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 2)


@lru_cache(maxsize=32)
def _read_coordinates_cached(path: str) -> np.ndarray:
    return read_coordinates(path)


def _read_star_coordinates(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    columns: list[str] = []
    data: list[list[float]] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("data_") or line == "loop_":
            continue
        if line.startswith("_"):
            columns.append(line.split()[0])
            continue
        if columns:
            parts = line.split()
            if len(parts) >= len(columns):
                data.append([float(v) for v in parts[: len(columns)]])
    normalized = [name.lower().lstrip("_") for name in columns]
    try:
        ix = normalized.index("rlncoordinatex")
        iy = normalized.index("rlncoordinatey")
    except ValueError as exc:
        raise ValueError(f"STAR file lacks RELION coordinate columns: {path}") from exc
    return np.asarray([[row[ix], row[iy]] for row in data], dtype=np.float32).reshape(-1, 2)


def read_manifest(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    base = path.parent
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: list[dict[str, Any]] = []
    for row in rows:
        if split is not None and row.get("split", "") != split:
            continue
        item: dict[str, Any] = dict(row)
        for key in ("micrograph_path", "coordinates_path"):
            value = Path(item[key]).expanduser()
            item[key] = value if value.is_absolute() else (base / value).resolve()
        if item.get("particle_diameter"):
            item["particle_diameter"] = float(item["particle_diameter"])
        result.append(item)
    return result


def gaussian_heatmap(
    height: int, width: int, coordinates: np.ndarray, sigma: float
) -> np.ndarray:
    heatmap = np.zeros((height, width), dtype=np.float32)
    radius = max(1, int(np.ceil(3 * sigma)))
    for x, y in np.asarray(coordinates, dtype=np.float32).reshape(-1, 2):
        cx, cy = int(round(float(x))), int(round(float(y)))
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        patch = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], patch)
    return heatmap


def _pad_to_size(image: np.ndarray, size: int) -> np.ndarray:
    pad_h, pad_w = max(0, size - image.shape[0]), max(0, size - image.shape[1])
    if not pad_h and not pad_w:
        return image
    return np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")


def _crop(
    image: np.ndarray,
    coordinates: np.ndarray,
    size: int,
    positive_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    image = _pad_to_size(image, size)
    height, width = image.shape
    if len(coordinates) and random.random() < positive_probability:
        x, y = coordinates[random.randrange(len(coordinates))]
        left = int(round(x - random.uniform(0.25, 0.75) * size))
        top = int(round(y - random.uniform(0.25, 0.75) * size))
        left = min(max(0, left), width - size)
        top = min(max(0, top), height - size)
    else:
        left = random.randint(0, width - size)
        top = random.randint(0, height - size)
    local = coordinates - np.array([left, top], dtype=np.float32)
    keep = (
        (local[:, 0] >= 0)
        & (local[:, 0] < size)
        & (local[:, 1] >= 0)
        & (local[:, 1] < size)
    )
    return image[top : top + size, left : left + size], local[keep]


def _augment(image: np.ndarray, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    coords = coordinates.copy()
    if random.random() < 0.5:
        image = image[:, ::-1]
        coords[:, 0] = width - 1 - coords[:, 0]
    if random.random() < 0.5:
        image = image[::-1, :]
        coords[:, 1] = height - 1 - coords[:, 1]
    k = random.randrange(4)
    for _ in range(k):
        image = np.rot90(image)
        old_x = coords[:, 0].copy()
        coords[:, 0] = coords[:, 1]
        coords[:, 1] = width - 1 - old_x
    image = image * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
    if random.random() < 0.5:
        image = image + np.random.normal(0, random.uniform(0, 0.08), image.shape)
    return np.ascontiguousarray(image, dtype=np.float32), coords


class CryoParticleDataset(Dataset):
    """Random-patch dataset constructed from complete micrographs."""

    def __init__(
        self,
        entries: list[dict[str, Any]],
        patch_size: int,
        patches_per_micrograph: int,
        heatmap_sigma: float,
        positive_crop_ratio: float = 0.7,
        augment: bool = True,
    ) -> None:
        if not entries:
            raise ValueError("Dataset has no manifest entries")
        self.entries = entries
        self.patch_size = patch_size
        self.patches_per_micrograph = patches_per_micrograph
        self.heatmap_sigma = heatmap_sigma
        self.positive_crop_ratio = positive_crop_ratio
        self.augment = augment

    def __len__(self) -> int:
        return len(self.entries) * self.patches_per_micrograph

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[index % len(self.entries)]
        image = _read_normalized_cached(str(entry["micrograph_path"]))
        coordinates = _read_coordinates_cached(str(entry["coordinates_path"]))
        image, coordinates = _crop(
            image, coordinates, self.patch_size, self.positive_crop_ratio
        )
        if self.augment:
            image, coordinates = _augment(image, coordinates)
        target = gaussian_heatmap(
            self.patch_size, self.patch_size, coordinates, self.heatmap_sigma
        )
        return torch.from_numpy(image[None]), torch.from_numpy(target[None])

