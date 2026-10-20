"""Coordinate-level evaluation for particle picking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data_processing import load_config, read_coordinates, read_manifest


def read_predictions(path: str | Path) -> np.ndarray:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray(
        [[float(row["x"]), float(row["y"]), float(row.get("confidence", 1.0))] for row in rows],
        dtype=np.float32,
    ).reshape(-1, 3)


def match_coordinates(
    predictions: np.ndarray, ground_truth: np.ndarray, max_distance: float
) -> dict[str, float | int]:
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 3)
    ground_truth = np.asarray(ground_truth, dtype=np.float32).reshape(-1, 2)
    order = np.argsort(-predictions[:, 2]) if len(predictions) else []
    unmatched = set(range(len(ground_truth)))
    distances: list[float] = []
    for index in order:
        if not unmatched:
            break
        candidates = np.asarray(sorted(unmatched), dtype=int)
        delta = ground_truth[candidates] - predictions[index, :2]
        values = np.sqrt(np.sum(delta**2, axis=1))
        nearest = int(np.argmin(values))
        if values[nearest] <= max_distance:
            unmatched.remove(int(candidates[nearest]))
            distances.append(float(values[nearest]))
    tp = len(distances)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_localization_error": float(np.mean(distances)) if distances else None,
    }


def evaluate_manifest(config: dict, prediction_dir: str | Path) -> dict:
    prediction_dir = Path(prediction_dir)
    rows = read_manifest(config["data"]["manifest"], "test")
    per_micrograph = []
    total_tp = total_fp = total_fn = 0
    all_dist_weighted = 0.0
    for row in rows:
        stem = Path(row["micrograph_path"]).stem
        pred_path = prediction_dir / f"{stem}.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")
        metrics = match_coordinates(
            read_predictions(pred_path),
            read_coordinates(row["coordinates_path"]),
            float(config["evaluation"]["matching_distance"]),
        )
        metrics["micrograph"] = stem
        per_micrograph.append(metrics)
        total_tp += int(metrics["tp"])
        total_fp += int(metrics["fp"])
        total_fn += int(metrics["fn"])
        if metrics["mean_localization_error"] is not None:
            all_dist_weighted += float(metrics["mean_localization_error"]) * int(metrics["tp"])
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    summary = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_localization_error": all_dist_weighted / total_tp if total_tp else None,
        "per_micrograph": per_micrograph,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    metrics = evaluate_manifest(config, args.pred_dir)
    output = Path(args.output) if args.output else Path(config["output"]["directory"]) / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_micrograph"}, indent=2))
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()

