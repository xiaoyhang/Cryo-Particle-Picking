"""Train the U-Net particle-center detector."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_processing import CryoParticleDataset, load_config, read_manifest, seed_everything
from unet import UNet


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probability = torch.sigmoid(logits)
        pt = probability * targets + (1 - probability) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataset(config: dict, split: str, augment: bool) -> CryoParticleDataset:
    data = config["data"]
    return CryoParticleDataset(
        read_manifest(data["manifest"], split),
        patch_size=int(data["patch_size"]),
        patches_per_micrograph=int(data["patches_per_micrograph"]),
        heatmap_sigma=float(data["heatmap_sigma"]),
        positive_crop_ratio=float(data["positive_crop_ratio"]),
        augment=augment,
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        use_amp = device.type == "cuda"
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=use_amp
        ):
            loss = loss_fn(model(images), targets)
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach()) * images.shape[0]
    return total / len(loader.dataset)


def train(config_path: str) -> Path:
    config = load_config(config_path)
    train_cfg = config["training"]
    seed_everything(int(train_cfg["seed"]))
    device = choose_device(str(train_cfg.get("device", "auto")))
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_set = make_dataset(config, "train", bool(config["data"].get("augment", True)))
    val_set = make_dataset(config, "val", False)
    loader_args = dict(
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = UNet(base_channels=int(config["model"]["base_channels"])).to(device)
    if train_cfg["loss"] == "focal":
        loss_fn = FocalLoss(float(train_cfg["focal_alpha"]), float(train_cfg["focal_gamma"]))
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_epoch, best_loss = 1, float("inf")
    resume = train_cfg.get("resume")
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state.get("best_val_loss", best_loss))

    history_path = output_dir / "training_history.csv"
    with history_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if history_path.stat().st_size == 0:
            writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])
        for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
            train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler)
            val_loss = run_epoch(model, val_loader, loss_fn, device)
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, train_loss, val_loss, lr])
            handle.flush()
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": min(best_loss, val_loss),
                "config": config,
            }
            torch.save(state, output_dir / "last_model.pt")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(state, output_dir / "best_model.pt")
            print(
                f"Epoch {epoch:03d} | train={train_loss:.6f} "
                f"val={val_loss:.6f} lr={lr:.2e}"
            )
    return output_dir / "best_model.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    checkpoint = train(args.config)
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

