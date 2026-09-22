from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, num_frequencies: int = 0, scale: float = 1.0) -> None:
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        if self.num_frequencies > 0:
            freq = (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)) * scale
            self.register_buffer("freq", freq)
        else:
            self.register_buffer("freq", torch.empty(0))

    @property
    def out_dim(self) -> int:
        return 3 if self.num_frequencies <= 0 else 3 + 3 * 2 * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_frequencies <= 0:
            return x
        xb = x[..., None, :] * self.freq[:, None] * math.pi
        enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-2).flatten(start_dim=-2)
        return torch.cat([x, enc], dim=-1)


class SDFMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 6,
        fourier_features: int = 0,
        fourier_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = FourierFeatures(fourier_features, fourier_scale)
        layers: List[nn.Module] = []
        in_dim = self.encoder.out_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoder(x)).squeeze(-1)


def shape_uids(data_dir: Path, selected: Optional[List[str]] = None) -> List[str]:
    if selected:
        return selected
    return sorted(p.name for p in data_dir.iterdir() if p.is_dir() and (p / "sdf.npz").exists())


def load_sdf_data(shape_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(shape_dir / "sdf.npz")
    points = data["points"].astype(np.float32)
    sdf = data["sdf"].astype(np.float32).reshape(-1)
    grad = data["grad"].astype(np.float32)
    return points, sdf, grad


def compute_bounds(points: np.ndarray, margin: float = 0.05) -> np.ndarray:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = hi - lo
    pad = np.maximum(span * margin, 1e-3)
    return np.stack([lo - pad, hi + pad], axis=0).astype(np.float32)


def make_model(args: argparse.Namespace) -> SDFMLP:
    return SDFMLP(
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        fourier_features=args.fourier_features,
        fourier_scale=args.fourier_scale,
    )


def checkpoint_payload(model: SDFMLP, args: argparse.Namespace, bounds: np.ndarray) -> dict:
    return {
        "model": model.state_dict(),
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "fourier_features": args.fourier_features,
        "fourier_scale": args.fourier_scale,
        "bounds": bounds.tolist(),
    }


def load_checkpoint_model(
    uid: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[SDFMLP, np.ndarray]:
    ckpt_path = args.checkpoint_dir / f"{uid}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=device)

    model_args = argparse.Namespace(
        hidden_dim=int(payload.get("hidden_dim", args.hidden_dim)),
        layers=int(payload.get("layers", args.layers)),
        fourier_features=int(payload.get("fourier_features", args.fourier_features)),
        fourier_scale=float(payload.get("fourier_scale", args.fourier_scale)),
    )
    print(
        f"[config] {uid} loaded checkpoint hidden_dim={model_args.hidden_dim} "
        f"layers={model_args.layers} fourier_features={model_args.fourier_features} "
        f"fourier_scale={model_args.fourier_scale}"
    )
    model = make_model(model_args).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    bounds = np.asarray(payload.get("bounds"), dtype=np.float32)
    if bounds.shape != (2, 3):
        points_np, _, _ = load_sdf_data(args.data_dir / uid)
        bounds = compute_bounds(points_np, args.bounds_margin)
    return model, bounds


def grid_points(bounds: np.ndarray, resolution: int) -> np.ndarray:
    axes = [np.linspace(bounds[0, i], bounds[1, i], resolution, dtype=np.float32) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
