"""
服务器：RTX PRO 6000

基础：
python train.py --fourier-features 0

Fourier features:
python train.py --checkpoint-dir outputs/checkpoints_Fourier --log-dir outputs/logs_Fourier --fourier-features 6  --fourier-scale 1.0   
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sdf_model import checkpoint_payload, compute_bounds, load_sdf_data, make_model, shape_uids


def train_one(uid: str, args: argparse.Namespace, device: torch.device) -> Dict[str, float]:
    shape_dir = args.data_dir / uid
    points_np, sdf_np, grad_np = load_sdf_data(shape_dir)
    bounds = compute_bounds(points_np, args.bounds_margin)

    points_all = torch.from_numpy(points_np).to(device)
    sdf_all = torch.from_numpy(sdf_np).to(device)
    grad_all = torch.from_numpy(grad_np).to(device)
    n_samples = points_all.shape[0]
    steps = args.steps
    if steps <= 0:
        steps = args.epochs * math.ceil(n_samples / args.batch_size)

    model = make_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"[config] {uid} hidden_dim={args.hidden_dim} layers={args.layers} "
        f"fourier_features={args.fourier_features} fourier_scale={args.fourier_scale}"
    )

    start = time.perf_counter()
    wall_start = time.time()
    step_times: List[float] = []
    loss_history: List[float] = []
    sdf_loss_history: List[float] = []
    grad_loss_history: List[float] = []
    final_loss = 0.0
    final_sdf_loss = 0.0
    final_grad_loss = 0.0
    model.train()
    progress = tqdm(
        range(1, steps + 1),
        desc=f"train {uid}",
        dynamic_ncols=True,
        leave=True,
    )
    for step in progress:
        step_start = time.perf_counter()
        idx = torch.randint(0, n_samples, (args.batch_size,), device=device)
        pts = points_all[idx].detach().clone().requires_grad_(True)
        sdf = sdf_all[idx]
        grad = grad_all[idx]

        pred = model(pts)
        sdf_loss = F.mse_loss(pred, sdf)
        pred_grad = torch.autograd.grad(
            pred.sum(),
            pts,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_loss = F.mse_loss(pred_grad, grad)
        loss = sdf_loss + args.lambda_grad * grad_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        final_loss = float(loss.detach())
        final_sdf_loss = float(sdf_loss.detach())
        final_grad_loss = float(grad_loss.detach())
        loss_history.append(final_loss)
        sdf_loss_history.append(final_sdf_loss)
        grad_loss_history.append(final_grad_loss)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.checkpoint_dir / f"{uid}.pt"
    torch.save(checkpoint_payload(model, args, bounds), ckpt_path)

    total_train_sec = time.perf_counter() - start
    avg_step_sec = sum(step_times) / len(step_times)
    stats = {
        "uid": uid,
        "steps": steps,
        "batch_size": args.batch_size,
        "n_samples": int(n_samples),
        "checkpoint": str(ckpt_path),
        "total_train_sec": total_train_sec,
        "avg_step_sec": avg_step_sec,
        "min_step_sec": min(step_times),
        "max_step_sec": max(step_times),
        "final_loss": final_loss,
        "final_sdf_loss": final_sdf_loss,
        "final_grad_loss": final_grad_loss,
        "wall_start_time": wall_start,
        "wall_end_time": time.time(),
        "step_times_sec": step_times,
        "loss_history": loss_history,
        "sdf_loss_history": sdf_loss_history,
        "grad_loss_history": grad_loss_history,
    }

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"train_{uid}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    tqdm.write(
        f"[train] {uid} saved {ckpt_path} total={total_train_sec:.2f}s "
        f"avg_step={avg_step_sec:.4f}s final_loss={final_loss:.6f} "
        f"final_sdf={final_sdf_loss:.6f} final_grad={final_grad_loss:.6f} log={log_path}"
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one SDF MLP per shape")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/logs"))
    parser.add_argument("--uids", nargs="*", default=None, help="shape ids; default: all folders in data-dir")

    parser.add_argument("--steps", type=int, default=20000, help="random mini-batch updates per shape; <=0 uses epochs")
    parser.add_argument("--epochs", type=int, default=50, help="used only when --steps <= 0")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-grad", type=float, default=0.1)

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    
    parser.add_argument("--fourier-features", type=int, default=0)
    parser.add_argument("--fourier-scale", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--bounds-margin", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.log_dir = args.log_dir.resolve()

    device = torch.device(args.device)
    uids = shape_uids(args.data_dir, args.uids)
    if not uids:
        raise RuntimeError(f"no shapes found in {args.data_dir}")

    print(f"device={device}, train shapes={len(uids)}")
    all_start = time.perf_counter()
    all_stats = []
    for uid in uids:
        all_stats.append(train_one(uid, args, device))

    total_sec = time.perf_counter() - all_start
    summary = {
        "num_shapes": len(all_stats),
        "total_train_sec": total_sec,
        "avg_shape_train_sec": total_sec / max(len(all_stats), 1),
        "avg_step_sec": sum(s["avg_step_sec"] for s in all_stats) / max(len(all_stats), 1),
        "avg_final_loss": sum(s["final_loss"] for s in all_stats) / max(len(all_stats), 1),
        "avg_final_sdf_loss": sum(s["final_sdf_loss"] for s in all_stats) / max(len(all_stats), 1),
        "avg_final_grad_loss": sum(s["final_grad_loss"] for s in all_stats) / max(len(all_stats), 1),
        "shapes": all_stats,
    }
    args.log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.log_dir / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[train] all shapes done total={total_sec:.2f}s "
        f"avg_shape={summary['avg_shape_train_sec']:.2f}s "
        f"avg_step={summary['avg_step_sec']:.4f}s "
        f"avg_final_loss={summary['avg_final_loss']:.6f} "
        f"avg_final_sdf={summary['avg_final_sdf_loss']:.6f} "
        f"avg_final_grad={summary['avg_final_grad_loss']:.6f} "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
