'''
预测sdf并marshing cubes重建mesh：

基础：
python test.py --resolution 256

Fourier feature：
python test.py --resolution 256 --checkpoint-dir outputs/checkpoints_Fourier --mesh-dir outputs/mesh_Fourier --sdf-dir outputs/sdf_Fourier --fourier-features 6  --fourier-scale 1.0 
'''
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from sdf_model import grid_points, load_checkpoint_model, shape_uids


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_one(uid: str, args: argparse.Namespace, device: torch.device) -> Path:
    model, bounds = load_checkpoint_model(uid, args, device)
    pts = grid_points(bounds, args.resolution)
    values: List[np.ndarray] = []
    start = time.time()
    pred_start = time.perf_counter()

    for begin in range(0, len(pts), args.eval_batch_size):
        batch = torch.from_numpy(pts[begin : begin + args.eval_batch_size]).to(device)
        pred = model(batch).detach().cpu().numpy().astype(np.float32)
        values.append(pred)

    sync_if_cuda(device)
    pred_time = time.perf_counter() - pred_start
    sdf_grid = np.concatenate(values, axis=0).reshape(args.resolution, args.resolution, args.resolution)
    args.sdf_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = args.sdf_dir / f"{uid}.npy"
    np.save(sdf_path, sdf_grid)

    meta = {
        "uid": uid,
        "resolution": args.resolution,
        "bounds": bounds.tolist(),
        "axis_order": "ij",
        "iso_level": args.iso_level,
    }
    with open(args.sdf_dir / f"{uid}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[test] {uid} saved {sdf_path} ({time.time() - start:.1f}s)")
    print(f"[time] {uid} model SDF prediction: {pred_time:.3f}s")
    return sdf_path


def mesh_one(uid: str, args: argparse.Namespace) -> Path:
    args.mesh_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = args.sdf_dir / f"{uid}.npy"
    meta_path = args.sdf_dir / f"{uid}.json"
    out_path = args.mesh_dir / f"{uid}_npredict.obj"
    command = [
        sys.executable,
        str(Path(__file__).with_name("getmesh.py")),
        "--sdf",
        str(sdf_path),
        "--out",
        str(out_path),
        "--meta",
        str(meta_path),
        "--iso-level",
        str(args.iso_level),
    ]
    mesh_start = time.perf_counter()
    subprocess.run(command, check=True)
    mesh_time = time.perf_counter() - mesh_start
    print(f"[mesh] {uid} saved {out_path}")
    print(f"[time] {uid} marching cubes reconstruction: {mesh_time:.3f}s")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load trained SDF MLPs and reconstruct meshes")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--sdf-dir", type=Path, default=Path("outputs/sdf"))
    parser.add_argument("--mesh-dir", type=Path, default=Path("outputs/mesh"))
    parser.add_argument("--uids", nargs="*", default=None, help="shape ids; default: all folders in data-dir")

    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--bounds-margin", type=float, default=0.05)
    parser.add_argument("--iso-level", type=float, default=0.0)
    
    parser.add_argument("--no-mesh", action="store_true", help="保存sdf的预测结果，然后进入getmesh.py重建mesh")

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)

    parser.add_argument("--fourier-features", type=int, default=0)
    parser.add_argument("--fourier-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.sdf_dir = args.sdf_dir.resolve()
    args.mesh_dir = args.mesh_dir.resolve()

    device = torch.device(args.device)
    uids = shape_uids(args.data_dir, args.uids)
    if not uids:
        raise RuntimeError(f"no shapes found in {args.data_dir}")

    print(f"device={device}, test shapes={len(uids)}")
    for uid in uids:
        evaluate_one(uid, args, device)
        if not args.no_mesh:
            mesh_one(uid, args)


if __name__ == "__main__":
    main()
