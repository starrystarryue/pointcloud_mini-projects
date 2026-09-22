"""
借助skimage.measure.marching_cubes 重建mesh

输入
  outputs/sdf/<uid>.npy   
  outputs/sdf/<uid>.json 
输出
  outputs/mesh/<uid>.obj
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from skimage import measure


def load_bounds(args: argparse.Namespace) -> np.ndarray:
    if args.bounds is not None:
        values = np.asarray(args.bounds, dtype=np.float32)
        if values.size != 6:
            raise ValueError("--bounds expects six numbers: xmin ymin zmin xmax ymax zmax")
        return values.reshape(2, 3)

    if args.meta is not None and args.meta.exists():
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        bounds = np.asarray(meta["bounds"], dtype=np.float32)
        if bounds.shape == (2, 3):
            return bounds

    return np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float32)


def marching_cubes_from_sdf(
    sdf: np.ndarray,
    bounds: np.ndarray,
    iso_level: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    
    if sdf.ndim != 3:
        raise ValueError(f"SDF grid must be 3D, got shape {sdf.shape}")
    if not (float(sdf.min()) <= iso_level <= float(sdf.max())):
        raise ValueError(
            f"iso level {iso_level} is outside SDF range "
            f"[{float(sdf.min()):.6f}, {float(sdf.max()):.6f}]"
        )

    spacing = (bounds[1] - bounds[0]) / np.maximum(np.asarray(sdf.shape, dtype=np.float32) - 1.0, 1.0)
    vertices, faces, _, _ = measure.marching_cubes(
        sdf,
        level=iso_level,
        spacing=tuple(float(x) for x in spacing),
        gradient_direction="descent",
    )
    vertices = vertices.astype(np.float32) + bounds[0]
    faces = faces.astype(np.int32) + 1
    
    return vertices, faces


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# reconstructed SDF mesh by skimage.measure.marching_cubes\n")
        for v in vertices:
            f.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OBJ mesh from a saved SDF grid")
    parser.add_argument("--sdf", type=Path, required=True, help="input .npy SDF grid")
    parser.add_argument("--out", type=Path, required=True, help="output .obj path")
    parser.add_argument("--meta", type=Path, default=None, help="metadata JSON saved by reconstruct.py")
    parser.add_argument("--bounds", nargs=6, type=float, default=None)
    parser.add_argument("--iso-level", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sdf = np.load(args.sdf).astype(np.float32)
    bounds = load_bounds(args)
    vertices, faces = marching_cubes_from_sdf(sdf, bounds, args.iso_level)
    write_obj(args.out, vertices, faces)
    print(f"saved {args.out}: {len(vertices)} vertices, {len(faces)} faces")


if __name__ == "__main__":
    main()
