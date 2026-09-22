from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def normalize_unit_sphere(points):
    points = points.astype(np.float32, copy=False)
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points.astype(np.float32)


def rotate_point_cloud_y(points, rng):
    angle = rng.random() * 2.0 * np.pi
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation = np.array(
        [[cosval, 0.0, sinval], [0.0, 1.0, 0.0], [-sinval, 0.0, cosval]],
        dtype=np.float32,
    )
    return points @ rotation.T


def jitter_point_cloud(points, rng, sigma=0.01, clip=0.05):
    noise = rng.normal(0.0, sigma, size=points.shape)
    noise = np.clip(noise, -clip, clip).astype(np.float32)
    return points + noise


class ModelNet40SSGDataset(Dataset):
    """modelnet40_normal_resampled loader for PointNet++ SSG classification.

    The dataset split files contain shape ids such as ``airplane_0001``.
    Each shape txt file is an N x 6 comma-separated matrix: xyz + normal.
    The original PointNet++ SSG classification model uses xyz input, so this
    dataset returns only the first three columns.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        num_points=1024,
        augment=False,
        seed=0,
        cache_size=15000,
        max_samples=None,
        max_samples_per_class=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.seed = seed
        self.cache_size = cache_size
        self.cache = {}

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        shape_names_path = self.root_dir / "modelnet40_shape_names.txt"
        if not shape_names_path.exists():
            raise FileNotFoundError(f"Missing class name file: {shape_names_path}")
        self.classes = [
            line.strip()
            for line in shape_names_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

        split_path = self.root_dir / f"modelnet40_{split}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        shape_ids = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        per_class_counts = {class_name: 0 for class_name in self.classes}
        self.samples = []
        for shape_id in shape_ids:
            class_name = self.class_name_from_shape_id(shape_id)
            if class_name not in self.class_to_idx:
                raise ValueError(f"Unknown class '{class_name}' from shape id '{shape_id}'")
            if max_samples_per_class is not None:
                if per_class_counts[class_name] >= max_samples_per_class:
                    continue
                per_class_counts[class_name] += 1

            path = self.root_dir / class_name / f"{shape_id}.txt"
            if not path.exists():
                raise FileNotFoundError(f"Missing shape file: {path}")
            self.samples.append((path, self.class_to_idx[class_name]))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError(f"No samples found in {self.root_dir} for split={split}")

    @staticmethod
    def class_name_from_shape_id(shape_id):
        parts = shape_id.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"Invalid ModelNet40 shape id: {shape_id}")
        return parts[0]

    def __len__(self):
        return len(self.samples)

    def _load_points(self, path):
        if path in self.cache:
            return self.cache[path]

        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            raise ValueError(f"{path} must contain at least xyz columns")

        points = data[:, :3]
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            raise ValueError(f"{path} has no finite xyz points")

        if self.cache_size > 0 and len(self.cache) < self.cache_size:
            self.cache[path] = points
        return points

    def __getitem__(self, index):
        path, label = self.samples[index]
        points = self._load_points(path)

        rng = np.random.default_rng() if self.augment else np.random.default_rng(self.seed + index)
        replace = len(points) < self.num_points
        choice = rng.choice(len(points), size=self.num_points, replace=replace)
        points = points[choice].copy()

        points = normalize_unit_sphere(points)
        if self.augment:
            points = rotate_point_cloud_y(points, rng)
            points = jitter_point_cloud(points, rng)

        return torch.from_numpy(points.astype(np.float32)), torch.tensor(label, dtype=torch.long)


def create_dataloaders(
    root_dir,
    batch_size=16,
    num_points=1024,
    num_workers=4,
    seed=0,
    cache_size=15000,
    max_train_samples=None,
    max_test_samples=None,
    max_train_samples_per_class=None,
    max_test_samples_per_class=None,
):
    train_set = ModelNet40SSGDataset(
        root_dir=root_dir,
        split="train",
        num_points=num_points,
        augment=True,
        seed=seed,
        cache_size=cache_size,
        max_samples=max_train_samples,
        max_samples_per_class=max_train_samples_per_class,
    )
    test_set = ModelNet40SSGDataset(
        root_dir=root_dir,
        split="test",
        num_points=num_points,
        augment=False,
        seed=seed,
        cache_size=cache_size,
        max_samples=max_test_samples,
        max_samples_per_class=max_test_samples_per_class,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, train_set.classes
