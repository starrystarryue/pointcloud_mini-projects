from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def read_off(path):
    """Read vertices and triangular faces from an OFF mesh."""

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        lines = [line.strip() for line in file if line.strip() and not line.startswith("#")]

    if not lines:
        raise ValueError(f"{path} is empty")

    first_line = lines[0]
    if not first_line.startswith("OFF"):
        raise ValueError(f"{path} is not a valid OFF file")

    counts = first_line[3:].strip().split()
    start = 1
    if len(counts) < 3:
        counts.extend(lines[1].split())
        start = 2

    num_vertices = int(counts[0])
    num_faces = int(counts[1])

    vertices = np.asarray(
        [list(map(float, lines[start + i].split()[:3])) for i in range(num_vertices)],
        dtype=np.float32,
    )
    finite_vertices = np.isfinite(vertices).all(axis=1)
    if not finite_vertices.all():
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[finite_vertices] = np.arange(finite_vertices.sum())
        vertices = vertices[finite_vertices]
    else:
        remap = None

    faces = []
    face_start = start + num_vertices
    for i in range(num_faces):
        parts = lines[face_start + i].split()
        if not parts:
            continue
        n = int(parts[0])
        indices = list(map(int, parts[1 : 1 + n]))
        if n < 3:
            continue
        if remap is not None:
            indices = [remap[index] for index in indices if 0 <= index < len(remap)]
            if len(indices) < 3 or any(index < 0 for index in indices):
                continue
        for j in range(1, n - 1):
            faces.append([indices[0], indices[j], indices[j + 1]])

    faces = np.asarray(faces, dtype=np.int64) if faces else np.empty((0, 3), dtype=np.int64)
    if len(vertices) == 0:
        raise ValueError(f"{path} has no finite vertices")
    return vertices, faces


def normalize_unit_sphere(points):
    points = points.astype(np.float64, copy=False)
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points, axis=1).max()
    if np.isfinite(scale) and scale > 0:
        points = points / scale
    else:
        points = np.zeros_like(points)
    return points.astype(np.float32)


def sample_vertices(vertices, num_points, rng):
    vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if len(vertices) == 0:
        raise ValueError("Cannot sample vertices from an empty finite vertex set")
    replace = len(vertices) < num_points
    indices = rng.choice(len(vertices), size=num_points, replace=replace)
    return vertices[indices]


def sample_surface(vertices, faces, num_points, rng):
    if len(faces) == 0:
        return sample_vertices(vertices, num_points, rng)

    valid_faces = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    faces = faces[valid_faces]
    if len(faces) == 0:
        return sample_vertices(vertices, num_points, rng)

    triangles = vertices[faces].astype(np.float64, copy=False)
    vec_a = triangles[:, 1] - triangles[:, 0]
    vec_b = triangles[:, 2] - triangles[:, 0]
    areas = np.linalg.norm(np.cross(vec_a, vec_b), axis=1) * 0.5

    area_sum = areas.sum()
    if not np.isfinite(area_sum) or area_sum <= 0 or not np.isfinite(areas).all():
        return sample_vertices(vertices, num_points, rng)

    probabilities = areas / area_sum
    tri_indices = rng.choice(len(faces), size=num_points, replace=True, p=probabilities)
    chosen = triangles[tri_indices]

    r1 = np.sqrt(rng.random(num_points, dtype=np.float32))
    r2 = rng.random(num_points, dtype=np.float32)
    points = (
        (1.0 - r1)[:, None] * chosen[:, 0]
        + (r1 * (1.0 - r2))[:, None] * chosen[:, 1]
        + (r1 * r2)[:, None] * chosen[:, 2]
    )
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


class ModelNet40Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        num_points=1024,
        sample_method="surface",
        augment=False,
        seed=0,
        cache_size=15000,
        max_samples=None,
        max_samples_per_class=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_points = num_points
        self.sample_method = sample_method
        self.augment = augment
        self.seed = seed
        self.cache_size = cache_size
        self.cache = {}

        if sample_method not in {"surface", "vertex"}:
            raise ValueError("sample_method must be 'surface' or 'vertex'")

        self.classes = sorted(
            item.name for item in self.root_dir.iterdir() if item.is_dir()
        )
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name / split
            if not class_dir.exists():
                continue
            paths = sorted(class_dir.glob("*.off"))
            if max_samples_per_class is not None:
                paths = paths[:max_samples_per_class]
            for path in paths:
                self.samples.append((path, self.class_to_idx[class_name]))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise RuntimeError(f"No OFF files found in {self.root_dir} for split={split}")

    def __len__(self):
        return len(self.samples)

    def _load_mesh(self, path):
        if path in self.cache:
            return self.cache[path]

        mesh = read_off(path)
        if self.cache_size > 0 and len(self.cache) < self.cache_size:
            self.cache[path] = mesh
        return mesh

    def __getitem__(self, index):
        path, label = self.samples[index]
        vertices, faces = self._load_mesh(path)

        rng = np.random.default_rng() if self.augment else np.random.default_rng(self.seed + index)
        if self.sample_method == "surface":
            points = sample_surface(vertices, faces, self.num_points, rng)
        else:
            points = sample_vertices(vertices, self.num_points, rng)

        points = normalize_unit_sphere(points)
        if self.augment:
            points = rotate_point_cloud_y(points, rng)
            points = jitter_point_cloud(points, rng)

        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite sampled points from {path}")

        return torch.from_numpy(points.astype(np.float32)), torch.tensor(label, dtype=torch.long)


def create_dataloaders(
    root_dir,
    batch_size=32,
    num_points=1024,
    num_workers=4,
    sample_method="surface",
    seed=0,
    cache_size=15000,
    max_train_samples=None,
    max_test_samples=None,
    max_train_samples_per_class=None,
    max_test_samples_per_class=None,
):
    train_set = ModelNet40Dataset(
        root_dir=root_dir,
        split="train",
        num_points=num_points,
        sample_method=sample_method,
        augment=True,
        seed=seed,
        cache_size=cache_size,
        max_samples=max_train_samples,
        max_samples_per_class=max_train_samples_per_class,
    )
    test_set = ModelNet40Dataset(
        root_dir=root_dir,
        split="test",
        num_points=num_points,
        sample_method=sample_method,
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


def dataloader(root_dir, batch_size=32, num_points=1024, num_workers=4):
    train_loader, test_loader, classes = create_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        num_points=num_points,
        num_workers=num_workers,
    )
    return train_loader, test_loader, len(classes)
