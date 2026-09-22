import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from PointNet.dataset import ModelNet40Dataset
    from PointNet.dataset_new import ModelNet40NormalResampledDataset
    from PointNet.model import PointNetCls
except ModuleNotFoundError:
    from dataset import ModelNet40Dataset
    from dataset_new import ModelNet40NormalResampledDataset
    from model import PointNetCls


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PointNet checkpoint")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/pointnet/best_model.pth")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dataset", choices=["off", "normal_resampled"], default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--num_points", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_method", choices=["surface", "vertex"], default=None)
    parser.add_argument("--use_normals", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="./eval_results/pointnet")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    data_path = args.data_path or ckpt_args.get("data_path", "./ModelNet40")
    dataset_type = args.dataset or ckpt_args.get("dataset", "off")
    num_points = args.num_points or ckpt_args.get("num_points", 1024)
    sample_method = args.sample_method or ckpt_args.get("sample_method", "surface")
    use_normals = args.use_normals or ckpt_args.get("use_normals", False)
    classes = checkpoint.get("classes")

    if dataset_type == "normal_resampled":
        dataset = ModelNet40NormalResampledDataset(
            root_dir=data_path,
            split=args.split,
            num_points=num_points,
            augment=False,
            use_normals=use_normals,
            seed=args.seed,
            max_samples=args.max_samples,
        )
    else:
        dataset = ModelNet40Dataset(
            root_dir=data_path,
            split=args.split,
            num_points=num_points,
            sample_method=sample_method,
            augment=False,
            seed=args.seed,
            max_samples=args.max_samples,
        )
    if classes is None:
        classes = dataset.classes

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    input_channels = 6 if use_normals else 3
    model = PointNetCls(num_classes=len(classes), input_channels=input_channels, feature_transform=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_correct = 0
    total_seen = 0
    class_correct = torch.zeros(len(classes), dtype=torch.long)
    class_seen = torch.zeros(len(classes), dtype=torch.long)
    predictions_rows = []

    with torch.no_grad():
        for batch_index, (points, labels) in enumerate(tqdm(loader, desc="Evaluate")):
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _, _ = model(points)
            predictions = logits.argmax(dim=1)

            total_seen += labels.size(0)
            total_correct += predictions.eq(labels).sum().item()

            for label, prediction in zip(labels.cpu(), predictions.cpu()):
                class_seen[label] += 1
                class_correct[label] += int(label == prediction)
                predictions_rows.append(
                    {
                        "label": classes[label.item()],
                        "prediction": classes[prediction.item()],
                        "correct": int(label.item() == prediction.item()),
                    }
                )

    overall_acc = 100.0 * total_correct / total_seen
    class_acc = torch.where(class_seen > 0, class_correct.float() / class_seen.float(), torch.zeros_like(class_seen.float()))
    mean_class_acc = 100.0 * class_acc.mean().item()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_samples": total_seen,
        "overall_accuracy": overall_acc,
        "mean_class_accuracy": mean_class_acc,
        "class_accuracy": {
            classes[index]: 100.0 * class_acc[index].item() for index in range(len(classes))
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(predictions_rows)

    print(f"Overall accuracy: {overall_acc:.2f}%")
    print(f"Mean class accuracy: {mean_class_acc:.2f}%")
    print(f"Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
