import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_ssg import ModelNet40SSGDataset
from train_ssg import PointNet2SSGCls


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PointNet++ SSG checkpoint")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/pointnet2_ssg/best_model.pth")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--num_points", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="./eval_results/pointnet2_ssg")
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

    data_path = args.data_path or ckpt_args.get("data_path", "./modelnet40_normal_resampled")
    num_points = args.num_points or ckpt_args.get("num_points", 1024)
    classes = checkpoint.get("classes")

    dataset = ModelNet40SSGDataset(
        root_dir=data_path,
        split=args.split,
        num_points=num_points,
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

    model = PointNet2SSGCls(num_classes=len(classes), dropout=ckpt_args.get("dropout", 0.5)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    class_correct = torch.zeros(len(classes), dtype=torch.long)
    class_seen = torch.zeros(len(classes), dtype=torch.long)
    predictions_rows = []

    with torch.no_grad():
        for points, labels in tqdm(loader, desc="Evaluate"):
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(points)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_seen += batch_size
            total_loss += loss.item() * batch_size
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

    avg_loss = total_loss / total_seen
    overall_acc = 100.0 * total_correct / total_seen
    class_acc = torch.where(
        class_seen > 0,
        class_correct.float() / class_seen.float(),
        torch.zeros_like(class_seen.float()),
    )
    mean_class_acc = 100.0 * class_acc.mean().item()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_samples": total_seen,
        "loss": avg_loss,
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

    print(f"Loss: {avg_loss:.4f}")
    print(f"Overall accuracy: {overall_acc:.2f}%")
    print(f"Mean class accuracy: {mean_class_acc:.2f}%")
    print(f"Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
