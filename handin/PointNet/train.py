import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from PointNet.dataset import create_dataloaders as create_off_dataloaders
    from PointNet.dataset_new import create_dataloaders as create_normal_resampled_dataloaders
    from PointNet.model import PointNetCls, feature_transform_regularizer
except ModuleNotFoundError:
    from dataset import create_dataloaders as create_off_dataloaders
    from dataset_new import create_dataloaders as create_normal_resampled_dataloaders
    from model import PointNetCls, feature_transform_regularizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def pointnet_loss(logits, target, trans_feat, reg_weight=0.001):
    cls_loss = nn.CrossEntropyLoss()(logits, target)
    if trans_feat is None:
        reg_loss = logits.new_tensor(0.0)
    else:
        reg_loss = feature_transform_regularizer(trans_feat)
    return cls_loss + reg_weight * reg_loss, cls_loss, reg_loss


def run_epoch(model, loader, optimizer, device, reg_weight, desc, debug_device=False):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_correct = 0
    total_seen = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        progress = tqdm(loader, desc=desc, leave=False)
        for batch_index, (points, labels) in enumerate(progress):
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, _, trans_feat = model(points)
            loss, cls_loss, reg_loss = pointnet_loss(logits, labels, trans_feat, reg_weight)
            if debug_device and batch_index == 0:
                print(
                    f"{desc} device check: "
                    f"points={points.device}, labels={labels.device}, "
                    f"logits={logits.device}, loss={loss.device}"
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in {desc}, batch {batch_index}: "
                    f"loss={loss.item()}, cls_loss={cls_loss.item()}, reg_loss={reg_loss.item()}, "
                    f"logits_finite={torch.isfinite(logits).all().item()}"
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            predictions = logits.argmax(dim=1)
            total_seen += batch_size
            total_correct += predictions.eq(labels).sum().item()
            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            total_reg_loss += reg_loss.item() * batch_size

            progress.set_postfix(
                loss=f"{total_loss / total_seen:.4f}",
                acc=f"{100.0 * total_correct / total_seen:.2f}%",
            )

    return {
        "loss": total_loss / total_seen,
        "cls_loss": total_cls_loss / total_seen,
        "reg_loss": total_reg_loss / total_seen,
        "acc": 100.0 * total_correct / total_seen,
    }


def save_history(history, out_dir):
    if not history:
        return
    csv_path = out_dir / "history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_acc, classes, args, history):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "classes": classes,
            "args": vars(args),
            "history": history,
        },
        path,
    )


def write_tensorboard(writer, epoch, current_lr, train_stats, test_stats, best_acc):
    writer.add_scalar("Loss/train", train_stats["loss"], epoch)
    writer.add_scalar("Loss/test", test_stats["loss"], epoch)
    writer.add_scalar("Loss_classification/train", train_stats["cls_loss"], epoch)
    writer.add_scalar("Loss_classification/test", test_stats["cls_loss"], epoch)
    writer.add_scalar("Loss_regularization/train", train_stats["reg_loss"], epoch)
    writer.add_scalar("Loss_regularization/test", test_stats["reg_loss"], epoch)
    writer.add_scalar("Accuracy/train", train_stats["acc"], epoch)
    writer.add_scalar("Accuracy/test", test_stats["acc"], epoch)
    writer.add_scalar("Accuracy/best_test", best_acc, epoch)
    writer.add_scalar("Learning_rate", current_lr, epoch)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PointNet on ModelNet40")
    parser.add_argument("--data_path", type=str, default="./ModelNet40")
    parser.add_argument("--dataset", choices=["off", "normal_resampled"], default="off")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/pointnet")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--reg_weight", type=float, default=0.001)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_method", choices=["surface", "vertex"], default="surface")
    parser.add_argument("--use_normals", action="store_true")
    parser.add_argument("--cache_size", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--max_train_samples_per_class", type=int, default=None)
    parser.add_argument("--max_test_samples_per_class", type=int, default=None)
    parser.add_argument("--debug_device", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else out_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(log_dir))
    with open(out_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)
    print(f"TensorBoard logs: {log_dir.resolve()}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.dataset == "normal_resampled":
        train_loader, test_loader, classes = create_normal_resampled_dataloaders(
            root_dir=args.data_path,
            batch_size=args.batch_size,
            num_points=args.num_points,
            num_workers=args.num_workers,
            seed=args.seed,
            cache_size=args.cache_size,
            use_normals=args.use_normals,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            max_train_samples_per_class=args.max_train_samples_per_class,
            max_test_samples_per_class=args.max_test_samples_per_class,
        )
    else:
        train_loader, test_loader, classes = create_off_dataloaders(
            root_dir=args.data_path,
            batch_size=args.batch_size,
            num_points=args.num_points,
            num_workers=args.num_workers,
            sample_method=args.sample_method,
            seed=args.seed,
            cache_size=args.cache_size,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            max_train_samples_per_class=args.max_train_samples_per_class,
            max_test_samples_per_class=args.max_test_samples_per_class,
        )
    print(f"Classes: {len(classes)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    input_channels = 6 if args.use_normals else 3
    model = PointNetCls(
        num_classes=len(classes),
        input_channels=input_channels,
        dropout=args.dropout,
        feature_transform=True,
    ).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    if args.debug_device:
        print(f"Model parameter device: {next(model.parameters()).device}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    start_epoch = 1
    best_acc = 0.0
    history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        history = checkpoint.get("history", [])
        print(f"Resumed from epoch {checkpoint['epoch']} with best acc {best_acc:.2f}%")

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.reg_weight,
            desc=f"Epoch {epoch:03d}/{args.epochs} train",
            debug_device=args.debug_device and epoch == start_epoch,
        )
        test_stats = run_epoch(
            model,
            test_loader,
            optimizer=None,
            device=device,
            reg_weight=args.reg_weight,
            desc=f"Epoch {epoch:03d}/{args.epochs} test",
            debug_device=args.debug_device and epoch == start_epoch,
        )

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_stats["loss"],
            "train_cls_loss": train_stats["cls_loss"],
            "train_reg_loss": train_stats["reg_loss"],
            "train_acc": train_stats["acc"],
            "test_loss": test_stats["loss"],
            "test_cls_loss": test_stats["cls_loss"],
            "test_reg_loss": test_stats["reg_loss"],
            "test_acc": test_stats["acc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}: "
            f"train loss {train_stats['loss']:.4f}, train acc {train_stats['acc']:.2f}%, "
            f"test loss {test_stats['loss']:.4f}, test acc {test_stats['acc']:.2f}%, "
            f"lr {current_lr:.6f}"
        )

        if test_stats["acc"] > best_acc:
            best_acc = test_stats["acc"]
            save_checkpoint(
                out_dir / "best_model.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                best_acc,
                classes,
                args,
                history,
            )
            print(f"Saved best model: {best_acc:.2f}%")

        if args.save_interval > 0 and epoch % args.save_interval == 0:
            save_checkpoint(
                out_dir / f"checkpoint_epoch_{epoch}.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                best_acc,
                classes,
                args,
                history,
            )

        write_tensorboard(writer, epoch, current_lr, train_stats, test_stats, best_acc)
        writer.flush()
        save_history(history, out_dir)
        scheduler.step()

    save_checkpoint(
        out_dir / "final_model.pth",
        args.epochs,
        model,
        optimizer,
        scheduler,
        best_acc,
        classes,
        args,
        history,
    )
    elapsed = (time.time() - start_time) / 3600.0
    print(f"Training finished in {elapsed:.2f} hours. Best test accuracy: {best_acc:.2f}%")
    print(f"Artifacts saved to: {os.path.abspath(out_dir)}")
    writer.close()


if __name__ == "__main__":
    main()
