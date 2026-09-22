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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_ssg import create_dataloaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def set_bn_momentum(model, momentum):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.momentum = momentum


def get_decay_value(base_value, decay_rate, decay_step, processed_samples, min_value=None):
    value = base_value * (decay_rate ** (processed_samples // decay_step))
    if min_value is not None:
        value = max(value, min_value)
    return value


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes):02d}m {seconds:05.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {seconds:05.2f}s"
    return f"{seconds:.2f}s"


def square_distance(src, dst):
    # src: [B, N, C], dst: [B, M, C]
    dist = -2.0 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


def index_points(points, idx):
    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.ones(batch_size, num_points, device=device) * 1e10
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    _, npoint, _ = new_xyz.shape
    group_idx = torch.arange(num_points, dtype=torch.long, device=device).view(1, 1, num_points)
    group_idx = group_idx.repeat(batch_size, npoint, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = num_points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(batch_size, npoint, 1).repeat(1, 1, nsample)
    group_idx[group_idx == num_points] = group_first[group_idx == num_points]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    batch_size, num_points, _ = xyz.shape
    new_xyz = torch.zeros(batch_size, 1, 3, device=xyz.device, dtype=xyz.dtype)
    grouped_xyz = xyz.view(batch_size, 1, num_points, 3)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(batch_size, 1, num_points, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        layers = []
        last_channels = in_channels + 3
        for out_channels in mlp:
            layers.append(nn.Conv2d(last_channels, out_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            last_channels = out_channels
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # [B, npoint, nsample, C] -> [B, C, nsample, npoint]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, dim=2)[0]
        new_points = new_points.transpose(1, 2).contiguous()
        return new_xyz, new_points


class PointNet2SSGCls(nn.Module):
    """Original PointNet++ SSG classification architecture for ModelNet40."""

    def __init__(self, num_classes=40, dropout=0.5):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channels=0,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channels=128,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channels=256,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, points):
        # points: [B, N, 3]
        xyz = points[:, :, :3].contiguous()
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.squeeze(1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x)), inplace=True))
        x = self.drop2(F.relu(self.bn2(self.fc2(x)), inplace=True))
        return self.fc3(x)


def run_epoch(model, loader, optimizer, device, desc, debug_device=False):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    criterion = nn.CrossEntropyLoss()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        progress = tqdm(loader, desc=desc, leave=False)
        for batch_index, (points, labels) in enumerate(progress):
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(points)
            loss = criterion(logits, labels)
            if debug_device and batch_index == 0:
                print(
                    f"{desc} device check: points={points.device}, labels={labels.device}, "
                    f"logits={logits.device}, loss={loss.device}"
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in {desc}, batch {batch_index}: "
                    f"loss={loss.item()}, logits_finite={torch.isfinite(logits).all().item()}"
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
            progress.set_postfix(
                loss=f"{total_loss / total_seen:.4f}",
                acc=f"{100.0 * total_correct / total_seen:.2f}%",
            )

    return {
        "loss": total_loss / total_seen,
        "acc": 100.0 * total_correct / total_seen,
    }


def save_history(history, out_dir):
    if not history:
        return
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    best_acc,
    classes,
    args,
    history,
    processed_samples,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "classes": classes,
            "args": vars(args),
            "history": history,
            "processed_samples": processed_samples,
        },
        path,
    )


def write_tensorboard(writer, epoch, current_lr, train_stats, test_stats, best_acc, timing_stats):
    writer.add_scalar("Loss/train", train_stats["loss"], epoch)
    writer.add_scalar("Loss/test", test_stats["loss"], epoch)
    writer.add_scalar("Accuracy/train", train_stats["acc"], epoch)
    writer.add_scalar("Accuracy/test", test_stats["acc"], epoch)
    writer.add_scalar("Accuracy/best_test", best_acc, epoch)
    writer.add_scalar("Learning_rate", current_lr, epoch)
    writer.add_scalar("Time/train_epoch_seconds", timing_stats["train_time_sec"], epoch)
    writer.add_scalar("Time/test_epoch_seconds", timing_stats["test_time_sec"], epoch)
    writer.add_scalar("Time/epoch_seconds", timing_stats["epoch_time_sec"], epoch)
    writer.add_scalar("Time/total_seconds", timing_stats["total_time_sec"], epoch)
    writer.add_scalar("Time/avg_epoch_seconds", timing_stats["avg_epoch_time_sec"], epoch)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PointNet++ SSG on ModelNet40")
    parser.add_argument("--data_path", type=str, default="./modelnet40_normal_resampled")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/pointnet2_ssg")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=251)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--decay_step", type=int, default=200000)
    parser.add_argument("--decay_rate", type=float, default=0.7)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--bn_init_momentum", type=float, default=0.5)
    parser.add_argument("--bn_decay_rate", type=float, default=0.5)
    parser.add_argument("--bn_min_momentum", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
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

    train_loader, test_loader, classes = create_dataloaders(
        root_dir=args.data_path,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_workers=args.num_workers,
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

    model = PointNet2SSGCls(num_classes=len(classes), dropout=args.dropout).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    if args.debug_device:
        print(f"Model parameter device: {next(model.parameters()).device}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_acc = 0.0
    history = []
    processed_samples = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        history = checkpoint.get("history", [])
        processed_samples = checkpoint.get("processed_samples", len(train_loader.dataset) * (start_epoch - 1))
        print(f"Resumed from epoch {checkpoint['epoch']} with best acc {best_acc:.2f}%")

    run_start_time = time.perf_counter()
    completed_epochs_this_run = 0
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        current_lr = get_decay_value(
            args.lr,
            args.decay_rate,
            args.decay_step,
            processed_samples,
            min_value=args.min_lr,
        )
        current_bn_momentum = get_decay_value(
            args.bn_init_momentum,
            args.bn_decay_rate,
            args.decay_step,
            processed_samples,
            min_value=args.bn_min_momentum,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        set_bn_momentum(model, current_bn_momentum)

        synchronize_if_cuda(device)
        train_start_time = time.perf_counter()
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            desc=f"Epoch {epoch:03d}/{args.epochs} train",
            debug_device=args.debug_device and epoch == start_epoch,
        )
        synchronize_if_cuda(device)
        train_time_sec = time.perf_counter() - train_start_time

        test_start_time = time.perf_counter()
        test_stats = run_epoch(
            model,
            test_loader,
            optimizer=None,
            device=device,
            desc=f"Epoch {epoch:03d}/{args.epochs} test",
            debug_device=args.debug_device and epoch == start_epoch,
        )
        synchronize_if_cuda(device)
        test_time_sec = time.perf_counter() - test_start_time
        epoch_time_sec = time.perf_counter() - epoch_start_time
        completed_epochs_this_run += 1
        total_time_sec = time.perf_counter() - run_start_time
        avg_epoch_time_sec = total_time_sec / completed_epochs_this_run
        timing_stats = {
            "train_time_sec": train_time_sec,
            "test_time_sec": test_time_sec,
            "epoch_time_sec": epoch_time_sec,
            "total_time_sec": total_time_sec,
            "avg_epoch_time_sec": avg_epoch_time_sec,
        }

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "bn_momentum": current_bn_momentum,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "test_loss": test_stats["loss"],
            "test_acc": test_stats["acc"],
            **timing_stats,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}: "
            f"train loss {train_stats['loss']:.4f}, train acc {train_stats['acc']:.2f}%, "
            f"test loss {test_stats['loss']:.4f}, test acc {test_stats['acc']:.2f}%, "
            f"lr {current_lr:.6f}, bn momentum {current_bn_momentum:.4f}, "
            f"train time {format_duration(train_time_sec)}, "
            f"epoch time {format_duration(epoch_time_sec)}, "
            f"total time {format_duration(total_time_sec)}, "
            f"avg/epoch {format_duration(avg_epoch_time_sec)}"
        )

        processed_samples += len(train_loader.dataset)

        if test_stats["acc"] > best_acc:
            best_acc = test_stats["acc"]
            save_checkpoint(
                out_dir / "best_model.pth",
                epoch,
                model,
                optimizer,
                best_acc,
                classes,
                args,
                history,
                processed_samples,
            )
            print(f"Saved best model: {best_acc:.2f}%")

        if args.save_interval > 0 and epoch % args.save_interval == 0:
            save_checkpoint(
                out_dir / f"checkpoint_epoch_{epoch}.pth",
                epoch,
                model,
                optimizer,
                best_acc,
                classes,
                args,
                history,
                processed_samples,
            )

        write_tensorboard(writer, epoch, current_lr, train_stats, test_stats, best_acc, timing_stats)
        writer.add_scalar("BatchNorm/momentum", current_bn_momentum, epoch)
        writer.flush()
        save_history(history, out_dir)

    save_checkpoint(
        out_dir / "final_model.pth",
        args.epochs,
        model,
        optimizer,
        best_acc,
        classes,
        args,
        history,
        processed_samples,
    )
    elapsed_sec = time.perf_counter() - run_start_time
    avg_epoch_sec = elapsed_sec / max(completed_epochs_this_run, 1)
    print(
        f"Training finished in {format_duration(elapsed_sec)} "
        f"(avg/epoch {format_duration(avg_epoch_sec)}). Best test accuracy: {best_acc:.2f}%"
    )
    print(f"Artifacts saved to: {os.path.abspath(out_dir)}")
    writer.close()


if __name__ == "__main__":
    main()
