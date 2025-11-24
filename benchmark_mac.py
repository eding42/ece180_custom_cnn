#!/usr/bin/env python3
"""
Heavy PyTorch ML benchmark for big GPUs (e.g., A100, RTX 50-series, Apple M-series).

- Default dataset: Food-101 (auto-download, large, 101 classes, high-res).
- Other options: CIFAR-10, CIFAR-100 (auto-download), ImageNet (manual folder).
- Heavy models: ResNet-50 / ResNet-101 / Wide-ResNet-50-2.
- Mixed precision / AMP support (CUDA only).
- Measures:
    * Training images/sec
    * Inference images/sec
    * Validation accuracy
    * Combined benchmark score

Usage examples:

  # Default: Food-101, ResNet-50, AMP (CUDA only), decent batch size
  python heavy_ml_benchmark_food101_mac.py \
      --dataset food101 --batch-size 512 --epochs 2 --fp16 --workers 16

  # Heavier model:
  python heavy_ml_benchmark_food101_mac.py \
      --dataset food101 --model resnet101 --img-size 256 --batch-size 512 --fp16

  # Quick sanity run (limit samples):
  python heavy_ml_benchmark_food101_mac.py \
      --dataset food101 --epochs 1 --limit-train 20000 --limit-val 5000
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, datasets, models


# ------------------------
# Device helpers
# ------------------------
def get_device():
    """Pick best available device: CUDA -> MPS (Apple) -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_synchronize(device):
    """Synchronize current device (for accurate timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    # CPU: nothing to do


# ------------------------
# Train / eval helpers
# ------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, use_amp=False):
    model.train()

    # AMP + GradScaler are CUDA-only here.
    amp_enabled = use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    start = time.perf_counter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    device_synchronize(device)
    end = time.perf_counter()

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    images_per_sec = total_samples / (end - start)

    return avg_loss, accuracy, images_per_sec


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()

    # We’ll just run in full precision on non-CUDA devices.
    amp_enabled = use_amp and (device.type == "cuda")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    start = time.perf_counter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if amp_enabled:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    device_synchronize(device)
    end = time.perf_counter()

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    images_per_sec = total_samples / (end - start)

    return avg_loss, accuracy, images_per_sec


# ------------------------
# Dataset setup
# ------------------------
def get_datasets(args):
    """
    Returns (train_dataset, val_dataset, num_classes).
    Supports: food101, cifar10, cifar100, imagenet.
    """
    dataset = args.dataset.lower()
    img_size = args.img_size

    # Common normalization (ImageNet stats)
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)

    if dataset == "food101":
        # FOOD-101: 75,750 train, 25,250 test, 101 classes, fairly large images.
        num_classes = 101

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        train_dataset = datasets.Food101(
            root=args.data_dir,
            split="train",
            download=True,
            transform=train_transform,
        )
        val_dataset = datasets.Food101(
            root=args.data_dir,
            split="test",
            download=True,
            transform=val_transform,
        )

    elif dataset in ("cifar10", "cifar100"):
        if dataset == "cifar10":
            num_classes = 10
            DatasetClass = torchvision.datasets.CIFAR10
        else:
            num_classes = 100
            DatasetClass = torchvision.datasets.CIFAR100

        # Upscale CIFAR to img_size so the conv net has more work to do.
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        train_dataset = DatasetClass(
            root=args.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )
        val_dataset = DatasetClass(
            root=args.data_dir,
            train=False,
            download=True,
            transform=val_transform,
        )

    elif dataset == "imagenet":
        # Expect:
        #   data_dir/train/<class>/*.jpeg
        #   data_dir/val/<class>/*.jpeg
        train_dir = os.path.join(args.data_dir, "train")
        val_dir = os.path.join(args.data_dir, "val")

        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise RuntimeError(
                f"ImageNet requires 'train' and 'val' folders inside {args.data_dir}. "
                "Example: /imagenet/train, /imagenet/val"
            )

        num_classes = args.num_classes if args.num_classes > 0 else 1000

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return train_dataset, val_dataset, num_classes


# ------------------------
# Model factory
# ------------------------
def create_model(model_name: str, num_classes: int):
    model_name = model_name.lower()
    if model_name == "resnet50":
        model = models.resnet50(weights=None, num_classes=num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(weights=None, num_classes=num_classes)
    elif model_name in ("wideresnet50", "wide_resnet50", "wide_resnet50_2"):
        model = models.wide_resnet50_2(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Heavy PyTorch ML benchmark (Food-101 default, Mac/CPU/CUDA)")
    parser.add_argument("--dataset", type=str, default="food101",
                        choices=["food101", "cifar10", "cifar100", "imagenet"],
                        help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Root directory for dataset downloads / storage")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "resnet101", "wide_resnet50_2"],
                        help="Backbone model")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Input resolution (e.g., 224, 256, 320)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=96,
                        help="Global batch size")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--workers", type=int, default=16,
                        help="DataLoader workers per process")
    parser.add_argument("--fp16", action="store_true",
                        help="Use Torch AMP mixed precision (CUDA only)")
    parser.add_argument("--limit-train", type=int, default=0,
                        help="Limit number of training samples (0 = use all)")
    parser.add_argument("--limit-val", type=int, default=0,
                        help="Limit number of validation samples (0 = use all)")
    parser.add_argument("--num-classes", type=int, default=0,
                        help="Override num_classes (for custom ImageNet-like dirs)")
    args = parser.parse_args()

    device = get_device()
    cuda_available = (device.type == "cuda")
    mps_available = (device.type == "mps")

    print(f"Using device: {device}")
    if cuda_available:
        props = torch.cuda.get_device_properties(device)
        print(f"  CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"  SMs: {props.multi_processor_count}, "
              f"Memory: {props.total_memory / 1024**3:.1f} GB, "
              f"Compute capability: {props.major}.{props.minor}")
    elif mps_available:
        print("  Backend: Apple Metal (MPS)")
    else:
        print("  Backend: CPU (no CUDA / MPS available)")

    print("-" * 70)
    print("Config:")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Model:        {args.model}")
    print(f"  Img size:     {args.img_size}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Workers:      {args.workers}")
    print(f"  FP16 (AMP):   {args.fp16 and cuda_available} (CUDA-only)")
    print(f"  Limit train:  {args.limit_train}")
    print(f"  Limit val:    {args.limit_val}")
    print("-" * 70)

    if cuda_available:
        torch.backends.cudnn.benchmark = True

    # ------------------------
    # Data
    # ------------------------
    train_dataset, val_dataset, num_classes = get_datasets(args)

    if args.limit_train > 0:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(min(args.limit_train, len(train_dataset)))
        )
    if args.limit_val > 0:
        val_dataset = torch.utils.data.Subset(
            val_dataset, range(min(args.limit_val, len(val_dataset)))
        )

    pin_memory = cuda_available  # avoid MPS warnings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=True if args.workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=True if args.workers > 0 else False,
    )

    # ------------------------
    # Model / optimizer
    # ------------------------
    model = create_model(args.model, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    # Effective AMP flag (CUDA only)
    use_amp = args.fp16 and cuda_available

    # ------------------------
    # Training loop
    # ------------------------
    all_train_ips = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc, train_ips = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_amp=use_amp
        )
        all_train_ips.append(train_ips)

        print(f"  Train loss:        {train_loss:.4f}")
        print(f"  Train accuracy:    {train_acc:.2f}%")
        print(f"  Train throughput:  {train_ips:.1f} images/s")

    avg_train_ips = sum(all_train_ips) / max(len(all_train_ips), 1)

    # ------------------------
    # Inference / evaluation
    # ------------------------
    print("\nRunning inference / validation...")
    val_loss, val_acc, infer_ips = evaluate(
        model, val_loader, criterion, device, use_amp=use_amp
    )

    print(f"\nValidation loss:          {val_loss:.4f}")
    print(f"Validation accuracy:      {val_acc:.2f}%")
    print(f"Inference throughput:     {infer_ips:.1f} images/s")

    # ------------------------
    # Benchmark score
    # ------------------------
    # Combined score:
    #     score = (0.6 * avg_train_ips + 0.4 * infer_ips)
    score = (0.6 * avg_train_ips + 0.4 * infer_ips)

    print("\n===================== BENCHMARK RESULT =====================")
    print(f"Average train throughput: {avg_train_ips:.1f} images/s")
    print(f"Inference throughput:     {infer_ips:.1f} images/s")
    print(f"Final validation accuracy:{val_acc:.2f}%")
    print(f"Benchmark score:          {score:.2f}")
    print("============================================================")


if __name__ == "__main__":
    main()