import os
from pathlib import Path
import csv
import re
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.simple_cnn import SimpleCNN


def _exists(p):
    try:
        return Path(p).exists()
    except Exception:
        return False


def _load_ckpt(p, device):
    return torch.load(p, map_location=device)


def _eval(loader, model, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += float(loss.detach().item()) * images.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
    return total_loss / total, correct / total


def _make_food101_loaders(root, batch_size=128, workers=4):
    import torchvision
    from torchvision import transforms
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    t_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = torchvision.datasets.Food101(root=str(root), split="test", transform=t_val, download=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return loader, ds.classes


def _make_cifar10_loader(root, batch_size=256, workers=2):
    import torchvision
    from torchvision import transforms
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = torchvision.datasets.CIFAR10(root=str(root), train=False, transform=t, download=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return loader, ds.classes


def _build_resnet(arch, num_classes, device, ckpt):
    import torchvision.models as tvm
    ctor = getattr(tvm, arch)
    model = ctor(weights=None).to(device)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes).to(device)
    state = ckpt.get("model_state") or ckpt.get("state_dict")
    if state is not None:
        model.load_state_dict(state, strict=False)
    return model


def _build_simplecnn(ckpt, device, dataset_classes):
    state = ckpt.get("model_state") or ckpt.get("state_dict") or {}
    c1 = None
    for k, v in state.items():
        if k.endswith("features.0.weight") or k == "features.0.weight":
            c1 = v.shape[0]
            break
    width_mult = 1.0
    if c1 is not None and c1 >= 8:
        width_mult = c1 / 32.0
    num_classes = dataset_classes if isinstance(dataset_classes, int) else len(dataset_classes)
    model = SimpleCNN(num_classes=num_classes, width_mult=width_mult).to(device)
    if state:
        target = model.state_dict()
        pruned = {}
        for k, v in state.items():
            if k in target and target[k].shape != v.shape:
                continue
            pruned[k] = v
        model.load_state_dict(pruned, strict=False)
    return model


def _plot_bars(rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("skip_plots")
        return
    labels = [f"{r[0]}-{r[1]}" for r in rows]
    accs = [float(r[2]) for r in rows]
    losses = [float(r[3]) for r in rows]
    fig = plt.figure(figsize=(8, 4))
    plt.bar(labels, accs, color="#4e79a7")
    for i, v in enumerate(accs):
        plt.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylabel("val_acc")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p1 = Path(out_dir) / "compare_bar_acc.png"
    plt.savefig(p1)
    plt.close(fig)
    fig = plt.figure(figsize=(8, 4))
    plt.bar(labels, losses, color="#f28e2b")
    for i, v in enumerate(losses):
        plt.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylabel("val_loss")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p2 = Path(out_dir) / "compare_bar_loss.png"
    plt.savefig(p2)
    plt.close(fig)
    print(f"saved={p1}")
    print(f"saved={p2}")


def _parse_log(log_path):
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return []
    patt = re.compile(r"epoch=(\d+)\s+train_loss=([0-9\.]+)\s+train_acc=([0-9\.]+)\s+val_loss=([0-9\.]+)\s+val_acc=([0-9\.]+)")
    out = []
    for m in patt.finditer(text):
        ep = int(m.group(1))
        tl = float(m.group(2))
        ta = float(m.group(3))
        vl = float(m.group(4))
        va = float(m.group(5))
        out.append({"epoch": ep, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
    return out


def _plot_curves(logs, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("skip_plots")
        return
    series = []
    for label, path in logs:
        data = _parse_log(path)
        if data:
            xs = [d["epoch"] for d in data]
            ys = [d["val_acc"] for d in data]
            series.append((label, xs, ys))
    if not series:
        return
    fig = plt.figure(figsize=(8, 4))
    for label, xs, ys in series:
        plt.plot(xs, ys, label=label)
    plt.xlabel("epoch")
    plt.ylabel("val_acc")
    plt.legend()
    plt.tight_layout()
    p = Path(out_dir) / "train_curves_food101.png"
    plt.savefig(p)
    plt.close(fig)
    print(f"saved={p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(Path("data")))
    parser.add_argument("--save-dir", type=str, default=str(Path("outputs")))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--batch-size-food", type=int, default=128)
    parser.add_argument("--batch-size-cifar", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--simplecnn-ckpt", type=str, default="")
    parser.add_argument("--simplecnn-num-classes", type=int, default=10)
    parser.add_argument("--simplecnn-dataset", type=str, default="CIFAR10")
    args = parser.parse_args()
    root = Path(".")
    data_root = Path(args.data_dir)
    outputs_root = Path(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    # ResNet18 on Food101
    r18_path = outputs_root / "best_resnet18_food101.pt"
    if _exists(r18_path):
        ck = _load_ckpt(r18_path, device)
        loader, classes = _make_food101_loaders(data_root, batch_size=args.batch_size_food, workers=args.workers)
        model = _build_resnet("resnet18", len(classes), device, ck)
        val_loss, val_acc = _eval(loader, model, device)
        rows.append(["ResNet18", "Food101", f"{val_acc:.4f}", f"{val_loss:.4f}", str(r18_path)])
        print(f"ResNet18 Food101 val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
    # ResNet50 on Food101
    r50_path = outputs_root / "best_resnet50_food101.pt"
    if _exists(r50_path):
        ck = _load_ckpt(r50_path, device)
        loader, classes = _make_food101_loaders(data_root, batch_size=args.batch_size_food, workers=args.workers)
        model = _build_resnet("resnet50", len(classes), device, ck)
        val_loss, val_acc = _eval(loader, model, device)
        rows.append(["ResNet50", "Food101", f"{val_acc:.4f}", f"{val_loss:.4f}", str(r50_path)])
        print(f"ResNet50 Food101 val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
    # SimpleCNN on CIFAR10 (only if a valid 10-class checkpoint is found)
    used = None
    if args.simplecnn_ckpt and _exists(args.simplecnn_ckpt):
        p = Path(args.simplecnn_ckpt)
        ck = _load_ckpt(p, device)
        state = ck.get("model_state") or ck.get("state_dict") or {}
        if any(k.startswith("features.") for k in state.keys()):
            used = (p, ck)
    else:
        scnn_candidates = [
            outputs_root / "best_model_simple_cnn.pt",
            outputs_root / "last_model_simple_cnn.pt",
            outputs_root / "checkpoint_simple_cnn.pt",
            outputs_root / "best_model.pt",
            outputs_root / "last_model.pt",
            outputs_root / "checkpoint.pt",
        ]
        for p in scnn_candidates:
            if not _exists(p):
                continue
            ck = _load_ckpt(p, device)
            state = ck.get("model_state") or ck.get("state_dict") or {}
            if any(k.startswith("features.") for k in state.keys()):
                used = (p, ck)
                break
    if used is not None:
        p, ck = used
        ds_name = args.simplecnn_dataset.lower()
        if ds_name == "cifar10":
            loader, classes = _make_cifar10_loader(data_root, batch_size=args.batch_size_cifar, workers=max(1, args.workers // 2))
            model = _build_simplecnn(ck, device, len(classes))
            val_loss, val_acc = _eval(loader, model, device)
            rows.append(["SimpleCNN", "CIFAR10", f"{val_acc:.4f}", f"{val_loss:.4f}", str(p)])
            print(f"SimpleCNN CIFAR10 val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
        else:
            loader, classes = _make_food101_loaders(data_root, batch_size=args.batch_size_food, workers=args.workers)
            ncls = len(classes)
            model = _build_simplecnn({"model_state": ck.get("model_state"), "state_dict": ck.get("state_dict")}, device, ncls)
            val_loss, val_acc = _eval(loader, model, device)
            rows.append(["SimpleCNN", "Food101", f"{val_acc:.4f}", f"{val_loss:.4f}", str(p)])
            print(f"SimpleCNN Food101 val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
    out_csv = outputs_root / "compare_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "val_acc", "val_loss", "checkpoint"])
        for r in rows:
            w.writerow(r)
    print(f"saved={out_csv}")
    if rows and args.plot:
        _plot_bars(rows, outputs_root)
        logs = []
        p18 = outputs_root / "train_resnet18_food101.log"
        p50 = outputs_root / "train_resnet50_food101.log"
        if _exists(p18):
            logs.append(("ResNet18", str(p18)))
        if _exists(p50):
            logs.append(("ResNet50", str(p50)))
        _plot_curves(logs, outputs_root)


if __name__ == "__main__":
    main()
