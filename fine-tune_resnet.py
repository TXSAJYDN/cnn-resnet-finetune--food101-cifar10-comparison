import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from PIL import Image
import importlib
from models.simple_cnn import SimpleCNN
from torch.optim.lr_scheduler import SequentialLR, LinearLR


def build_dataloaders(args):
    try:
        import torchvision
        from torchvision import transforms
    except Exception:
        if args.dataset.lower() != "fake":
            raise RuntimeError("需要 torchvision 才能使用 CIFAR10 或 ImageFolder 数据集。请安装: conda install -c pytorch torchvision 或 pip install torchvision")
        torchvision = None
        transforms = None
    if args.dataset.lower() == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        t_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        t_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, transform=t_train, download=True)
        val_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=t_val, download=True)
        num_classes = 10
    elif args.dataset.lower() == "food101":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        ops = [transforms.RandomResizedCrop(224)]
        if getattr(args, "aug", "none") == "rand":
            ops.append(transforms.RandAugment(num_ops=2, magnitude=getattr(args, "rand_magnitude", 9)))
        elif getattr(args, "aug", "none") == "auto":
            ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
        elif getattr(args, "aug", "none") == "cjre":
            ops.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))
        ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(mean, std))
        if getattr(args, "aug", "none") != "none":
            ops.append(transforms.RandomErasing(p=0.25))
        t_train = transforms.Compose(ops)
        t_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = torchvision.datasets.Food101(root=args.data_dir, split="train", transform=t_train, download=True)
        val_set = torchvision.datasets.Food101(root=args.data_dir, split="test", transform=t_val, download=True)
        num_classes = 101
    elif args.dataset.lower() == "imagefolder":
        size = args.img_size
        t_train = transforms.Compose([
            transforms.Resize(size + 32),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        t_val = transforms.Compose([
            transforms.Resize(size + 32),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])
        full = torchvision.datasets.ImageFolder(root=args.data_dir, transform=t_train)
        n_val = max(1, int(len(full) * args.val_split))
        n_train = len(full) - n_val
        train_set, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
        train_set.dataset.transform = t_train
        val_set.dataset.transform = t_val
        num_classes = len(full.classes)
    elif args.dataset.lower() == "fake":
        n = args.fake_size
        c = 3
        h = args.img_size
        w = args.img_size
        x = torch.rand(n, c, h, w)
        y = torch.randint(0, args.fake_classes, (n,))
        ds = TensorDataset(x, y)
        n_val = max(1, int(n * args.val_split))
        n_train = n - n_val
        train_set, val_set = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
        num_classes = args.fake_classes
    else:
        raise ValueError("dataset 仅支持: CIFAR10 | ImageFolder | Fake")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader, num_classes


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    total_batches = len(loader)
    log_interval = getattr(train_one_epoch, "_log_interval", 0)
    current_epoch = getattr(train_one_epoch, "_current_epoch", 0)
    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            loss_value = float(loss.detach().item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_value = float(loss.detach().item())
            loss.backward()
            optimizer.step()
        total_loss += loss_value * images.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()
        if log_interval and ((i + 1) % log_interval == 0 or (i + 1) == total_batches):
            pct = (i + 1) * 100.0 / total_batches
            print(f"epoch={current_epoch} step={i+1}/{total_batches} ({pct:.1f}%) loss={loss_value:.4f}", flush=True)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.detach().item() * images.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def predict_files(model, files, classes, dataset, img_size, device, topk=5, transform_override=None):
    try:
        import torchvision
        from torchvision import transforms
    except Exception:
        torchvision = None
        transforms = None
    if transform_override is not None:
        t = transform_override
    elif dataset.lower() == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        t = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        t = transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
    model.eval()
    for fp in files:
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            print(f"skip={fp}")
            continue
        x = t(img).unsqueeze(0).to(device)
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                y = model(x)
        else:
            y = model(x)
        p = torch.softmax(y, dim=1)
        k = min(topk, p.size(1))
        vals, idxs = p.topk(k, dim=1)
        idxs = idxs[0].tolist()
        vals = vals[0].tolist()
        name = classes[idxs[0]] if 0 <= idxs[0] < len(classes) else str(idxs[0])
        print(f"image={fp} top1={name} prob={vals[0]:.4f}")
        s = []
        for i in range(k):
            ci = classes[idxs[i]] if 0 <= idxs[i] < len(classes) else str(idxs[i])
            s.append(f"{ci}:{vals[i]:.4f}")
        print(f"top{k}=" + ", ".join(s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet18", choices=["simple_cnn", "resnet18", "resnet34", "resnet50"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--imagenet-classes", action="store_true")
    parser.add_argument("--dataset", type=str, default="Food101")
    parser.add_argument("--data-dir", type=str, default=str(Path("data")))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine"])
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--predict", type=str, default="")
    parser.add_argument("--predict-dir", type=str, default="")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--fake-size", type=int, default=1000)
    parser.add_argument("--fake-classes", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default=str(Path("outputs")))
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--aug", type=str, default="none", choices=["none", "cjre", "rand", "auto"])
    parser.add_argument("--rand-magnitude", type=int, default=9)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--accumulate-steps", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = None
    val_loader = None
    num_classes = None
    if not (args.imagenet_classes and (args.predict or args.predict_dir) and args.arch.startswith("resnet")):
        train_loader, val_loader, num_classes = build_dataloaders(args)
    if args.arch == "simple_cnn":
        if num_classes is None:
            num_classes = 1000 if args.imagenet_classes else (10 if args.dataset.lower() == "cifar10" else args.fake_classes)
        model = SimpleCNN(num_classes=num_classes, width_mult=args.width_mult).to(device)
        weights = None
    else:
        tv = importlib.import_module("torchvision.models")
        wmap = {
            "resnet18": getattr(tv, "ResNet18_Weights", None),
            "resnet34": getattr(tv, "ResNet34_Weights", None),
            "resnet50": getattr(tv, "ResNet50_Weights", None),
        }
        ctor = getattr(tv, args.arch)
        weights = None
        if args.pretrained and wmap.get(args.arch) is not None:
            weights = wmap[args.arch].DEFAULT
        model = ctor(weights=weights).to(device)
        if not (args.imagenet_classes and args.pretrained and (args.predict or args.predict_dir)):
            if num_classes is None:
                if args.dataset.lower() == "food101":
                    num_classes = 101
                elif args.dataset.lower() == "cifar10":
                    num_classes = 10
                else:
                    num_classes = args.fake_classes
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, num_classes).to(device)
        else:
            num_classes = 1000
    criterion = nn.CrossEntropyLoss(label_smoothing=max(0.0, args.label_smoothing))
    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch = 0
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if args.test_only and not args.resume:
        tag = f"{args.arch}_{args.dataset.lower()}"
        default_best = os.path.join(args.save_dir, f"best_{tag}.pt")
        if os.path.exists(default_best):
            args.resume = default_best
    if args.resume and not (args.imagenet_classes and args.pretrained and (args.predict or args.predict_dir)):
        if os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
        else:
            ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("model_state") or ckpt.get("state_dict")
        if state is not None:
            model.load_state_dict(state, strict=False)
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", 0.0)
    if args.predict or args.predict_dir:
        classes = []
        t_override = None
        if args.arch.startswith("resnet") and args.pretrained and args.imagenet_classes:
            tv = importlib.import_module("torchvision.models")
            wmap = {
                "resnet18": getattr(tv, "ResNet18_Weights", None),
                "resnet34": getattr(tv, "ResNet34_Weights", None),
                "resnet50": getattr(tv, "ResNet50_Weights", None),
            }
            wcls = wmap.get(args.arch)
            if wcls is not None:
                w = wcls.DEFAULT
                meta = getattr(w, "meta", {})
                classes = meta.get("categories", [])
                t_override = w.transforms()
        if not classes:
            if args.dataset.lower() == "cifar10":
                classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
            elif args.dataset.lower() == "food101":
                try:
                    import torchvision
                    ds = torchvision.datasets.Food101(root=args.data_dir, split="train", download=True)
                    classes = ds.classes
                except Exception:
                    classes = []
            elif args.dataset.lower() == "imagefolder":
                try:
                    import torchvision
                    ds = torchvision.datasets.ImageFolder(root=args.data_dir)
                    classes = ds.classes
                except Exception:
                    classes = []
            else:
                classes = [str(i) for i in range(num_classes if num_classes is not None else 0)]
        files = []
        if args.predict and os.path.isfile(args.predict):
            files.append(args.predict)
        if args.predict_dir and os.path.isdir(args.predict_dir):
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            for root, _, fnames in os.walk(args.predict_dir):
                for f in fnames:
                    if os.path.splitext(f.lower())[1] in exts:
                        files.append(os.path.join(root, f))
        if not files:
            print("no_valid_input")
            return
        predict_files(model, files, classes, args.dataset, args.img_size, device, topk=args.topk, transform_override=t_override)
        return
    if args.test_only:
        if val_loader is None:
            train_loader, val_loader, num_classes = build_dataloaders(args)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"eval_only val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        return
    if train_loader is None or val_loader is None:
        train_loader, val_loader, num_classes = build_dataloaders(args)
    if args.resume:
        params = []
        for n, p in model.named_parameters():
            if "fc" in n:
                params.append({"params": p, "lr": args.lr_head})
            else:
                params.append({"params": p, "lr": args.lr_backbone})
        optimizer = optim.Adam(params, weight_decay=args.weight_decay)
        if args.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
        for epoch in range(1, args.epochs + 1):
            current_epoch = start_epoch + epoch
            train_one_epoch._log_interval = args.log_interval
            train_one_epoch._current_epoch = current_epoch
            train_one_epoch._accumulate_steps = max(1, args.accumulate_steps)
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler if use_amp else None)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            print(f"epoch={current_epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            tag = f"{args.arch}_{args.dataset.lower()}"
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_{tag}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "epoch": current_epoch,
                "best_acc": best_acc,
                "num_classes": num_classes,
            }, ckpt_path)
            if val_acc > best_acc:
                best_acc = val_acc
                best_path = os.path.join(args.save_dir, f"best_{tag}.pt")
                torch.save({"model_state": model.state_dict(), "num_classes": num_classes}, best_path)
                print(f"saved={best_path} best_acc={best_acc:.4f}")
        tag = f"{args.arch}_{args.dataset.lower()}"
        final_path = os.path.join(args.save_dir, f"last_{tag}.pt")
        torch.save({"model_state": model.state_dict(), "num_classes": num_classes}, final_path)
        print(f"saved={final_path} done")
        return
    if args.freeze_epochs > 0:
        for n, p in model.named_parameters():
            if not n.startswith("fc"):
                p.requires_grad = False
        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
        if args.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, max(1, args.freeze_epochs) // 2), gamma=0.5)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.freeze_epochs))
        for epoch in range(1, args.freeze_epochs + 1):
            current_epoch = epoch
            train_one_epoch._log_interval = args.log_interval
            train_one_epoch._current_epoch = current_epoch
            train_one_epoch._accumulate_steps = max(1, args.accumulate_steps)
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler if use_amp else None)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            print(f"epoch={current_epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            tag = f"{args.arch}_{args.dataset.lower()}"
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_{tag}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "epoch": current_epoch,
                "best_acc": best_acc,
                "num_classes": num_classes,
            }, ckpt_path)
            if val_acc > best_acc:
                best_acc = val_acc
                best_path = os.path.join(args.save_dir, f"best_{tag}.pt")
                torch.save({"model_state": model.state_dict(), "num_classes": num_classes}, best_path)
                print(f"saved={best_path} best_acc={best_acc:.4f}")
        for p in model.parameters():
            p.requires_grad = True
    params = []
    for n, p in model.named_parameters():
        if "fc" in n:
            params.append({"params": p, "lr": args.lr_head})
        else:
            params.append({"params": p, "lr": args.lr_backbone})
    optimizer = optim.Adam(params, weight_decay=args.weight_decay)
    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)
    else:
        main_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
        if args.warmup_epochs > 0:
            warm = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, args.warmup_epochs))
            scheduler = SequentialLR(optimizer, [warm, main_sched], milestones=[args.warmup_epochs])
        else:
            scheduler = main_sched
    for epoch in range(1, args.epochs + 1):
        current_epoch = args.freeze_epochs + epoch
        train_one_epoch._log_interval = args.log_interval
        train_one_epoch._current_epoch = current_epoch
        train_one_epoch._accumulate_steps = max(1, args.accumulate_steps)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler if use_amp else None)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"epoch={current_epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        tag = f"{args.arch}_{args.dataset.lower()}"
        ckpt_path = os.path.join(args.save_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "model_state": model.state_dict(),
            "epoch": current_epoch,
            "best_acc": best_acc,
            "num_classes": num_classes,
        }, ckpt_path)
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.save_dir, f"best_{tag}.pt")
            torch.save({"model_state": model.state_dict(), "num_classes": num_classes}, best_path)
            print(f"saved={best_path} best_acc={best_acc:.4f}")
    tag = f"{args.arch}_{args.dataset.lower()}"
    final_path = os.path.join(args.save_dir, f"last_{tag}.pt")
    torch.save({"model_state": model.state_dict(), "num_classes": num_classes}, final_path)
    print(f"saved={final_path} done")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
