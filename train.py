import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

# ==========================================
# 1. 配置参数 (全局变量，但在 main 中调用)
# ==========================================
DATA_ROOT = "/home/nt/code/catsdogs/"  # 请务必确认这个路径是正确的！

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16       # 1660Ti 6GB 显存，16比较稳妥
NUM_EPOCHS = 12
LR = 3e-4
IMG_SIZE = 224
SEED = 42

def seed_all(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

# ==========================================
# 2. 主函数入口
# ==========================================
def main():
    seed_all()

    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cpu":
        print("Warning: CUDA not available, training on CPU.")

    TRAIN_DIR = os.path.join(DATA_ROOT, "training_set")
    VAL_DIR   = os.path.join(DATA_ROOT, "val_set")
    TEST_DIR  = os.path.join(DATA_ROOT, "test_set")

    # ---------- transforms ----------
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ---------- datasets ----------
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=val_tf)

    assert train_ds.classes == val_ds.classes == test_ds.classes
    print("Classes:", train_ds.classes)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # 关键点：num_workers 如果在 Linux 下报错，可以暂时设为 0 调试
    # 如果设为 0 能跑，说明是多进程问题。设为 0 只是慢一点点，不影响结果。
    num_workers = 4

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    # ---------- model ----------
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    best_val_acc = 0.0
    best_path = "best_catdog.pth"

    # ---------- Training Loop ----------
    print("\nStarting Training...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        run_loss, corr, tot = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * x.size(0)
            corr += (out.argmax(1) == y).sum().item()
            tot += y.size(0)

        tr_loss = run_loss / max(tot, 1)
        tr_acc = corr / max(tot, 1)

        # ---- Validation ----
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                v_corr += (out.argmax(1) == y).sum().item()
                v_tot += y.size(0)

        v_loss = v_loss / max(v_tot, 1)
        v_acc = v_corr / max(v_tot, 1)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
            }, best_path)
            print(f"[epoch {epoch}] ★ New best val_acc={v_acc:.2%} saved")

        print(f"epoch {epoch:02d}/{NUM_EPOCHS} "
              f"| train {tr_loss:.4f}/{tr_acc:.2%}  "
              f"| val {v_loss:.4f}/{v_acc:.2%}  "
              f"({time.time()-t0:.1f}s)")

    # ---------- Test ----------
    model.eval()
    te_corr, te_tot = 0, 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            te_corr += (model(x).argmax(1) == y).sum().item()
            te_tot += y.size(0)

    print("\nTest acc (final weight):", f"{te_corr/te_tot:.2%}")
    print("Best val weight saved at :", best_path)

# ==========================================
# 3. 保护入口
# ==========================================
if __name__ == '__main__':
    # 这一行是为了防止多进程在某些平台上报错（虽然你是Linux，但加上更安全）
    # from multiprocessing import freeze_support
    # freeze_support()

    main()
