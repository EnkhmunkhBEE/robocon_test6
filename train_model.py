"""
=============================================================
 Robocon Fake / Real Classifier — Training Script
 Model: EfficientNet-B0 (pretrained on ImageNet, fine-tuned)
=============================================================

 FOLDER STRUCTURE EXPECTED:
   data/
   ├── train/
   │   ├── fake/   ← all fake images (flat OR nested img1/fake1.png)
   │   └── real/
   └── val/
       ├── fake/
       └── real/

 USAGE:
   python train_model.py
   python train_model.py --data data --epochs 30 --batch 16

 OUTPUT:
   model.pt          ← best checkpoint (use this with detect_realtime.py)
   training_log.csv  ← epoch-by-epoch metrics
"""

import os, csv, time, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",    default="data",    help="Root data folder")
parser.add_argument("--epochs",  default=30, type=int)
parser.add_argument("--batch",   default=16, type=int)
parser.add_argument("--lr",      default=1e-4, type=float)
parser.add_argument("--img-size",default=224, type=int)
parser.add_argument("--workers", default=0,   type=int,  # 0 = safe on Windows
                    help="DataLoader workers (keep 0 on Windows)")
parser.add_argument("--output",  default="model.pt")
args = parser.parse_args()

IMG_SIZE = args.img_size
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(20),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

# ── Datasets ──────────────────────────────────────────────────────────────────
# ImageFolder automatically labels folders: fake=0, real=1 (alphabetical)
train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(args.data, "val"),   transform=val_tf)

print(f"[INFO] Classes : {train_ds.classes}")   # should print ['fake', 'real']
print(f"[INFO] Train   : {len(train_ds)} images")
print(f"[INFO] Val     : {len(val_ds)}   images")

train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                      num_workers=args.workers, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                      num_workers=args.workers, pin_memory=True)

# ── Model: EfficientNet-B0 ────────────────────────────────────────────────────
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Replace classifier head for 2-class output
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 2),
)

model = model.to(DEVICE)

# ── Loss, optimizer, scheduler ───────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-6)

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0
log_rows     = []

print("\n" + "="*60)
print(f"  Training EfficientNet-B0 for {args.epochs} epochs")
print("="*60)

for epoch in range(1, args.epochs + 1):
    t0 = time.time()

    # ── Train ──────────────────────────────────────────────────────
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for imgs, labels in train_dl:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * imgs.size(0)
        preds          = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += imgs.size(0)

    train_acc  = train_correct / train_total
    train_loss = train_loss    / train_total

    # ── Validate ───────────────────────────────────────────────────
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs      = model(imgs)
            loss         = criterion(outputs, labels)

            val_loss    += loss.item() * imgs.size(0)
            preds        = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += imgs.size(0)

    val_acc  = val_correct / val_total
    val_loss = val_loss    / val_total
    scheduler.step()

    elapsed = time.time() - t0
    star    = " ★ BEST" if val_acc > best_val_acc else ""

    print(f"Epoch [{epoch:02d}/{args.epochs}]  "
          f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
          f"({elapsed:.1f}s){star}")

    log_rows.append({
        "epoch": epoch,
        "train_loss": round(train_loss,4),
        "train_acc":  round(train_acc, 4),
        "val_loss":   round(val_loss,  4),
        "val_acc":    round(val_acc,   4),
    })

    # ── Save best model ────────────────────────────────────────────
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model, args.output)
        print(f"           → Saved best model  (val_acc={best_val_acc:.4f})")

# ── Save training log ─────────────────────────────────────────────────────────
with open("training_log.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
    writer.writeheader()
    writer.writerows(log_rows)

print("\n" + "="*60)
print(f"  Training complete!")
print(f"  Best val accuracy : {best_val_acc:.4f}")
print(f"  Model saved to    : {args.output}")
print(f"  Log saved to      : training_log.csv")
print("="*60)
print(f"\n  Now run:")
print(f"  python detect_realtime.py --model {args.output} --framework pytorch\n")