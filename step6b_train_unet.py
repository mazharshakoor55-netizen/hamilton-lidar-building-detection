"""
STEP 6B - Train a U-Net to segment buildings from elevation
===========================================================

Inputs are three physical bands - nDSM, slope, roughness - not imagery. The
model learns what a roof looks like in elevation, which is why it can then be
applied to 2019 and 2023 separately and the results differenced.

Design notes worth knowing
--------------------------
LOSS      Dice + boundary-weighted BCE. Plain BCE converges to predicting
          background everywhere: at ~13% positives that is a strong local
          minimum costing almost nothing. Dice is insensitive to class
          frequency and pulls the model off it.

ENCODER   ImageNet-pretrained ResNet-34, stem adapted from 3 RGB channels to
          our 3 physical bands. The pretrained early filters detect edges and
          corners, which transfer fine even though the input is not a photo.
          Same transfer-learning move as the EuroSAT work, applied to dense
          prediction.

AUGMENT   Dihedral flips and 90-degree rotations only. NO brightness, blur or
          scale jitter - these bands are measurements in metres and degrees.
          A 6 m wall is not a 3 m wall seen differently, and distorting the
          values destroys exactly the signal the model needs.

SELECTION By validation IoU, never training loss. With Dice in the objective
          the loss keeps falling after val IoU has turned over.

CPU is workable at this size (~30 min/epoch); a GPU makes it minutes.

RUN
---
    cd /d D:\\hamilton
    python step6b_train_unet.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    sys.exit("Needs PyTorch:  pip install torch")

HERE = Path(__file__).parent
TRAIN = HERE / "data" / "training"
OUT = HERE / "outputs"
MODELS = OUT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

EPOCHS = 40
BATCH = 8
LR = 3e-4
POS_WEIGHT = 2.0
BOUNDARY_WEIGHT = 2.5
PATIENCE = 10


class Patches(Dataset):
    def __init__(self, split, augment=False):
        man = json.loads((TRAIN / "manifest.json").read_text())
        self.files = [TRAIN / m["file"] for m in man if m["split"] == split]
        self.augment = augment
        if not self.files:
            sys.exit(f"No patches for split '{split}' - run step6a first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        x, y = d["x"].astype(np.float32), d["y"].astype(np.int64)
        if self.augment:
            k = np.random.randint(4)
            if k:
                x, y = np.rot90(x, k, (1, 2)).copy(), np.rot90(y, k, (0, 1)).copy()
            if np.random.rand() < 0.5:
                x, y = np.flip(x, 2).copy(), np.flip(y, 1).copy()
        return torch.from_numpy(x), torch.from_numpy(y)


class DiceBoundaryBCE(nn.Module):
    def __init__(self, pos_weight=POS_WEIGHT, boundary=BOUNDARY_WEIGHT, smooth=1.0):
        super().__init__()
        self.pw = pos_weight
        self.bw = boundary
        self.smooth = smooth

    def forward(self, logits, target):
        logits = logits.squeeze(1)
        binary = (target > 0).float()

        w = torch.ones_like(binary)
        w[target == 2] = self.bw          # roof edges matter most

        bce = F.binary_cross_entropy_with_logits(
            logits, binary, weight=w,
            pos_weight=torch.tensor(self.pw, device=logits.device))

        p = torch.sigmoid(logits)
        inter = (p * binary).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + binary.sum(dim=(1, 2))
        dice = 1 - ((2 * inter + self.smooth) / (denom + self.smooth)).mean()

        return bce + dice


def build_model(in_ch=3):
    try:
        import segmentation_models_pytorch as smp
        m = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=in_ch, classes=1)
        print("  U-Net, ImageNet-pretrained ResNet-34 encoder")
        return m
    except ImportError:
        print("  segmentation-models-pytorch not found - using a plain U-Net")
        return SimpleUNet(in_ch)


class SimpleUNet(nn.Module):
    """Fallback so the step still runs without smp installed."""

    def __init__(self, in_ch=3, base=32):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

        self.e1, self.e2 = blk(in_ch, base), blk(base, base * 2)
        self.e3, self.e4 = blk(base * 2, base * 4), blk(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = blk(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = blk(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = blk(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = torch.sigmoid(model(x).squeeze(1)) > thr
        truth = y > 0
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"iou": iou, "precision": prec, "recall": rec, "f1": f1}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
    else:
        print("  CPU - expect roughly 20-40 min per epoch at this size.")
        print("  Reduce EPOCHS, or train this step on Colab and bring back the .pt")

    tr = DataLoader(Patches("train", augment=True), batch_size=BATCH,
                    shuffle=True, drop_last=True)
    va = DataLoader(Patches("val"), batch_size=BATCH)
    te = DataLoader(Patches("test"), batch_size=BATCH)
    print(f"\n  train {len(tr.dataset)} | val {len(va.dataset)} | "
          f"test {len(te.dataset)} patches")

    model = build_model().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  {n_par/1e6:.1f}M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = DiceBoundaryBCE()

    best, best_ep, history = 0.0, -1, []
    ckpt = MODELS / "unet_buildings.pt"
    t0 = time.time()

    for ep in range(EPOCHS):
        model.train()
        losses = []
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        m = evaluate(model, va, device)
        m.update(epoch=ep, loss=float(np.mean(losses)),
                 mins=round((time.time() - t0) / 60, 1))
        history.append(m)

        flag = ""
        if m["iou"] > best:
            best, best_ep = m["iou"], ep
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_iou": best, "in_ch": 3}, ckpt)
            flag = "  * saved"

        print(f"  ep {ep:02d}  loss {m['loss']:.4f}  "
              f"val IoU {m['iou']:.4f}  P {m['precision']:.3f}  "
              f"R {m['recall']:.3f}  [{m['mins']:.0f}m]{flag}")

        if ep - best_ep >= PATIENCE:
            print(f"\n  no improvement for {PATIENCE} epochs - stopping")
            break

    print(f"\n  best val IoU {best:.4f} at epoch {best_ep}")

    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    t = evaluate(model, te, device)

    print(f"\n{'='*58}")
    print("HELD-OUT TEST (spatially separate blocks)")
    print(f"{'='*58}")
    print(f"  IoU       {t['iou']:.4f}")
    print(f"  precision {t['precision']:.4f}")
    print(f"  recall    {t['recall']:.4f}")
    print(f"  F1        {t['f1']:.4f}")

    (OUT / "training_history.json").write_text(
        json.dumps({"history": history, "test": t,
                    "best_val_iou": best, "best_epoch": best_ep}, indent=1))

    print(f"\n  model  -> {ckpt}")
    print(f"  history-> outputs/training_history.json")
    print(f"\n  Note: the test blocks have a much lower building fraction than")
    print(f"  train (5.8% vs 18.6%), so test IoU understates performance on")
    print(f"  typical urban ground. Report both numbers.")
    print(f"\n  Next: python step6c_apply_model.py")


if __name__ == "__main__":
    main()
