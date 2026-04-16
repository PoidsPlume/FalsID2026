"""
Pipeline complet : split → entraînement ResNet18 (multi-GPU) → métriques + matrice de confusion
Niveaux d'évaluation : bloc (prédiction unitaire) et page (vote majoritaire).
Split : 72 % train | 18 % val | 10 % test  (stratifié par image d'origine)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)


# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
CONFIG = {
    "batch_size": 32,          # augmenté pour profiter des 2 GPU
    "lr": 1e-4,
    "epochs": 15,
    "seed": 42,
    "num_workers": 4,
    "weight_decay": 1e-4,
    "scheduler_patience": 2,
    "scheduler_factor": 0.5,
    # Chemins
    "bin_dir": Path("Competition_dezip/FalseID/ICDAR/Christine de Pizan/1. Training dataset with images labeled as authentic or fake/binarized_1B"),
    "out_dir": Path("data/ICDAR/Christine de Pizan"),
    # Proportions
    "val_ratio": 0.10,         # 10 % val  (≈ 80/20 du bloc train+val, i.e. 90 % du total)
    "test_ratio": 0.10,        # 10 % test
}
CONFIG["out_dir"].mkdir(parents=True, exist_ok=True)

# GPU
if torch.cuda.device_count() >= 2:
    device = torch.device("cuda")
    use_multi_gpu = True
    print(f"Multi-GPU détecté : {torch.cuda.device_count()} GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    use_multi_gpu = False
    print("Un seul GPU disponible.")
else:
    device = torch.device("cpu")
    use_multi_gpu = False
    print("CPU uniquement.")

CONFIG["device"] = device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])


# ──────────────────────────────────────────────────────────────
# 1. COLLECTE DES FICHIERS ET SPLIT STRATIFIÉ
# ──────────────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def build_splits(bin_dir: Path, val_ratio: float, test_ratio: float, seed: int):
    """
    Retourne train_list, val_list, test_list sous la forme [(path, label, origin), ...].
    Le split est réalisé au niveau des images d'origine (pas des blocs)
    pour éviter toute fuite de données entre les ensembles.
    """
    file_by_base: dict[str, list] = defaultdict(list)
    for p in bin_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            stem = p.stem
            base = "_".join(stem.split("_")[:-1]) if "_" in stem else stem
            file_by_base[base].append(str(p))

    total_origins = len(file_by_base)
    total_blocks  = sum(len(v) for v in file_by_base.values())
    print(f" {total_origins} images d'origine | {total_blocks} blocs au total")

    # Séparer authentic / fake pour un split stratifié
    auth_bases = [b for b in file_by_base if b.startswith("aut_")]
    fake_bases = [b for b in file_by_base if not b.startswith("aut_")]

    rng = random.Random(seed)
    rng.shuffle(auth_bases)
    rng.shuffle(fake_bases)

    def split_list(lst):
        n = len(lst)
        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        return lst[n_test + n_val:], lst[n_test:n_test + n_val], lst[:n_test]

    auth_train, auth_val, auth_test = split_list(auth_bases)
    fake_train, fake_val, fake_test = split_list(fake_bases)

    train_bases = set(auth_train + fake_train)
    val_bases   = set(auth_val   + fake_val)
    test_bases  = set(auth_test  + fake_test)

    def make_list(bases_set):
        items = []
        for base, paths in file_by_base.items():
            if base not in bases_set:
                continue
            label = 1 if base.startswith("aut_") else 0
            origin = "_".join(base.rsplit("_", 1)[:-1]) if "_" in base else base # pour attraper le nom du fichier d'origine
            items.extend((p, label, origin) for p in paths)
        return items

    train_list = make_list(train_bases)
    val_list   = make_list(val_bases)
    test_list  = make_list(test_bases)

    for split_name, lst, bases in [
        ("Train", train_list, train_bases),
        ("Val",   val_list,   val_bases),
        ("Test",  test_list,  test_bases),
    ]:
        n_auth = sum(1 for b in bases if b.startswith("aut_"))
        n_fake = len(bases) - n_auth
        print(f"  {split_name:5s}: {len(bases):4d} origines ({n_auth} auth / {n_fake} fake) → {len(lst)} blocs")

    return train_list, val_list, test_list


train_list, val_list, test_list = build_splits(
    CONFIG["bin_dir"],
    CONFIG["val_ratio"],
    CONFIG["test_ratio"],
    CONFIG["seed"],
)

# Sauvegarde CSV
for name, lst in [("train", train_list), ("val", val_list), ("test", test_list)]:
    df = pd.DataFrame(lst, columns=["path", "label", "origin"])
    df.to_csv(CONFIG["out_dir"] / f"{name}_list.csv", index=False)
    print(f" {name}_list.csv sauvegardé ({len(df)} lignes)")


# ──────────────────────────────────────────────────────────────
# 2. DATASET ET TRANSFORMS
# ──────────────────────────────────────────────────────────────
class PadToSquare:
    def __init__(self, fill: int = 255):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        pad_w = (max_side - w) // 2
        pad_h = (max_side - h) // 2
        return transforms.functional.pad(
            img,
            (pad_w, pad_h, max_side - w - pad_w, max_side - h - pad_h),
            fill=self.fill,
        )


class PaleoDataset(Dataset):
    def __init__(self, data_list: list, transform=None):
        # data_list = [(path, label, origin), ...]
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        path, label, origin = self.data[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    PadToSquare(fill=255),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    PadToSquare(fill=255),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_ds = PaleoDataset(train_list, transform=train_transforms)
val_ds   = PaleoDataset(val_list,   transform=eval_transforms)
test_ds  = PaleoDataset(test_list,  transform=eval_transforms)

loader_kwargs = dict(num_workers=CONFIG["num_workers"], pin_memory=torch.cuda.is_available())
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  **loader_kwargs)
val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False, **loader_kwargs)


# ──────────────────────────────────────────────────────────────
# 3. MODÈLE (ResNet18 + multi-GPU)
# ──────────────────────────────────────────────────────────────
print("\nChargement du modèle ResNet18…")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, 2),
)

if use_multi_gpu:
    # Utilise les deux premiers GPU disponibles
    model = nn.DataParallel(model, device_ids=[0, 1])
    print("DataParallel activé sur GPU 0 et 1")

model = model.to(device)


# ──────────────────────────────────────────────────────────────
# 4. ENTRAÎNEMENT
# ──────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min",
    factor=CONFIG["scheduler_factor"],
    patience=CONFIG["scheduler_patience"]
)

best_val_acc = 0.0
best_model_path = CONFIG["out_dir"] / "best_resnet_paleo.pth"
history = []

print(f"\n Début de l'entraînement sur {device} | {CONFIG['epochs']} époques")

for epoch in range(CONFIG["epochs"]):
    # ── Train ──────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    train_preds, train_targets = [], []

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())
        loop.set_postfix(loss=f"{loss.item():.4f}")

    # ── Validation ─────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    val_preds, val_targets = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    train_acc = accuracy_score(train_targets, train_preds)
    val_acc   = accuracy_score(val_targets,   val_preds)
    val_f1    = f1_score(val_targets, val_preds, average="weighted", zero_division=0)

    history.append({
        "epoch": epoch + 1,
        "train_acc": train_acc,
        "val_loss":  avg_val_loss,
        "val_acc":   val_acc,
        "val_f1":    val_f1,
    })

    # Sauvegarde du meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # DataParallel → sauvegarder module sous-jacent
        state = model.module.state_dict() if use_multi_gpu else model.state_dict()
        torch.save(state, best_model_path)
        marker = " best"
    else:
        marker = ""

    scheduler.step(avg_val_loss)
    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f}"
        f"{marker}"
    )

print(f"\n Entraînement terminé. Meilleure Val Acc = {best_val_acc:.4f}")


# ──────────────────────────────────────────────────────────────
# 5. ÉVALUATION COMPLÈTE SUR LE TEST SET
# ──────────────────────────────────────────────────────────────
print("\n📊 Chargement du meilleur modèle pour l'évaluation test…")

# Recréer un modèle propre (sans DataParallel) pour l'inférence
eval_model = resnet18(weights=None)
eval_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
eval_model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_ftrs, 2))
eval_model.load_state_dict(torch.load(best_model_path, map_location=device))
eval_model = eval_model.to(device)
eval_model.eval()

test_preds, test_targets = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Test inference"):
        images = images.to(device)
        outputs = eval_model(images)
        preds = outputs.argmax(dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

# Associer les prédictions aux métadonnées
df_test = pd.DataFrame(test_list, columns=["path", "label", "origin"])
df_test["y_pred"] = test_preds

# ── Métriques niveau BLOC ──────────────────────────────────
print("\n" + "═"*55)
print("📊 MÉTRIQUES NIVEAU BLOC (prédictions unitaires)")
print("═"*55)
print(classification_report(
    df_test["label"], df_test["y_pred"],
    target_names=["FAKE", "AUTHENTIC"],
    labels=[0, 1],
    digits=4,
))

# ── Métriques niveau PAGE (vote majoritaire) ────────────────
df_pages = (
    df_test
    .groupby("origin")
    .agg(
        y_true  = ("label",  "first"),
        y_pred  = ("y_pred", lambda x: x.mode()[0]),
        n_blocs = ("y_pred", "count"),
    )
    .reset_index()
)

print("═"*55)
print("📊 MÉTRIQUES NIVEAU PAGE (vote majoritaire)")
print("═"*55)
print(f"Nombre de pages : {len(df_pages)}")
print(classification_report(
    df_pages["y_true"], df_pages["y_pred"],
    target_names=["FAKE", "AUTHENTIC"],
    labels=[0, 1],
    digits=4,
))


# ──────────────────────────────────────────────────────────────
# 6. MATRICES DE CONFUSION + COURBES D'APPRENTISSAGE
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("Résultats — ResNet18 Paléographie", fontsize=14, fontweight="bold")

# ── Confusion matrices ─────────────────────────────────────
for ax, (y_true, y_pred, title) in zip(axes[:2], [
    (df_test["label"],    df_test["y_pred"],   "Confusion Matrix — Blocs"),
    (df_pages["y_true"],  df_pages["y_pred"],  "Confusion Matrix — Pages"),
]):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["FAKE", "AUTHENTIC"],
        yticklabels=["FAKE", "AUTHENTIC"],
        ax=ax, cbar=False,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")

# ── Courbe Accuracy ────────────────────────────────────────
ax3 = axes[2]
epochs = [h["epoch"] for h in history]
ax3.plot(epochs, [h["train_acc"] for h in history], "o-", label="Train Acc", color="#1f77b4")
ax3.plot(epochs, [h["val_acc"]   for h in history], "s--", label="Val Acc",  color="#ff7f0e")
ax3.set_title("Accuracy par époque")
ax3.set_xlabel("Époque")
ax3.set_ylabel("Accuracy")
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── Courbe Val Loss ────────────────────────────────────────
ax4 = axes[3]
ax4.plot(epochs, [h["val_loss"] for h in history], "s--", label="Val Loss", color="#d62728")
ax4.set_title("Val Loss par époque")
ax4.set_xlabel("Époque")
ax4.set_ylabel("Loss")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = CONFIG["out_dir"] / "results.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nFigure sauvegardée → {fig_path}")

# Sauvegarder l'historique d'entraînement
pd.DataFrame(history).to_csv(CONFIG["out_dir"] / "training_history.csv", index=False)
print("training_history.csv sauvegardé")

# Sauvegarder les prédictions détaillées (test)
df_test.to_csv(CONFIG["out_dir"] / "test_predictions_blocs.csv", index=False)
df_pages.to_csv(CONFIG["out_dir"] / "test_predictions_pages.csv", index=False)
print(" Prédictions test (blocs + pages) sauvegardées")
print("\n Pipeline complet terminé.")