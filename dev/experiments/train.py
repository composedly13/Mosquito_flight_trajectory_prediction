import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from config import *
from preprocess import MosquitoDataset
from model import PhysicsCorrector


def r_hit(pred: np.ndarray, true: np.ndarray) -> float:
    dist = np.linalg.norm(pred - true, axis=-1)
    return float(np.mean(dist <= R_HIT_THRESHOLD))


def train():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = MosquitoDataset(TRAIN_DIR, LABELS_PATH)
    n_val   = int(len(dataset) * VAL_RATIO)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = PhysicsCorrector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=0.005)

    best_hit = 0.0
    pbar = tqdm(range(1, EPOCHS + 1), desc="Training")

    for epoch in pbar:
        # train
        model.train()
        train_loss = 0.0
        for x, physics, y in train_loader:
            x, physics, y = x.to(device), physics.to(device), y.to(device)
            pred = model(x, physics)
            # pred = physics + correction, y = true_abs
            # loss: correction이 (true - physics)에 가까워지도록
            loss = criterion(pred - physics, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        # val
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, physics, y in val_loader:
                pred = model(x.to(device), physics.to(device)).cpu().numpy()
                # true_abs = physics + y (correction)
                true_abs = (physics + y).numpy()
                preds.append(pred)
                trues.append(true_abs)

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        hit   = r_hit(preds, trues)

        scheduler.step()

        if hit > best_hit:
            best_hit = hit
            torch.save(model.state_dict(), MODEL_SAVE)

        pbar.set_postfix(loss=f"{train_loss:.6f}", hit=f"{hit:.4f}", best=f"{best_hit:.4f}")

    print(f"\nBest val R-Hit: {best_hit:.4f}")


if __name__ == "__main__":
    train()
