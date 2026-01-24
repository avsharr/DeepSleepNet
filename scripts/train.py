import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
from preprocessing import compute_class_weights
from datasets import SequentialSleepDataset
from models import DeepSleepNet

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PATIENCE = 7

logger.info(f"Device: {DEVICE}")


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for signals, labels in tqdm(loader, desc="Training", leave=False):
        signals = signals.to(DEVICE)
        labels = labels.to(DEVICE).view(-1)

        outputs = model(signals)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += loss.item()
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    return loss_sum / len(loader), 100 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE).view(-1)

            outputs = model(signals)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return loss_sum / len(loader), 100 * correct / total


def get_class_weights(data_path):
    train_ds = SequentialSleepDataset(data_path, mode='train', normalize=False)
    labels = np.concatenate([train_ds[i][1].numpy() for i in range(len(train_ds))])
    weights = compute_class_weights(labels)
    logger.info(f"Weights: {weights}")
    logger.info(f"Distribution: {np.bincount(labels, minlength=5)}")
    return weights


def main():
    data_path = os.path.join(ROOT, 'data', 'preprocessed')

    train_ds = SequentialSleepDataset(data_path, mode='train', normalize=True)
    val_ds = SequentialSleepDataset(data_path, mode='val', normalize=True)
    test_ds = SequentialSleepDataset(data_path, mode='test', normalize=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepSleepNet(n_classes=5).to(DEVICE)
    weights = torch.tensor(get_class_weights(data_path), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)

    ckpt_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "deepsleepnet_best_model.pth")

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")

        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = evaluate(model, val_loader, criterion)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(v_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            logger.info(f"LR reduced: {old_lr:.2e} -> {new_lr:.2e}")

        logger.info(f"Train loss: {t_loss:.4f}, acc: {t_acc:.2f}%")
        logger.info(f"Val loss: {v_loss:.4f}, acc: {v_acc:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            logger.info(f"Best model saved: {v_acc:.2f}%")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1}, best: {best_val_acc:.2f}%")
                break

    model.load_state_dict(torch.load(best_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    logger.info(f"Test loss: {test_loss:.4f}, acc: {test_acc:.2f}%")

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "deepsleepnet_model.pth"))
    logger.info("Model saved")

if __name__ == "__main__":
    main()
