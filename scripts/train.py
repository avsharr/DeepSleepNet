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
from preprocessing import compute_class_weights as compute_class_weights_from_labels
from datasets import SequentialSleepDataset
from models import DeepSleepNet

# --- Configuration ---
BATCH_SIZE = 16  # Smaller batch size for LSTM sequences
EPOCHS = 50  # Increased with early stopping
LEARNING_RATE = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 7  # Stop if no improvement for 7 epochs

print(f"Using device: {DEVICE}")


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)

    for signals, labels in loop:
        # Move to device
        signals = signals.to(DEVICE)
        labels = labels.to(DEVICE)

        # Flatten labels (Batch * Seq)
        labels_flat = labels.view(-1)

        # Forward
        outputs = model(signals)
        loss = criterion(outputs, labels_flat)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients (important for LSTM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels_flat.size(0)
        correct += (predicted == labels_flat).sum().item()

        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)
            labels_flat = labels.view(-1)

            outputs = model(signals)
            loss = criterion(outputs, labels_flat)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels_flat.size(0)
            correct += (predicted == labels_flat).sum().item()

    return running_loss / len(loader), 100 * correct / total


def compute_class_weights(data_path):
    """Compute class weights automatically based on train data"""
    train_ds = SequentialSleepDataset(data_path, mode='train', normalize=False)
    all_labels = []
    
    for i in range(len(train_ds)):
        _, labels = train_ds[i]
        all_labels.extend(labels.numpy())
    
    all_labels = np.array(all_labels)
    weights_array = compute_class_weights_from_labels(all_labels)
    print(f"Computed class weights: {weights_array}")
    print(f"Class distribution: {np.bincount(all_labels, minlength=5)}")
    
    return weights_array


def main():
    data_path = os.path.join(ROOT, 'data', 'preprocessed')

    # Load Data
    train_ds = SequentialSleepDataset(data_path, mode='train', normalize=True)
    val_ds = SequentialSleepDataset(data_path, mode='val', normalize=True)
    test_ds = SequentialSleepDataset(data_path, mode='test', normalize=True)

    # Shuffle=True mixes patients/sequences, but order INSIDE sequence is preserved
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Init Model
    model = DeepSleepNet(n_classes=5).to(DEVICE)

    # Compute class weights automatically
    class_weights_array = compute_class_weights(data_path)
    class_weights = torch.tensor(class_weights_array, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # weight_decay=1e-3 (L2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    # Early stopping
    best_val_acc = 0.0
    no_improve = 0
    checkpoints_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoints_dir, "deepsleepnet_best_model.pth")

    # Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = evaluate(model, val_loader, criterion)

        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(v_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

        print(f"Train Loss: {t_loss:.4f} | Acc: {t_acc:.2f}%")
        print(f"Val Loss:   {v_loss:.4f} | Acc: {v_acc:.2f}%")

        # Early stopping and save best model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved! Val Acc: {v_acc:.2f}%")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break

    # Load best model for final test
    print("\n" + "="*50)
    print("Loading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print("="*50)

    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, "deepsleepnet_model.pth"))
    print("Final model saved.")


if __name__ == "__main__":
    main()
