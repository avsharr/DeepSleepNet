import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from common import get_model_path, run_predictions
from datasets import SequentialSleepDataset
from models import DeepSleepNet
from preprocessing import CLASS_NAMES

# --- Configuration ---
BATCH_SIZE = 16
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    data_path = os.path.join(ROOT, 'data', 'preprocessed')
    model_path = get_model_path(ROOT, prefer_best=False)
    if model_path is None:
        print("Error: Model file not found. Searched: checkpoints/ and root.")
        return

    print("Loading test dataset...")
    test_ds = SequentialSleepDataset(data_path, mode='test')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading model...")
    model = DeepSleepNet(n_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print("Running predictions...")
    all_labels, all_preds = run_predictions(model, test_loader, DEVICE)

    print("\n" + "=" * 50)
    print("FINAL EVALUATION REPORT")
    print("=" * 50)

    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

    # Confusion Matrix
    print("Confusion Matrix (Rows=True, Cols=Predicted):")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    print("=" * 50)


if __name__ == "__main__":
    main()
