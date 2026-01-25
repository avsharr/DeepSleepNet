import sys
import os
import logging
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from common import get_model_path, run_predictions
from datasets import SequentialSleepDataset
from models import DeepSleepNet
from preprocessing import CLASS_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

BATCH_SIZE = 16
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    data_path = os.path.join(ROOT, 'data', 'preprocessed')
    model_path = get_model_path(ROOT, prefer_best=True)

    if model_path is None:
        logger.error("Model checkpoint not found")
        return

    logger.info(f"Loading model: {model_path}")

    test_ds = SequentialSleepDataset(data_path, mode='test')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepSleepNet(n_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    labels, preds = run_predictions(model, test_loader, DEVICE)

    acc = accuracy_score(labels, preds)
    logger.info(f"Accuracy: {acc * 100:.2f}%")
    logger.info(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    logger.info(f"Confusion matrix:\n{confusion_matrix(labels, preds)}")


if __name__ == "__main__":
    main()
