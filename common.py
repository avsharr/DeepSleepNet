"""
Shared helpers for scripts: model path resolution, prediction loop.
Import after: sys.path.insert(0, ROOT) so this module is found.
"""
import os
import numpy as np
import torch


def get_model_path(root_dir, prefer_best=True):
    """
    Resolve model checkpoint path: checkpoints/ first, then root (backward compatibility).
    prefer_best: if True, prefer deepsleepnet_best_model.pth over deepsleepnet_model.pth.
    Returns path string or None if not found.
    """
    ckpt = os.path.join(root_dir, 'checkpoints')
    root = root_dir
    if prefer_best:
        candidates = [
            os.path.join(ckpt, 'deepsleepnet_best_model.pth'),
            os.path.join(ckpt, 'deepsleepnet_model.pth'),
            os.path.join(root, 'deepsleepnet_best_model.pth'),
            os.path.join(root, 'deepsleepnet_model.pth'),
        ]
    else:
        candidates = [
            os.path.join(ckpt, 'deepsleepnet_model.pth'),
            os.path.join(root, 'deepsleepnet_model.pth'),
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def run_predictions(model, loader, device):
    """Run model on loader, return (all_labels, all_preds) as numpy arrays."""
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)
            labels_flat = labels.view(-1)
            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels_flat.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)
