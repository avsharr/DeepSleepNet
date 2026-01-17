"""
Script for visualizing model metrics: confusion matrix, precision, recall, F1-score and others.
"""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, cohen_kappa_score
)

from common import get_model_path, run_predictions
from datasets import SequentialSleepDataset
from models import DeepSleepNet
from preprocessing import CLASS_NAMES

# --- Configuration ---
BATCH_SIZE = 16
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except Exception:
    plt.style.use('default')
sns.set_palette("husl")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Visualize confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalization for percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute values
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Absolute Values)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Classes', fontsize=12)
    ax1.set_ylabel('True Classes', fontsize=12)
    
    # Percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Classes', fontsize=12)
    ax2.set_ylabel('True Classes', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()


def plot_metrics_per_class(y_true, y_pred, class_names, save_path='metrics_per_class.png'):
    """Visualize metrics for each class"""
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    # Add values on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    ax.set_title('Metrics per Class (Precision, Recall, F1-Score)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics per class saved: {save_path}")
    plt.close()


def plot_class_distribution(y_true, y_pred, class_names, save_path='class_distribution.png'):
    """Visualize class distribution (true vs predicted)"""
    true_counts = np.bincount(y_true, minlength=len(class_names))
    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, true_counts, width, label='True', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8, color='coral')
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Number of Epochs', fontsize=12)
    ax.set_title('Class Distribution: True vs Predicted', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Class distribution saved: {save_path}")
    plt.close()


def print_detailed_metrics(y_true, y_pred, class_names):
    """Print detailed metrics to console"""
    print("\n" + "=" * 70)
    print("DETAILED MODEL METRICS")
    print("=" * 70)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"   Accuracy:                     {accuracy * 100:.2f}%")
    print(f"   Cohen's Kappa:                {kappa:.4f}")
    
    # Metrics per class
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Count examples of each class
    class_counts = np.bincount(y_true, minlength=len(class_names))
    
    print(f"\n📈 METRICS PER CLASS:")
    print(f"{'Class':<10} {'Count':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    
    for i, name in enumerate(class_names):
        print(f"{name:<10} {class_counts[i]:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")
    
    # Average metrics
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 AVERAGE METRICS:")
    print(f"   Macro Average:")
    print(f"      Precision: {macro_precision:.4f}")
    print(f"      Recall:    {macro_recall:.4f}")
    print(f"      F1-Score:  {macro_f1:.4f}")
    print(f"   Weighted Average:")
    print(f"      Precision: {weighted_precision:.4f}")
    print(f"      Recall:    {weighted_recall:.4f}")
    print(f"      F1-Score:  {weighted_f1:.4f}")
    
    print("\n" + "=" * 70)
    
    # Classification report from sklearn
    print("\n📋 CLASSIFICATION REPORT (sklearn):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def main():
    data_path = os.path.join(ROOT, 'data', 'preprocessed')
    figures_dir = os.path.join(ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    model_path = get_model_path(ROOT, prefer_best=True)
    if model_path is None:
        print("Error: Model file not found. Searched: checkpoints/ and root.")
        return
    print("Using best model" if "best" in model_path else "Using final model")

    print("\nLoading test dataset...")
    test_ds = SequentialSleepDataset(data_path, mode='test', normalize=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Loaded {len(test_ds)} sequences")

    print("Loading model...")
    model = DeepSleepNet(n_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("✓ Model loaded")

    print("\nRunning predictions...")
    all_labels, all_preds = run_predictions(model, test_loader, DEVICE)
    print(f"✓ Predictions completed. Total epochs: {len(all_labels)}")

    print_detailed_metrics(all_labels, all_preds, CLASS_NAMES)

    print("\n" + "=" * 70)
    print("Creating visualizations...")
    print("=" * 70)
    plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES,
                         save_path=os.path.join(figures_dir, 'confusion_matrix.png'))
    plot_metrics_per_class(all_labels, all_preds, CLASS_NAMES,
                          save_path=os.path.join(figures_dir, 'metrics_per_class.png'))
    plot_class_distribution(all_labels, all_preds, CLASS_NAMES,
                            save_path=os.path.join(figures_dir, 'class_distribution.png'))
    print(f"\n✓ All visualizations saved to {figures_dir}")


if __name__ == "__main__":
    main()
