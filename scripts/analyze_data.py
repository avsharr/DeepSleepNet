"""
Script for data analysis: class distribution, statistics, visualization.
Helps understand data structure before training.
"""
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from datasets import SequentialSleepDataset
from preprocessing import compute_class_weights, CLASS_NAMES, NUM_CLASSES


def analyze_class_distribution(data_path, mode='train'):
    """Analyze class distribution in the dataset"""
    print(f"Class Distribution Analysis ({mode})")
    
    # load dataset without normalization for analysis
    dataset = SequentialSleepDataset(data_path, mode=mode, normalize=False)
    
    # collect all labels from the dataset
    all_labels = []
    for i in range(len(dataset)):
        _, labels = dataset[i]
        all_labels.extend(labels.numpy())
    
    all_labels = np.array(all_labels)
    total_epochs = len(all_labels)
    
    # count occurrences of each class
    unique_classes, class_counts = np.unique(all_labels, return_counts=True)
    
    # print class distribution table
    print(f"\nClass distribution:")
    print(f"{'Class':<10} {'Count':<15} {'Percentage':<10}")
    
    for cls, count in zip(unique_classes, class_counts):
        percentage = (count / total_epochs) * 100
        print(f"{CLASS_NAMES[int(cls)]:<10} {count:<15} {percentage:.2f}%")
    
    # compute class weights using inverse frequency (from preprocessing)
    class_weights = compute_class_weights(all_labels, num_classes=NUM_CLASSES)
    
    # print recommended weights
    print(f"\nclass weights = inverse frequency method")
    for name, weight in zip(CLASS_NAMES, class_weights):
        print(f"{name}: {weight:.4f}")
    
    # create visualization bar chart
    plt.figure(figsize=(10, 6))
    plt.bar([CLASS_NAMES[int(c)] for c in unique_classes], class_counts, color='steelblue')
    plt.xlabel('class')
    plt.ylabel('number of epochs')
    plt.title(f'class distribution ({mode})')
    plt.grid(axis='y', alpha=0.3)
    
    # add percentage labels on top of each bar
    for i, (cls, count) in enumerate(zip(unique_classes, class_counts)):
        percentage = (count / total_epochs) * 100
        plt.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    figures_dir = os.path.join(ROOT, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, f'class_distribution_{mode}.png')
    plt.savefig(save_path, dpi=150)
    
    return class_weights


def analyze_signal_statistics(data_path, mode='train'):
    """Analyze signal statistics"""
    print(f"Signal Statistics Analysis ({mode})")
    
    # load dataset without normalization
    dataset = SequentialSleepDataset(data_path, mode=mode, normalize=False)
    
    # collect signal data (limit to first 100 sequences for speed)
    all_signals = []
    num_sequences = min(100, len(dataset))
    for i in range(num_sequences):
        signals, _ = dataset[i]
        all_signals.append(signals.numpy())
    
    # concatenate all signals into one array
    all_signals = np.concatenate(all_signals, axis=0)
    
    # compute and print basic statistics
    print(f"Mean   {np.mean(all_signals):.4f}")
    print(f"Std    {np.std(all_signals):.4f}")
    print(f"Min    {np.min(all_signals):.4f}")
    print(f"Max    {np.max(all_signals):.4f}")
    print(f"Median {np.median(all_signals):.4f}")
    
    return np.mean(all_signals), np.std(all_signals)


def main():
    """Main function to run data analysis"""
    data_path = os.path.join(ROOT, 'data', 'preprocessed')
    analyze_class_distribution(data_path, mode='train')
    analyze_signal_statistics(data_path, mode='train')
    analyze_class_distribution(data_path, mode='test')


if __name__ == "__main__":
    main()
