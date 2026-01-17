import os
import torch
import numpy as np
from torch.utils.data import Dataset


class SequentialSleepDataset(Dataset):
    def __init__(self, data_dir, mode='train', seq_length=25, normalize=True):
        self.sequences = []
        self.normalize = normalize

        # Get all .npz files
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])

        # Split train (70%) / val (15%) / test (15%)
        split_idx1 = int(len(files) * 0.7)
        split_idx2 = int(len(files) * 0.85)
        
        if mode == 'train':
            files = files[:split_idx1]
        elif mode == 'val':
            files = files[split_idx1:split_idx2]
        else:  # test
            files = files[split_idx2:]

        print(f"Loading {len(files)} files for {mode}")

        # Collect statistics for normalization (only from train)
        if normalize:
            # Always use train data to compute statistics
            all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
            train_files_for_stats = all_files[:split_idx1]
            
            all_data = []
            for f in train_files_for_stats:
                path = os.path.join(data_dir, f)
                data = np.load(path)
                all_data.append(data['x'])
            
            # Compute global mean and std for normalization
            all_data_concat = np.concatenate(all_data, axis=0)
            self.global_mean = np.mean(all_data_concat)
            self.global_std = np.std(all_data_concat)
            
            if mode == 'train':
                print(f"Global stats (from train) - Mean: {self.global_mean:.4f}, Std: {self.global_std:.4f}")

        for f in files:
            path = os.path.join(data_dir, f)
            data = np.load(path)
            x_data = data['x']
            y_data = data['y']

            # Data normalization (Z-score)
            if normalize:
                x_data = (x_data - self.global_mean) / (self.global_std + 1e-8)

            # Slice into sequences
            num_seqs = len(x_data) // seq_length
            for i in range(num_seqs):
                start = i * seq_length
                end = start + seq_length

                # Append pair (Input, Label)
                self.sequences.append((x_data[start:end], y_data[start:end]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x_seq, y_seq = self.sequences[idx]
        # Convert to appropriate PyTorch types (Float for data, Long for labels)
        return torch.from_numpy(x_seq).float(), torch.from_numpy(y_seq).long()


if __name__ == "__main__":
    # project root (parent of datasets/)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, 'data', 'preprocessed')

    if os.path.exists(data_path):
        ds = SequentialSleepDataset(data_path, mode='train')
        if len(ds) > 0:
            print(f"Dataset check passed. Total sequences: {len(ds)}")
            print(f"Input shape: {ds[0][0].shape}")
    else:
        print("Data directory not found.")
