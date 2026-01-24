import os
import torch
import numpy as np
from torch.utils.data import Dataset


class SequentialSleepDataset(Dataset):
    def __init__(self, data_dir, mode='train', seq_length=25, normalize=True):
        self.sequences = []
        self.normalize = normalize

        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        split1 = int(len(files) * 0.7)
        split2 = int(len(files) * 0.85)
        
        if mode == 'train':
            files = files[:split1]
        elif mode == 'val':
            files = files[split1:split2]
        else:
            files = files[split2:]

        if normalize:
            all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
            train_files = all_files[:split1]
            
            all_data = [np.load(os.path.join(data_dir, f))['x'] for f in train_files]
            data_concat = np.concatenate(all_data, axis=0)
            self.global_mean = np.mean(data_concat)
            self.global_std = np.std(data_concat)

        for f in files:
            data = np.load(os.path.join(data_dir, f))
            x, y = data['x'], data['y']

            if normalize:
                x = (x - self.global_mean) / (self.global_std + 1e-8)

            n_seqs = len(x) // seq_length
            for i in range(n_seqs):
                start = i * seq_length
                self.sequences.append((x[start:start+seq_length], y[start:start+seq_length]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x_seq, y_seq = self.sequences[idx]
        return torch.from_numpy(x_seq).float(), torch.from_numpy(y_seq).long()
