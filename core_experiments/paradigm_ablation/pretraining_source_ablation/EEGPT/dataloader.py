import torch
import pickle
import os
from scipy.signal import resample
import numpy as np


def sliding_window_eeg(data, index, window_size, step_size):
    # n_samples, n_channels, n_timepoints = data.shape
    # n_windows = (n_timepoints - window_size) // step_size + 1
    # segmented = np.zeros((n_samples, n_windows, n_channels, window_size))

    start = (index - 1) * step_size
    end = start + window_size

    return data[:, :, start:end]  # shape: (n_samples, n_channels, window_size)


class BCICIV2bLoader(torch.utils.data.Dataset):
    def __init__(self, subject_ids, window_index=8, window_size=500, step_size=100,
                 transform=None, target_length=1024):
        self.subject_ids = subject_ids
        self.window_index = window_index
        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform
        self.target_length = target_length  # New parameter for upsampling target
        self.data_subjs = []
        self.labels_subjs = []

        self.X, self.y = self.load_subject_data(self.subject_ids)

    def load_subject_data(self, subject_ids):

        # Return (X, y) as numpy arrays
        Subj_id = {'S01': 1, 'S02': 2, 'S03': 3, 'S04': 4, 'S05': 5, 'S06': 6, 'S07': 7, 'S08': 8, 'S09': 9}

        for subject_id in subject_ids:
            subject_id = Subj_id[subject_id]

            x1 = np.load("/mnt/scratch2/users/xw2336/LLM_Eva/Data/BCICIV_2b/Subj{0}_1_X.npy".format(subject_id))
            x2 = np.load("/mnt/scratch2/users/xw2336/LLM_Eva/Data/BCICIV_2b/Subj{0}_2_X.npy".format(subject_id))

            y1 = np.load("/mnt/scratch2/users/xw2336/LLM_Eva/Data/BCICIV_2b/Subj{0}_1_y.npy".format(subject_id))
            y2 = np.load("/mnt/scratch2/users/xw2336/LLM_Eva/Data/BCICIV_2b/Subj{0}_2_y.npy".format(subject_id))

            x_combined_data = np.concatenate([x1, x2], axis=0)
            y_combined_labels = np.concatenate([y1, y2], axis=0)
            self.data_subjs.append(x_combined_data)
            self.labels_subjs.append(y_combined_labels)

        x_combined_data = np.concatenate([self.data_subjs[i] for i in range(len(self.data_subjs))], axis=0)
        y_combined_labels = np.concatenate([self.labels_subjs[i] for i in range(len(self.labels_subjs))], axis=0)

        # Apply sliding window
        x_seg_data = sliding_window_eeg(x_combined_data, self.window_index, self.window_size, self.step_size)

        # Upsample from 500 to target_length (1024) using scipy.signal.resample
        if x_seg_data.shape[2] != self.target_length:
            n_samples, n_channels, n_timepoints = x_seg_data.shape
            x_upsampled = np.zeros((n_samples, n_channels, self.target_length))

            for i in range(n_samples):
                for ch in range(n_channels):
                    # Resample each channel independently
                    x_upsampled[i, ch, :] = resample(x_seg_data[i, ch, :], self.target_length)

            x_seg_data = x_upsampled

        return x_seg_data, y_combined_labels


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
