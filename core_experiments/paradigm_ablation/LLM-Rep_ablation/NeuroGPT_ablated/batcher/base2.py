#!/usr/bin/env python3
from typing import Dict, List, Optional
import numpy as np
# import webdataset as wds
import torch
# import gzip
# import pickle
import h5py
import os
# import webdataset as wds
import pickle

from torch.utils.data import Dataset

def _pad_seq_right_to_n(seq: np.ndarray, n: int, pad_value: float = 0.) -> np.ndarray:
    if n == seq.shape[0]:
        return seq
    return np.concatenate(
        [
            seq,
            np.ones(
                (
                    n - seq.shape[0],
                    *seq.shape[1:]
                )
            ) * pad_value,
        ],
        axis=0,
    )


class EEGDataset(Dataset):
    def __init__(
        self,
        filenames: Optional[List[str]] = None,
        sample_keys: Optional[List[str]] = None,
        chunk_len: int = 500,
        num_chunks: int = 10,
        ovlp: int = 50,
        root_path: str = "",
        population_mean: float = 0,
        population_std: float = 1,
        gpt_only: bool = False,
        normalization: bool = True,
        start_samp_pnt: int = -1,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ):
        """
        General EEG dataset that either:
         - loads data from a list of filenames (each file contains one trial/subject),
         - or accepts preloaded X,y numpy arrays (trials, channels, time).
        """
        # If preloaded arrays provided, prefer them
        self.X = None
        self.y = None
        if X is not None and y is not None:
            assert len(X) == len(y), "X and y must have same first dimension (n_samples)."
            self.X = X
            self.y = y
            self.filenames = []
        else:
            # filenames mode
            if filenames is None:
                filenames = []
            if root_path != "":
                # filter only existing files
                self.filenames = [os.path.join(root_path, fn) for fn in filenames if os.path.isfile(os.path.join(root_path, fn))]
            else:
                self.filenames = filenames
        print("Number of files / subjects loaded: ", len(self.filenames) if self.filenames else (len(self.X) if self.X is not None else 0))

        # params
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.ovlp = ovlp
        self.sample_keys = sample_keys
        self.mean = population_mean
        self.std = population_std
        self.do_normalization = normalization
        self.gpt_only = gpt_only
        self.start_samp_pnt = start_samp_pnt

    def __len__(self):
        if self.X is not None:
            return len(self.X)
        else:
            # If only filenames are provided, treat each filename as one example
            return len(self.filenames)

    def __getitem__(self, idx):
        # Two modes: preloaded arrays or file loading
        if self.X is not None:
            data = self.X[idx]  # expected shape: (channels, time)
            label = self.y[idx]
        else:
            # load from file (pickle/hdf5/np)
            fn = self.filenames[idx]
            # try np.load, pickle, h5py in order
            if fn.endswith(".npy"):
                loaded = np.load(fn)
            elif fn.endswith(".npz"):
                loaded = np.load(fn)
                # if stored as dict-like with X, y
                if 'X' in loaded and 'y' in loaded:
                    loaded = loaded['X']
                else:
                    # assume array saved
                    loaded = loaded
            elif fn.endswith(".pkl") or fn.endswith(".pickle"):
                loaded = pickle.load(open(fn, "rb"))
            elif fn.endswith(".h5") or fn.endswith(".hdf5"):
                # implement custom loader or reuse load_single_file
                loaded = np.array(self.load_single_file(fn))
            else:
                # fallback: try np.load
                loaded = np.load(fn)

            # If file contains dict-like with X,y, handle that externally.
            # For compatibility with older code that used `data = data[:22]`
            # assume loaded is an array shaped (channels, time) or (n_channels, n_time)
            # Some files might have shape (trials, channels, time) — handle the simplest case
            if isinstance(loaded, dict):
                # If dict format: { "X": ..., "y": ... }
                data = loaded.get("X", None)
                label = loaded.get("y", None)
            else:
                data = loaded
                label = None

            # If label is None and file naming encodes label, user should override this behavior
        # Reorder channels if necessary (this function expects channels x time)
        data = self.reorder_channels(data)
        # pass label to preprocess so it gets included
        return self.preprocess_sample(data, seq_len=self.num_chunks, labels=label)

    # helper wrapper to your util
    @staticmethod
    def _pad_seq_right_to_n(seq: np.ndarray, n: int, pad_value: float = 0) -> np.ndarray:
        return _pad_seq_right_to_n(seq=seq, n=n, pad_value=pad_value)

    def load_single_file(self, filename):
        with h5py.File(filename, 'r') as file:
            data_dict = file['Result']
            data = []
            for i in range(data_dict['data'].shape[0]):
                ref = data_dict['data'][i][0]
                time_series = data_dict[ref]
                if len(data) > 0 and time_series.shape[0] < data[0].shape[0]:
                    time_series = np.zeros_like(data[0])
                data.append(np.array(time_series).squeeze())
        return data

    def load_tensor(self, filename):
        tensor_data = torch.load(filename)
        return tensor_data.numpy()

    def reorder_channels(self, data):
        """
        Default reorder: if data already has channels as first dim, keep it.
        If your data is 1D or different shape, modify this function.
        """
        # expect data shape (channels, time)
        data = np.asarray(data)
        if data.ndim == 1:
            # single channel
            return data[np.newaxis, :]
        if data.ndim == 2:
            # already (channels, time)
            return data
        # if shape is (time, channels) convert it
        if data.shape[0] < data.shape[1]:
            # heuristic — swap if needed (only when ambiguous)
            return data.T
        return data

    def split_chunks(self, data, length=500, ovlp=50, num_chunks=10, start_point=-1):
        all_chunks = []
        total_len = data.shape[1]
        actual_num_chunks = num_chunks

        if start_point == -1:
            if num_chunks * length > total_len - 1:
                start_point = 0
                actual_num_chunks = max(1, total_len // length)
            else:
                start_point = np.random.randint(0, max(1, total_len - num_chunks * length))

        for i in range(actual_num_chunks):
            chunk = data[:, start_point: start_point + length]
            all_chunks.append(np.array(chunk))
            start_point = start_point + length - ovlp
        return np.array(all_chunks), start_point

    def normalize(self, data):
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        return (data - mean) / (std + 1e-25)

    def preprocess_sample(self, sample, seq_len, labels=None) -> Dict[str, torch.Tensor]:
        out = {}
        if self.do_normalization:
            sample = self.normalize(sample)

        chunks, seq_on = self.split_chunks(sample, self.chunk_len, self.ovlp, seq_len, self.start_samp_pnt)

        attention_mask = np.ones(seq_len)
        chunks = self._pad_seq_right_to_n(seq=chunks, n=seq_len, pad_value=0)
        attention_mask = self._pad_seq_right_to_n(seq=attention_mask, n=seq_len, pad_value=0)

        if self.gpt_only:
            chunks = np.reshape(chunks, (seq_len, chunks.shape[1] * chunks.shape[2]))

        out["inputs"] = torch.from_numpy(chunks).to(torch.float)
        out["attention_mask"] = torch.from_numpy(attention_mask).to(torch.long)
        out['seq_on'] = seq_on
        out['seq_len'] = seq_len

        if labels is not None:
            # --- MODIFIED BLOCK START ---
            lbl_np = np.array(labels)
            
            # If label is a scalar (e.g., 0 or 1), it has ndim=0.
            # Reshape it to (1,) so it batches into [Batch_Size, 1]
            if lbl_np.ndim == 0:
                lbl_np = lbl_np.reshape(1)
            
            # NOTE: If using BCEWithLogitsLoss, labels usually need to be float. 
            # If using CrossEntropyLoss, labels usually need to be long.
            # Since your error was size [32, 1], you likely need Long for Class or Float for Binary.
            # I will keep torch.long as per your original code, but you might need .float() if doing binary.
            out['labels'] = torch.from_numpy(lbl_np).to(torch.long)
            # --- MODIFIED BLOCK END ---

        if self.sample_keys is not None:
            out = {key: out[key] for key in self.sample_keys if key in out}

        return out