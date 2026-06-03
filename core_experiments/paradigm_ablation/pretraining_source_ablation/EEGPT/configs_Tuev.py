from torch.utils.data import DataLoader
import torch
import torchvision
import math
import random
from dataloader import *

# EEGPT original channels
eegpt_channels = [
    'FP1', 'FPZ', 'FP2',
    "AF7", 'AF3', 'AF4', "AF8",
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8',
    'O1', 'OZ', 'O2'
]

# TUEV channels (23 total)
tuev_channel_names = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'
]

# Only keep channels that directly match (no conversions)
# 15 overlapping channels: FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, FZ, CZ, PZ
use_channels_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'FZ', 'CZ', 'PZ']

# Get indices of channels to keep from TUEV data
tuev_channels_to_keep = [i for i, ch in enumerate(tuev_channel_names) if ch in use_channels_names]


class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, target_length=1024):
        self.root = root
        self.files = files
        self.target_length = target_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        import pickle
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        
        X = sample["signal"]
        
        if X.dtype.byteorder == '>' or (X.dtype.byteorder == '|' and X.dtype.str.endswith('f4') == False):
            X = X.byteswap().newbyteorder('=')
        
        X = np.array(X, dtype=np.float32, copy=True)
        X = np.ascontiguousarray(X)
        
        # Keep only overlapping channels (15 channels)
        X = X[tuev_channels_to_keep, :]
        
        # Downsample from 2000 to 1024
        X = resample(X, self.target_length, axis=-1)
        X = np.ascontiguousarray(X, dtype=np.float32)
            
        Y = int(sample["label"][0] - 1)  # Labels are 1-6, convert to 0-5
        
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        
        return X_tensor, Y



max_epochs = 400
max_lr = 1e-4
batch_size=64
devices=[0]

root = '/mnt/scratch2/users/xw2336/LLM_Eva/Data/TUEV/v2.0.1/edf/processed'

# Load file lists
train_files = os.listdir(os.path.join(root, "processed_train"))
val_files = os.listdir(os.path.join(root, "processed_eval"))
test_files = os.listdir(os.path.join(root, "processed_test"))

# Create datasets
train_dataset = TUEVLoader(os.path.join(root, "processed_train"), train_files, target_length=1024)
val_dataset = TUEVLoader(os.path.join(root, "processed_eval"), val_files, target_length=1024)
test_dataset = TUEVLoader(os.path.join(root, "processed_test"), test_files, target_length=1024)


# Combine ALL three datasets for training
combined_train = torch.utils.data.ConcatDataset([train_dataset, test_dataset, val_dataset])
combined_loader = torch.utils.data.DataLoader(combined_train, batch_size=batch_size, num_workers=4, shuffle=True)
combined_valid_loader = torch.utils.data.DataLoader(combined_train, batch_size=batch_size, num_workers=4, shuffle=False)

steps_per_epoch = len(combined_loader)
print("steps_per_epoch: " + str(steps_per_epoch))
tag = "large"
variant = "D"

MODELS_CONFIGS = {
    "tiny1": {
        "embed_dim":64, "embed_num":1, "depth":[2,2,4], "num_heads":4},
    "tiny2": {
        "embed_dim":64, "embed_num":4, "depth":[2,2,4], "num_heads":4},
    "tiny3": {
        "embed_dim":64, "embed_num":4, "depth":[8,8,8], "num_heads":4},
    "little": {
        "embed_dim":128, "embed_num":4, "depth":[8,8,8], "num_heads":4},
    "base1": {
        "embed_dim":256, "embed_num":1, "depth":[6,6,6], "num_heads":4},
    "base2": {
        "embed_dim":256, "embed_num":4, "depth":[8,8,8], "num_heads":4},
    "base3": {
        "embed_dim":512, "embed_num":1, "depth":[6,6,6], "num_heads":8},
    "large": {
        "embed_dim":512, "embed_num":4, "depth":[8,8,8], "num_heads":8},
}

def get_config(embed_dim=512, embed_num=4, depth=[8,8,8], num_heads=4):
    
    models_configs = {
            'encoder': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'depth': depth[0],
                    'num_heads': num_heads,
                },
            'predictor': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'predictor_embed_dim': embed_dim,
                    'depth': depth[1],
                    'num_heads': num_heads,
                },
            'reconstructor': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'reconstructor_embed_dim': embed_dim,
                    'depth': depth[2],
                    'num_heads': num_heads,
                },
    }
    return models_configs



        
