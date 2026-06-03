from torch.utils.data import DataLoader
import torch
import torchvision
import math
import random
from dataloader import *

class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200, target_length=None, use_eegpt_channels=True):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        
        # TUAB has 23 channels, but we only use channels that directly match EEGPT
        # TUAB channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, 
        #                T3, T4, T5, T6, Fz, Cz, Pz, A1, A2, Fpz, Oz
        # Keep: 0,1,2,3,4,5,6,7,8,9,10,11,16,17,18,21,22 (17 channels)
        # Exclude: 12(T3), 13(T4), 14(T5), 15(T6), 19(A1), 20(A2)
        if use_eegpt_channels:
            self.channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 21, 22]
        else:
            self.channel_indices = None  # Use all 23 channels
        
        self.chans_num = len(self.channel_indices) if self.channel_indices else 23

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        
        # Resample to desired sampling rate
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        
        # Select only EEGPT-matching channels (17 channels)
        if self.channel_indices is not None:
            X = X[self.channel_indices, :]
        
        # Downsample to target length if specified
        if self.target_length is not None:
            current_length = X.shape[-1]
            if current_length != self.target_length:
                X = resample(X, self.target_length, axis=-1)
        
        Y = sample["y"]
        X = torch.FloatTensor(X)
        
        return X, Y
    
def prepare_TUAB_dataset(root, target_length=None, use_eegpt_channels=True):
    """
    Prepare TUAB dataset
    Args:
        root: Root directory containing train/val/test folders
        target_length: Target sequence length for downsampling (e.g., 1024)
        use_eegpt_channels: If True, only keep 17 channels matching EEGPT montage
    """
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(f"Files: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    if target_length is not None:
        print(f"Downsampling to {target_length} time points")
    if use_eegpt_channels:
        print(f"Using EEGPT channel subset: 17 channels (excluding T3, T4, T5, T6, A1, A2)")

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files, 
                               target_length=target_length, 
                               use_eegpt_channels=use_eegpt_channels)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files,
                              target_length=target_length, 
                              use_eegpt_channels=use_eegpt_channels)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files,
                             target_length=target_length, 
                             use_eegpt_channels=use_eegpt_channels)
    
    return train_dataset, test_dataset, val_dataset

max_epochs = 200
max_lr = 5e-4
batch_size=64
devices=[0]


train_dataset, test_dataset, val_dataset = prepare_TUAB_dataset(
    "/mnt/scratch2/users/xw2336/LLM_Eva/Data/TUAB/v3.0.1/edf/processed",
    target_length=1024,
    use_eegpt_channels=True
)

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



        
