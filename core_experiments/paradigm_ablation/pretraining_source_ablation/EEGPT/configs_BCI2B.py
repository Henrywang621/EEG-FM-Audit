from torch.utils.data import DataLoader
import torch
import torchvision
import math
import random
from dataloader import *

def load_fn(x):
    x = torch.load(x)
    
    window_length = 4*256  
    data_length = x.shape[1]  

    # Calculate the maximum starting index for the windows
    max_start_index = data_length - window_length

    # Generate random indices
    if max_start_index>0:
        index = random.randint(0, max_start_index)
        x = x[:, index:index+window_length]
    x = x.to(torch.float)
    return x

max_epochs = 200
max_lr = 5e-4
batch_size=64
devices=[0]


subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
train_dataset = BCICIV2bLoader(
    subject_ids=subjects,
    window_index=8,
    window_size=500,
    step_size=100,
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    num_workers=4,        # Reduced from 7
    shuffle=True,         # IMPORTANT: shuffle for training!
    persistent_workers=True,
    pin_memory=True
)

# Validation dataloader  
val_loader = DataLoader(
    train_dataset,          # Different dataset!
    batch_size=batch_size, 
    num_workers=0,        # Fix for sanity check hang
    shuffle=False,        # Don't shuffle validation
    persistent_workers=False,
    pin_memory=True
)


steps_per_epoch = math.ceil(len(train_loader)/len(devices))


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



        
