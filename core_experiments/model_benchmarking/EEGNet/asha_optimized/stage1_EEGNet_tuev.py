import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import pickle
import json
import ray
from ray import tune 
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import balanced_accuracy_score
import gc
import time

# ==========================================================
# MODEL DEFINITION (Optimized EEGNet) 
# ==========================================================
class TunableEEGNet(nn.Module):
    def __init__(self, n_classes, in_chans=23, input_len=2000, F1=8, D=2, 
                 dropout=0.25, temp_kernel=101, sep_kernel=25):
        super(TunableEEGNet, self).__init__()
        self.F2 = F1 * D
        
        # Layer 1: Temporal Conv (using dynamic kernel size)
        self.conv1 = nn.Conv2d(1, F1, (1, temp_kernel), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Layer 2: Spatial Conv (Depthwise)
        self.conv2 = nn.Conv2d(F1, self.F2, (in_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.F2)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        
        # Layer 3: Separable Conv (using dynamic kernel size)
        self.conv3 = nn.Conv2d(self.F2, self.F2, (1, sep_kernel), padding='same', groups=self.F2, bias=False)
        self.conv4 = nn.Conv2d(self.F2, self.F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        
        # Dynamic Flatten Size Calculation
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_chans, input_len)
            # Trace the forward path to get output shape
            x = self.bn1(self.conv1(dummy))
            x = self.drop1(self.pool1(self.elu(self.bn2(self.conv2(x)))))
            x = self.drop2(self.pool2(self.elu(self.bn3(self.conv4(self.conv3(x))))))
            self.flatten_size = x.numel()
            
        self.fc = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1) 
        x = self.bn1(self.conv1(x))
        x = self.drop1(self.pool1(self.elu(self.bn2(self.conv2(x)))))
        x = self.drop2(self.pool2(self.elu(self.bn3(self.conv4(self.conv3(x))))))
        return self.fc(x.view(x.size(0), -1))

# ==========================================================
# DATA LOADING
# ==========================================================
class TUEVLoader(Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        X = sample["signal"]
        y = int(sample["label"][0] - 1)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ==========================================================
# TUNING FUNCTION
# ==========================================================
def train_tune(config): 
    # Seed inside the worker for consistency
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    root_path = config["root_path"]
    train_dir = os.path.join(root_path, "processed_train")
    val_dir = os.path.join(root_path, "processed_eval")
    
    train_ds = TUEVLoader(train_dir, sorted(os.listdir(train_dir)))
    val_ds = TUEVLoader(val_dir, sorted(os.listdir(val_dir)))
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    
    model = TunableEEGNet(
        n_classes=6, 
        F1=config["F1"], 
        D=config["D"], 
        dropout=config["dropout"],
        temp_kernel=config["temp_kernel"],
        sep_kernel=config["sep_kernel"]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') 
    
    for epoch in range(config["epochs"]):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device=="cuda"):
                loss = criterion(model(X), y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                out = model(X.to(device))
                preds.extend(torch.argmax(out, 1).cpu().numpy())
                targets.extend(y.numpy())
        
        bacc = balanced_accuracy_score(targets, preds)
        # Report back to Ray Tune
        tune.report({"balanced_accuracy": bacc}) 

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # Path configuration
    TUEV_ROOT = "/homes/xw2336/data_portal/TUEV/resampled_2000"
    RESULT_STORAGE = "/homes/xw2336/data_portal/ASHA/TUEV/ray_results_EEGNet"
    CONFIG_SAVE_PATH = "best_config_eegnet.json"

    ray.init(ignore_reinit_error=True)

    print("Starting Stage 1: Hyperparameter Tuning (Including Kernel Sizes)...")
    
    analysis = tune.run(
        train_tune, 
        storage_path=RESULT_STORAGE,
        config={
            "root_path": TUEV_ROOT, 
            "lr": tune.loguniform(1e-4, 1e-2),
            "weight_decay": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([16, 32, 64]),
            "epochs": tune.choice([50, 80, 100]),
            "dropout": tune.choice([0.25, 0.5]),
            "F1": tune.choice([8, 16]),
            "D": tune.choice([1, 2]),
            # NEW SEARCH PARAMETERS
            # Original: [51, 101, 151] -> Added 125 (Standard) and 201 (Low-freq)   
            "temp_kernel": tune.choice([51, 101, 125, 151, 201]), 
            # Original: [15, 25, 35] -> Added 8 (High-freq) and 45 (Integration)
            "sep_kernel": tune.choice([8, 15, 25, 35, 45])
        },
        scheduler=ASHAScheduler(metric="balanced_accuracy", mode="max", grace_period=10),
        num_samples=50, 
        resources_per_trial={"cpu": 2, "gpu": 1},
        max_concurrent_trials=2
    )
    
    best_config = analysis.get_best_config(metric="balanced_accuracy", mode="max")
    
    print("\n" + "="*30)
    print(f"OPTIMAL CONFIG FOUND:\n{json.dumps(best_config, indent=4)}")
    print("="*30)

    # Save best config for Stage 2
    with open(CONFIG_SAVE_PATH, "w") as f:
        json.dump(best_config, f)
    
    print(f"Configuration saved to {CONFIG_SAVE_PATH}")
    
    elapsed = time.time() - start_time
    print(f"\nTuning Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.2f}s")