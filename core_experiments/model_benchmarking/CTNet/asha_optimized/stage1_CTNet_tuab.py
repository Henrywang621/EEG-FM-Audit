import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import os
import pickle
import json
import ray
from ray import tune 
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import balanced_accuracy_score
import gc
import time

# 🚨 IMPORT CTNET 🚨
# Ensure CTNet.py is in your working directory or PYTHONPATH
from CTNet import EEGTransformer

# ==========================================================
# DATA LOADING & INTERPOLATION
# ==========================================================
class TUABLoaderResampled(Dataset):
    def __init__(self, root, files, target_len=1000):
        self.root = root
        self.files = files
        self.target_len = target_len

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, index):
        try:
            with open(os.path.join(self.root, self.files[index]), "rb") as f:
                sample = pickle.load(f)
            # X shape: (Channels, Time)
            X = torch.tensor(sample["X"], dtype=torch.float32) 
            y = int(sample["y"])
            
            # Resampling to target_len via linear interpolation
            X = X.unsqueeze(0) 
            X = F.interpolate(X, size=self.target_len, mode='linear', align_corners=False)
            X = X.squeeze(0) 
            return X, torch.tensor(y, dtype=torch.long)
        except Exception:
            # Fallback for data loading errors
            return torch.zeros((23, self.target_len)), torch.tensor(0)

# ==========================================================
# TUNING FUNCTION (STABILIZED)
# ==========================================================
def train_tune_ctnet_tuab(config): 
    # Memory management setup
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optional: Reserve 2GB VRAM to reduce fragmentation
    if torch.cuda.is_available():
        squatter_size = (2 * 1024**3) // 4
        memory_squatter = torch.empty(squatter_size, dtype=torch.float32, device=device)

    torch.manual_seed(42)
    np.random.seed(42)
    
    train_dir = os.path.join(config["root_path"], "train")
    test_dir = os.path.join(config["root_path"], "test") 
    
    train_loader = DataLoader(
        TUABLoaderResampled(train_dir, sorted(os.listdir(train_dir))), 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        TUABLoaderResampled(test_dir, sorted(os.listdir(test_dir))), 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    model = EEGTransformer(
        depth=config["depth"],
        eeg1_kernel_size=config["eeg1_kernel_size"],
        eeg1_dropout_rate=config["eeg1_dropout_rate"],
        number_channel=23, 
        n_classes=2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config["epochs"]):
        model.train()
        for X, y in train_loader:
            # CTNet expects input as (Batch, 1, Channels, Length)
            X, y = X.to(device).unsqueeze(1), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                logits = model(X)
                loss = criterion(logits, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.unsqueeze(1).to(device)
                with torch.cuda.amp.autocast():
                    logits = model(X)
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                targets.extend(y.numpy())
        
        # Calculate Balanced Accuracy for BCI datasets
        bacc = balanced_accuracy_score(targets, preds)
        tune.report({"balanced_accuracy": bacc})

    # Strict cleanup to prevent trial-to-trial OOM
    del model, optimizer, memory_squatter
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # Path configuration
    TUAB_ROOT = "/homes/xw2336/data_portal/TUAB/processed"
    if not os.path.exists(TUAB_ROOT):
        TUAB_ROOT = "/homes/xw2336/data_portal/TUAB/TUAB/processed"
    
    RESULTS_DIR = "/homes/xw2336/data_portal/ASHA/TUAB/ray_results_CTNet_Stable1"

    # Initialize Ray for Single GPU usage
    ray.init(
        num_gpus=1, 
        object_store_memory=10 * 1024**3, 
        ignore_reinit_error=True
    )

    search_space = {
        "root_path": TUAB_ROOT,
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "epochs": tune.choice([20, 30, 40, 50]), 
        "depth": tune.choice([2, 4, 6]),
        "eeg1_kernel_size": tune.choice([16, 32, 64]),
        "eeg1_dropout_rate": tune.choice([0.25, 0.5]),
    }

    # Start Tuning
    analysis = tune.run(
        train_tune_ctnet_tuab, 
        storage_path=RESULTS_DIR,
        config=search_space,
        scheduler=ASHAScheduler(
            metric="balanced_accuracy", 
            mode="max", 
            max_t=50, 
            grace_period=15
        ),
        num_samples=50, 
        resources_per_trial={
            "cpu": 4, 
            "gpu": 1,            # 1 GPU per trial
            "memory": 16 * 1024**3  # 16GB System RAM per trial
        }, 
        max_concurrent_trials=1,  # Set to 1 because we only have 1 GPU
        name="CTNet_TUAB_Final_Stabilized1"
    )
    
    # ==========================================================
    # SAVE BEST CONFIGURATION
    # ==========================================================
    best_trial_config = analysis.get_best_config(metric="balanced_accuracy", mode="max")
    
    json_output_path = os.path.join(RESULTS_DIR, "CTNet_TUAB_Final_Stabilized", "best_config_CTNet1.json")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    
    with open(json_output_path, "w") as f:
        json.dump(best_trial_config, f, indent=4)
    
    print(f"\n--- Tuning Complete ---")
    print(f"Best BACC: {analysis.best_result['balanced_accuracy']:.4f}")
    print(f"Best Config saved to: {json_output_path}")
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")