import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

# 🚨 IMPORT CSPNET 🚨
# Ensure CSPNet.py is in the same directory
from CSPNet import CSPNet

# ==========================================================
# DATA LOADING (Retained from your TUEV script)
# ==========================================================
class TUEVLoader(Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        X = sample["signal"]  # Expected shape: [Channels, Time]
        y = int(sample["label"][0] - 1) # 0-5 for TUEV 6 classes
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ==========================================================
# TUNING FUNCTION (Adapted from train2.py)
# ==========================================================
def train_tune_cspnet(config): 
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset Paths
    root_path = config["root_path"]
    train_dir = os.path.join(root_path, "processed_train")
    val_dir = os.path.join(root_path, "processed_eval")
    
    train_ds = TUEVLoader(train_dir, sorted(os.listdir(train_dir)))
    val_ds = TUEVLoader(val_dir, sorted(os.listdir(val_dir)))
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize CSPNet with TUEV specific dimensions
    # TUEV resampled is 2000 samples, 23 channels, 6 classes
    model = CSPNet(
        chunk_size=2000,                 
        num_electrodes=23,               
        num_classes=6,                  
        num_filters_t=config["num_filters_t"],
        filter_size_t=config["filter_size_t"],
        num_filters_s=config["num_filters_s"],
        pool_size_1=config["pool_size_1"],
        pool_stride_1=config["pool_stride_1"],
        dropout=config["dropout"]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    criterion = nn.NLLLoss()
    
    for epoch in range(config["epochs"]):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            X = X.unsqueeze(1) 
            
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(X).squeeze(-1).squeeze(-1)
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
        
        # Validation (Monitoring Balanced Accuracy)
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.unsqueeze(1).to(device)
                logits = model(X).squeeze(-1).squeeze(-1)
                
                probs = torch.exp(logits)
                preds.extend(torch.argmax(probs, 1).cpu().numpy())
                targets.extend(y.numpy())
        
        bacc = balanced_accuracy_score(targets, preds)
        
        tune.report({"balanced_accuracy": bacc}) 

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    start_time = time.time()
    
    TUEV_ROOT = "/homes/xw2336/data_portal/TUEV/resampled_2000"
    RESULT_STORAGE = "/homes/xw2336/data_portal/ASHA/TUEV/ray_results_CSPNet"
    CONFIG_SAVE_PATH = "best_config_cspnet_tuev.json"

    ray.init(ignore_reinit_error=True)

    print("Starting CSPNet Hyperparameter Tuning on TUEV (ASHA)...")
    
    search_space = {
        "root_path": TUEV_ROOT,
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "epochs": tune.choice([50, 80, 100]), 
        
        # --- Tunable CSPNet Parameters ---
        "num_filters_t": tune.choice([10, 20, 30]),
        "filter_size_t": tune.choice([15, 25, 35]),
        "num_filters_s": tune.choice([2, 4]),
        "pool_size_1": tune.choice([50, 100]),
        "pool_stride_1": tune.choice([25, 50]),
        "dropout": tune.choice([0.25, 0.5, 0.75]),
    }

    asha_scheduler = ASHAScheduler(
        metric="balanced_accuracy",
        mode="max",
        max_t=100,
        grace_period=20,
        reduction_factor=2
    )

    analysis = tune.run(
        train_tune_cspnet, 
        storage_path=RESULT_STORAGE,
        config=search_space,
        scheduler=asha_scheduler,
        num_samples=50, 
        resources_per_trial={"cpu": 2, "gpu": 1},
        max_concurrent_trials=2
    )
    
    best_config = analysis.get_best_config(metric="balanced_accuracy", mode="max")
    
    print("\n" + "="*30)
    print(f"OPTIMAL CSPNET CONFIG:\n{json.dumps(best_config, indent=4)}")
    print("="*30)

    with open(CONFIG_SAVE_PATH, "w") as f:
        json.dump(best_config, f)
    
    elapsed = time.time() - start_time
    print(f"\nTuning Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.2f}s")