import torch
import torch.nn as nn
import torch.nn.functional as F
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

# 🚨 IMPORT TS_SEFFNet 🚨
from TS_SEFFNet import TS_SEFFNet

# ==========================================================
# DATA LOADING WITH RESAMPLING (Updated for TS_SEFFNet)
# ==========================================================
class TUEVLoader(Dataset):
    def __init__(self, root, files, target_len=1125):
        self.root = root
        self.files = files
        self.target_len = target_len

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        
        X = torch.tensor(sample["signal"], dtype=torch.float32) 
        
        # 🚨 RESAMPLING STEP 🚨
        # Interpolate expects (Batch, Channels, Length), so we unsqueeze(0)
        X = X.unsqueeze(0) 
        X = F.interpolate(X, size=self.target_len, mode='linear', align_corners=False)
        X = X.squeeze(0) # Back to [23, 1125]
        
        y = int(sample["label"][0] - 1)
        return X, torch.tensor(y, dtype=torch.long)

# ==========================================================
# TUNING FUNCTION
# ==========================================================
def train_tune_tsseffnet(config): 
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    root_path = config["root_path"]
    train_dir = os.path.join(root_path, "processed_train")
    val_dir = os.path.join(root_path, "processed_eval")
    
    # Passing target_len=1125 to the loader
    train_ds = TUEVLoader(train_dir, sorted(os.listdir(train_dir)), target_len=1125)
    val_ds = TUEVLoader(val_dir, sorted(os.listdir(val_dir)), target_len=1125)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    
    model = TS_SEFFNet(
        in_chans=23,                  
        n_classes=6,                
        drop_prob=config["drop_prob"],
        batch_norm_alpha=config["batch_norm_alpha"],
        reduction_ratio=config["reduction_ratio"],
        pool_stride=config["pool_stride"], 
        conv_stride=config["conv_stride"]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.NLLLoss() # TS_SEFFNet uses log_softmax
    
    for epoch in range(config["epochs"]):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            # Input format: (Batch, Channels, Time, 1)
            X = X.unsqueeze(-1) 
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.unsqueeze(-1).to(device)
                logits = model(X)
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
    RESULT_STORAGE = "/homes/xw2336/data_portal/ASHA/TUEV/ray_results_TSSEFFNet"
    CONFIG_SAVE_PATH = "best_config_tsseffnet_tuev.json"

    ray.init(ignore_reinit_error=True)

    print("Starting TS_SEFFNet Tuning with 1125-point resampling...")
    
    search_space = {
        "root_path": TUEV_ROOT,
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "epochs": tune.choice([50, 80, 100]), 
        "drop_prob": tune.choice([0.25, 0.5, 0.75]),
        "batch_norm_alpha": tune.choice([0.1, 0.01]),
        "reduction_ratio": tune.choice([4, 8]),
        "pool_stride": tune.choice([3]),  
        "conv_stride": tune.choice([1])   
    }

    asha_scheduler = ASHAScheduler(
        metric="balanced_accuracy",
        mode="max",
        max_t=100,
        grace_period=20,
        reduction_factor=2
    )

    analysis = tune.run(
        train_tune_tsseffnet, 
        storage_path=RESULT_STORAGE,
        config=search_space,
        scheduler=asha_scheduler,
        num_samples=50, 
        resources_per_trial={"cpu": 2, "gpu": 1},
        max_concurrent_trials=2
    )
    
    best_config = analysis.get_best_config(metric="balanced_accuracy", mode="max")
    
    print("\n" + "="*30)
    print(f"OPTIMAL CONFIG FOUND:\n{json.dumps(best_config, indent=4)}")
    print("="*30)

    with open(CONFIG_SAVE_PATH, "w") as f:
        json.dump(best_config, f)
    
    elapsed = time.time() - start_time
    print(f"\nTuning Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.2f}s")