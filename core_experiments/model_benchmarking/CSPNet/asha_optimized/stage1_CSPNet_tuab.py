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
import time

# 🚨 IMPORT CSPNET 🚨
from CSPNet import CSPNet

# ==========================================================
# DATA LOADING 
# ==========================================================
class TUABLoader(Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        # TUAB Expected shape: (Channels, Time) -> (23, 2000)
        X = torch.FloatTensor(sample["X"]) 
        y = torch.tensor(sample["y"], dtype=torch.long)
        return X, y

# ==========================================================
# TUNING FUNCTION
# ==========================================================
def train_tune_cspnet_tuab(config): 
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset Paths
    root_path = config["root_path"]
    train_dir = os.path.join(root_path, "train")
    val_dir = os.path.join(root_path, "val")
    
    train_ds = TUABLoader(train_dir, sorted(os.listdir(train_dir)))
    val_ds = TUABLoader(val_dir, sorted(os.listdir(val_dir)))
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize CSPNet with TUAB specific dimensions
    # TUAB: 2000 samples, 23 channels, 2 classes (Binary)
    model = CSPNet(
        chunk_size=2000,                 
        num_electrodes=23,               
        num_classes=2,                  
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
        
        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.unsqueeze(1).to(device)
                logits = model(X).squeeze(-1).squeeze(-1)
                
                # Convert log_softmax to probabilities for argmax
                probs = torch.exp(logits)
                preds.extend(torch.argmax(probs, 1).cpu().numpy())
                targets.extend(y.numpy())
        
        # Using Balanced Accuracy as the stable metric for tuning
        bacc = balanced_accuracy_score(targets, preds)
        
        tune.report({"balanced_accuracy": bacc}) 

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    start_time = time.time()
    
   
    TUAB_ROOT = "/homes/xw2336/data_portal/TUAB/processed"
    RESULT_STORAGE = "/homes/xw2336/data_portal/ASHA/TUAB/ray_results_CSPNet"
    CONFIG_SAVE_PATH = "best_config_cspnet_tuab.json"

    ray.init(ignore_reinit_error=True)

    print("Starting CSPNet Hyperparameter Tuning on TUAB (ASHA)...")
    
    # Search space
    search_space = {
        "root_path": TUAB_ROOT,
        "lr": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "epochs": tune.choice([50, 80, 100]), 
        
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
        train_tune_cspnet_tuab, 
        storage_path=RESULT_STORAGE,
        config=search_space,
        scheduler=asha_scheduler,
        num_samples=50, 
        resources_per_trial={"cpu": 2, "gpu": 0.5}, # Adjust based on your server capacity
        max_concurrent_trials=4 # Increased concurrency as TUAB trials might be faster
    )
    
    best_config = analysis.get_best_config(metric="balanced_accuracy", mode="max")
    
    print("\n" + "="*30)
    print(f"OPTIMAL CSPNET CONFIG (TUAB):\n{json.dumps(best_config, indent=4)}")
    print("="*30)

    with open(CONFIG_SAVE_PATH, "w") as f:
        json.dump(best_config, f)
    
    elapsed = time.time() - start_time
    print(f"\nTuning Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.2f}s")