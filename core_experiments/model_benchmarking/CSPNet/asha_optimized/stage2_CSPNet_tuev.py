import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import pickle
import pandas as pd
import json
import gc
import time
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score

# 🚨 IMPORT CSPNET 🚨
# Ensure CSPNet.py is in the same directory as this script
from CSPNet import CSPNet

# ==========================================================
# DATA LOADING (Matches TUEV structure)
# ==========================================================
class TUEVLoader(Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        X, y = sample["signal"], int(sample["label"][0] - 1)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ==========================================================
# EVALUATION LOGIC
# ==========================================================
def run_final_evaluation(best_config, root_path, seeds, save_dir="cspnet_weights_tuev"):
    # Create directory for weights if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f">>> Created directory: {save_dir}")

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dir = os.path.join(root_path, "processed_train")
    val_dir = os.path.join(root_path, "processed_eval")
    test_dir = os.path.join(root_path, "processed_test")
    
    for seed in seeds:
        print(f"\n>>> Evaluating Seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        train_ds = TUEVLoader(train_dir, sorted(os.listdir(train_dir)))
        val_ds = TUEVLoader(val_dir, sorted(os.listdir(val_dir)))
        test_ds = TUEVLoader(test_dir, sorted(os.listdir(test_dir)))
        
        train_loader = DataLoader(train_ds, batch_size=best_config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        
        # Initialize CSPNet with optimized parameters
        model = CSPNet(
            chunk_size=2000,                 
            num_electrodes=23,               
            num_classes=6,                  
            num_filters_t=best_config["num_filters_t"],
            filter_size_t=best_config["filter_size_t"],
            num_filters_s=best_config["num_filters_s"],
            pool_size_1=best_config["pool_size_1"],
            pool_stride_1=best_config["pool_stride_1"],
            dropout=best_config["dropout"]
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])
        
        criterion = nn.NLLLoss()
        scaler = torch.amp.GradScaler('cuda')
        
        best_val_bacc = 0
        save_path = os.path.join(save_dir, f"CSPNet_seed{seed}_best.pth") 

        for epoch in range(best_config["epochs"]):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                X = X.unsqueeze(1) # [B, 1, Chans, Time]
                
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device=="cuda"):
                    logits = model(X).squeeze(-1).squeeze(-1) #
                    loss = criterion(logits, y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            # Validation for model saving
            model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.unsqueeze(1).to(device)
                    logits = model(X).squeeze(-1).squeeze(-1)
                    v_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                    v_targets.extend(y.numpy())
            
            val_bacc = balanced_accuracy_score(v_targets, v_preds)
            if val_bacc > best_val_bacc:
                best_val_bacc = val_bacc
                torch.save(model.state_dict(), save_path)
        
        model.load_state_dict(torch.load(save_path))
        model.eval()
        t_preds, t_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.unsqueeze(1).to(device)
                logits = model(X).squeeze(-1).squeeze(-1)
                t_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                t_targets.extend(y.numpy())
        
        results.append({
            "seed": seed,
            "balanced_accuracy": balanced_accuracy_score(t_targets, t_preds),
            "f1_macro": f1_score(t_targets, t_preds, average="macro"),
            "kappa": cohen_kappa_score(t_targets, t_preds)
        })
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    return pd.DataFrame(results)

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. Load the optimal configuration from CSPNet Stage 1 JSON
    CONFIG_PATH = "best_config_cspnet_tuev.json"
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Could not find {CONFIG_PATH}. Run stage1_CSPNet.py first.")
    
    with open(CONFIG_PATH, "r") as f:
        best_config = json.load(f)
    
    TUEV_ROOT = best_config["root_path"]
    SEEDS = [42, 3407, 6, 16, 66] #
    WEIGHTS_FOLDER = "cspnet_weights_tuev"
    
    print(f">>> Best CSPNet Config Loaded from {CONFIG_PATH}")
    print(json.dumps(best_config, indent=4))
    
    # 2. Run Evaluation
    final_df = run_final_evaluation(best_config, TUEV_ROOT, SEEDS, save_dir=WEIGHTS_FOLDER)
    
    # 3. Process and Save Results
    avg_row = final_df.mean(numeric_only=True).to_dict()
    avg_row["seed"] = "AVERAGE"
    final_df = pd.concat([final_df, pd.DataFrame([avg_row])], ignore_index=True)
    
    print("\nFINAL CSPNET BENCHMARK RESULTS")
    print(final_df)
    final_df.to_csv("TUEV_CSPNet_Final_Benchmark.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\nTotal Evaluation Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.2f}s")