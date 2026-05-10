import torch
import torch.nn as nn
import torch.nn.functional as F
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

# 🚨 IMPORT TS_SEFFNet 🚨
from TS_SEFFNet import TS_SEFFNet

# ==========================================================
# DATA LOADING WITH RESAMPLING (Matches Stage 1)
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
        
        X = X.unsqueeze(0) 
        X = F.interpolate(X, size=self.target_len, mode='linear', align_corners=False)
        X = X.squeeze(0) 
        
        y = int(sample["label"][0] - 1)
        return X, torch.tensor(y, dtype=torch.long)

# ==========================================================
# EVALUATION LOGIC
# ==========================================================
def run_final_evaluation(best_config, root_path, seeds):
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dir = os.path.join(root_path, "processed_train")
    val_dir = os.path.join(root_path, "processed_eval")
    test_dir = os.path.join(root_path, "processed_test")
    
    for seed in seeds:
        print(f"\n>>> Evaluating Seed: {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        train_ds = TUEVLoader(train_dir, sorted(os.listdir(train_dir)), target_len=1125)
        val_ds = TUEVLoader(val_dir, sorted(os.listdir(val_dir)), target_len=1125)
        test_ds = TUEVLoader(test_dir, sorted(os.listdir(test_dir)), target_len=1125)
        
        train_loader = DataLoader(train_ds, batch_size=best_config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        
        model = TS_SEFFNet(
            in_chans=23,                  
            n_classes=6,                
            drop_prob=best_config["drop_prob"],
            batch_norm_alpha=best_config["batch_norm_alpha"],
            reduction_ratio=best_config["reduction_ratio"],
            pool_stride=best_config["pool_stride"], 
            conv_stride=best_config["conv_stride"]
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])
        criterion = nn.NLLLoss()
        
        best_val_bacc = 0
        save_name = f"TSSEFFNet_seed{seed}_best.pth" 

        for epoch in range(best_config["epochs"]):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                X = X.unsqueeze(-1) # Expected shape: (Batch, Channels, Time, 1)
                
                optimizer.zero_grad(set_to_none=True)
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            
            # Validation for model saving
            model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.unsqueeze(-1).to(device)
                    logits = model(X)
                    probs = torch.exp(logits) # Log_softmax to Probs
                    v_preds.extend(torch.argmax(probs, 1).cpu().numpy())
                    v_targets.extend(y.numpy())
            
            val_bacc = balanced_accuracy_score(v_targets, v_preds)
            if val_bacc > best_val_bacc:
                best_val_bacc = val_bacc
                torch.save(model.state_dict(), save_name)
        
        # Test Evaluation on Best Weights
        model.load_state_dict(torch.load(save_name))
        model.eval()
        t_preds, t_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.unsqueeze(-1).to(device)
                logits = model(X)
                probs = torch.exp(logits)
                t_preds.extend(torch.argmax(probs, 1).cpu().numpy())
                t_targets.extend(y.numpy())
        
        results.append({
            "seed": seed,
            "balanced_accuracy": balanced_accuracy_score(t_targets, t_preds),
            "f1_macro": f1_score(t_targets, t_preds, average="macro"),
            "kappa": cohen_kappa_score(t_targets, t_preds)
        })
        
        print(f"Seed {seed} Test BACC: {results[-1]['balanced_accuracy']:.4f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    return pd.DataFrame(results)

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. Load the optimal configuration
    CONFIG_PATH = "best_config_tsseffnet_tuev.json"
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Could not find {CONFIG_PATH}. Run stage1_TSSEFFNet.py first.")
    
    with open(CONFIG_PATH, "r") as f:
        best_config = json.load(f)
    
    TUEV_ROOT = best_config["root_path"]
    SEEDS = [42, 3407, 6, 16, 66] # Standard seeds for NeurIPS benchmarks
    
    print(f">>> TS_SEFFNet Final Evaluation Started")
    print(f">>> Root Path: {TUEV_ROOT}")
    
    # 2. Run Evaluation
    final_df = run_final_evaluation(best_config, TUEV_ROOT, SEEDS)
    
    # 3. Process and Save Results
    avg_row = final_df.mean(numeric_only=True).to_dict()
    avg_row["seed"] = "AVERAGE"
    final_df = pd.concat([final_df, pd.DataFrame([avg_row])], ignore_index=True)
    
    print("\nFINAL TS_SEFFNET BENCHMARK RESULTS")
    print(final_df)
    final_df.to_csv("TUEV_TSSEFFNet_Final_Benchmark.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\nTotal Evaluation Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.2f}s")