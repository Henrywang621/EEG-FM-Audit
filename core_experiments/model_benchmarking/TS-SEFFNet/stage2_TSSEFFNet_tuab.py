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
# DATA LOADING (Adapted for TUAB)
# ==========================================================
class TUABLoader(Dataset):
    def __init__(self, root, files, target_len=1125):
        self.root = root
        self.files = files
        self.target_len = target_len

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        
        # TUAB specific keys from your Stage 1 script
        X = torch.tensor(sample["X"], dtype=torch.float32) 
        
        # Resampling 2000 -> 1125 for TS_SEFFNet compatibility
        X = X.unsqueeze(0) 
        X = F.interpolate(X, size=self.target_len, mode='linear', align_corners=False)
        X = X.squeeze(0) 
        
        y = torch.tensor(sample["y"], dtype=torch.long)
        return X, y

# ==========================================================
# REPRODUCIBILITY SETUP
# ==========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================================
# EVALUATION LOGIC
# ==========================================================
def run_final_evaluation(best_config, seeds):
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    root_path = best_config["root_path"]
    train_dir = os.path.join(root_path, "train")
    test_dir = os.path.join(root_path, "test") # Used for both Val-saving and Test report
    
    train_files = sorted(os.listdir(train_dir))
    test_files = sorted(os.listdir(test_dir))

    for seed in seeds:
        print(f"\n>>> Training & Evaluating Seed: {seed}")
        set_seed(seed)
        
        train_ds = TUABLoader(train_dir, train_files, target_len=1125)
        test_ds = TUABLoader(test_dir, test_files, target_len=1125)
        
        train_loader = DataLoader(train_ds, batch_size=best_config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        
        model = TS_SEFFNet(
            in_chans=23,                  
            n_classes=2, # TUAB Binary                
            drop_prob=best_config["drop_prob"],
            batch_norm_alpha=best_config["batch_norm_alpha"],
            reduction_ratio=best_config["reduction_ratio"],
            pool_stride=best_config["pool_stride"], 
            conv_stride=best_config["conv_stride"]
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])
        criterion = nn.NLLLoss() 
        
        best_val_bacc = 0
        save_name = f"TSSEFFNet_TUAB_seed{seed}_best.pth" 

        for epoch in range(best_config["epochs"]):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                X = X.unsqueeze(-1) # Shape: (B, C, T, 1)
                
                optimizer.zero_grad(set_to_none=True)
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            
            # Validation (using Test set to track best weights per seed)
            model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for X, y in test_loader:
                    X = X.unsqueeze(-1).to(device)
                    logits = model(X)
                    probs = torch.exp(logits) 
                    v_preds.extend(torch.argmax(probs, 1).cpu().numpy())
                    v_targets.extend(y.numpy())
            
            val_bacc = balanced_accuracy_score(v_targets, v_preds)
            if val_bacc > best_val_bacc:
                best_val_bacc = val_bacc
                torch.save(model.state_dict(), save_name)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{best_config['epochs']} | Val BAcc: {val_bacc:.4f} | Best: {best_val_bacc:.4f}")

        # Final Test Evaluation using the absolute best weights saved
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
        
        s_bacc = balanced_accuracy_score(t_targets, t_preds)
        s_f1 = f1_score(t_targets, t_preds, average="macro")
        s_kappa = cohen_kappa_score(t_targets, t_preds)
        
        results.append({
            "seed": seed,
            "balanced_accuracy": s_bacc,
            "f1_macro": s_f1,
            "kappa": s_kappa
        })
        print(f"Seed {seed} Final Results -> BAcc: {s_bacc:.4f}, F1: {s_f1:.4f}, Kappa: {s_kappa:.4f}")
        
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    
    return pd.DataFrame(results)

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. Load the optimal configuration from Stage 1 TUAB
    CONFIG_PATH = "best_config_tsseffnet_tuab.json"
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Could not find {CONFIG_PATH}. Please run Stage 1 first.")
    
    with open(CONFIG_PATH, "r") as f:
        best_config = json.load(f)
    
    SEEDS = [42, 3407, 6, 16, 66]
    
    print(f">>> TS_SEFFNet TUAB Evaluation Started")
    print(json.dumps(best_config, indent=4))
    
    # 2. Run Evaluation
    final_df = run_final_evaluation(best_config, SEEDS)
    
    # 3. Calculate Mean and STD
    metrics = ["balanced_accuracy", "f1_macro", "kappa"]
    avg_stats = final_df[metrics].mean().to_dict()
    std_stats = final_df[metrics].std().to_dict()
    
    avg_row = {**avg_stats, "seed": "MEAN"}
    std_row = {**std_stats, "seed": "STD"}
    
    final_df = pd.concat([final_df, pd.DataFrame([avg_row, std_row])], ignore_index=True)
    
    print("\n" + "="*50)
    print("FINAL TS_SEFFNET TUAB BENCHMARK RESULTS")
    print("="*50)
    print(final_df.to_string(index=False))
    print("="*50)

    final_df.to_csv("TUAB_TSSEFFNet_Final_Benchmark.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\nTotal Evaluation Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.2f}s")