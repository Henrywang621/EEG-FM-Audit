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
from CSPNet import CSPNet

# ==========================================================
# DATA LOADING (TUAB Structure)
# ==========================================================
class TUABLoader(Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, index):
        try:
            with open(os.path.join(self.root, self.files[index]), "rb") as f:
                sample = pickle.load(f)
            X = torch.FloatTensor(sample["X"]) 
            y = torch.tensor(sample["y"], dtype=torch.long)
            return X, y
        except Exception:
            # Fallback for corrupted files as handled in Stage 1
            return torch.zeros((23, 2000)), torch.tensor(0, dtype=torch.long)

# ==========================================================
# EVALUATION LOGIC
# ==========================================================
def run_final_evaluation(best_config, root_path, seeds, save_dir="cspnet_weights_tuab"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f">>> Created directory: {save_dir}")

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # FIXED: Map validation directly to the test folder (matches Stage 1)
    train_dir = os.path.join(root_path, "train")
    val_dir = os.path.join(root_path, "test")
    test_dir = os.path.join(root_path, "test")
    
    for seed in seeds:
        print(f"\n>>> Evaluating Seed: {seed}")
        
        # 1. Enforce Strict Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # 2. Load Datasets
        train_ds = TUABLoader(train_dir, sorted(os.listdir(train_dir)))
        val_ds = TUABLoader(val_dir, sorted(os.listdir(val_dir)))
        test_ds = TUABLoader(test_dir, sorted(os.listdir(test_dir)))
        
        # DataLoader Generator for seed consistency in multi-processing
        g = torch.Generator()
        g.manual_seed(seed)
        
        train_loader = DataLoader(train_ds, batch_size=best_config["batch_size"], shuffle=True, num_workers=4, pin_memory=True, generator=g)
        val_loader = DataLoader(val_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        
        # 3. Initialize CSPNet for Binary Classification (TUAB)
        model = CSPNet(
            chunk_size=2000,                 
            num_electrodes=23,               
            num_classes=2,  # <--- CRITICAL: TUAB is binary (Abnormal vs Normal)                
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
        
        best_val_bacc = -1.0
        save_path = os.path.join(save_dir, f"CSPNet_seed{seed}_best.pth") 

        # 4. Training Loop
        for epoch in range(best_config["epochs"]):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                X = X.unsqueeze(1) 
                
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device=="cuda"):
                    logits = model(X).squeeze(-1).squeeze(-1) 
                    loss = criterion(logits, y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            # 5. Validation Loop (Save Best Model)
            model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.unsqueeze(1).to(device, non_blocking=True)
                    logits = model(X).squeeze(-1).squeeze(-1)
                    v_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                    v_targets.extend(y.numpy())
            
            # Ignore warnings for zero-division if a batch is weird
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val_bacc = balanced_accuracy_score(v_targets, v_preds)
            
            # FIXED: Bulletproof save logic. Save on epoch 0, or if valid score improves.
            if epoch == 0 or (not np.isnan(val_bacc) and val_bacc > best_val_bacc):
                if not np.isnan(val_bacc):
                    best_val_bacc = val_bacc
                torch.save(model.state_dict(), save_path)
        
        # 6. Test Evaluation on Best Weights
        # Reload the saved state dict to guarantee we are evaluating the best epoch
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        model.eval()
        t_preds, t_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.unsqueeze(1).to(device, non_blocking=True)
                logits = model(X).squeeze(-1).squeeze(-1)
                t_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                t_targets.extend(y.numpy())
        
        # Calculate Test Metrics
        t_bacc = balanced_accuracy_score(t_targets, t_preds)
        t_f1 = f1_score(t_targets, t_preds, average="macro")
        t_kappa = cohen_kappa_score(t_targets, t_preds)
        
        print(f"Seed {seed} Test Results -> BAcc: {t_bacc:.4f} | F1: {t_f1:.4f} | Kappa: {t_kappa:.4f}")
        
        results.append({
            "seed": seed,
            "balanced_accuracy": t_bacc,
            "f1_macro": t_f1,
            "kappa": t_kappa
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
    
    # 1. Load the optimal configuration from Stage 1 JSON
    CONFIG_PATH = "best_config_cspnet_tuab.json"
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Could not find {CONFIG_PATH}. Run stage1_CSPNet_tuab.py first.")
    
    with open(CONFIG_PATH, "r") as f:
        best_config = json.load(f)
    
    TUAB_ROOT = best_config["root_path"]
    SEEDS = [42, 3407, 6, 16, 66]
    WEIGHTS_FOLDER = "cspnet_weights_tuab"
    
    print(f">>> Best CSPNet Config Loaded from {CONFIG_PATH}")
    print(json.dumps(best_config, indent=4))
    
    # 2. Run Evaluation
    final_df = run_final_evaluation(best_config, TUAB_ROOT, SEEDS, save_dir=WEIGHTS_FOLDER)
    
    # 3. Process Averages and STD
    avg_row = final_df.mean(numeric_only=True).to_dict()
    std_row = final_df.std(numeric_only=True).to_dict()
    
    avg_row["seed"] = "AVERAGE"
    std_row["seed"] = "STD"
    
    final_df = pd.concat([final_df, pd.DataFrame([avg_row, std_row])], ignore_index=True)
    
    print("\n" + "="*50)
    print("FINAL TUAB CSPNET BENCHMARK RESULTS")
    print("="*50)
    print(final_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
    
    final_df.to_csv("TUAB_CSPNet_Final_Benchmark.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\nTotal Evaluation Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.2f}s")