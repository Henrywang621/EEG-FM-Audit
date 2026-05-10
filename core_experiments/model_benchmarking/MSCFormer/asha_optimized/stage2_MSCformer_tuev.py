#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import random
import os
import pickle
import pandas as pd
import json
import gc
import time
import math
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score

# 🚨 IMPORT MSCFormer & Parameters 🚨
# Ensure MSCFormer.py is in the same directory or python path
from MSCFormer import MSCFormer, Parameters

# ==========================================================
# PATCHING LOGIC (From Stage 1)
# ==========================================================
def patch_mscformer_pe(model, max_len=5000):
    for name, module in model.named_modules():
        if 'encoding' in module._buffers or 'encoding' in module._parameters or hasattr(module, 'encoding'):
            old_enc = getattr(module, 'encoding')
            if isinstance(old_enc, torch.Tensor) and old_enc.dim() == 3:
                d_model = old_enc.shape[-1]
                pe = torch.zeros(1, max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[0, :, 0::2] = torch.sin(position * div_term)
                if d_model % 2 != 0:
                    pe[0, :, 1::2] = torch.cos(position * div_term)[:, :-1]
                else:
                    pe[0, :, 1::2] = torch.cos(position * div_term)
                
                if 'encoding' in module._parameters:
                    module.register_parameter('encoding', nn.Parameter(pe))
                else:
                    module.register_buffer('encoding', pe)

def patch_mscformer_classifier(model, target_classes=6):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.out_features in [1, 2]:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[-1]
                parent = model if parent_name == '' else model.get_submodule(parent_name)
                setattr(parent, child_name, nn.Linear(module.in_features, target_classes))

# ==========================================================
# DATA LOADING
# ==========================================================
class TUEVLoader(Dataset):
    def __init__(self, root, files, orig_freq=2000, target_freq=1000):
        self.root = root
        self.files = files
        self.resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=target_freq)

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        X = torch.tensor(sample["signal"], dtype=torch.float32)
        y = int(sample["label"][0] - 1)
        return self.resampler(X), torch.tensor(y, dtype=torch.long)

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
        print(f"\n" + "="*40)
        print(f">>> Evaluating Seed: {seed}")
        print("="*40)
        
        # 1. Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 2. Setup Data
        train_ds = TUEVLoader(train_dir, sorted(os.listdir(train_dir)))
        val_ds = TUEVLoader(val_dir, sorted(os.listdir(val_dir)))
        test_ds = TUEVLoader(test_dir, sorted(os.listdir(test_dir)))
        
        train_loader = DataLoader(train_ds, batch_size=best_config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        
        # 3. Initialize & Patch Model
        params = Parameters(dropout_rate=best_config["dropout_rate"])
        params.f1, params.depth, params.pooling_size = best_config["f1"], best_config["depth"], best_config["pooling_size"]
        params.emb_size = 3 * params.f1
        params.num_classes, params.in_channels = 6, 23
        
        model = MSCFormer(params)
        patch_mscformer_pe(model, max_len=5000)
        patch_mscformer_classifier(model, target_classes=6)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])
        criterion = nn.CrossEntropyLoss()
        
        best_val_bacc = 0
        save_name = f"MSCFormer_seed{seed}_best.pth"
        start_train = time.time()

        # 4. Training Loop with Val Monitoring
        for epoch in range(best_config["epochs"]):
            model.train()
            train_loss = 0
            for X, y in train_loader:
                X, y = X.to(device).unsqueeze(1), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                _, logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(device).unsqueeze(1)
                    _, logits = model(X)
                    v_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                    v_targets.extend(y.numpy())
            
            val_bacc = balanced_accuracy_score(v_targets, v_preds)
            if val_bacc > best_val_bacc:
                best_val_bacc = val_bacc
                # Save full state dict for complete reproducibility
                torch.save(model.state_dict(), save_name)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{best_config['epochs']} | Val BAcc: {val_bacc:.4f} | Best: {best_val_bacc:.4f}")

        train_time = time.time() - start_train
        
        # 5. Test Evaluation on Best Weights
        print(f"Loading best model from {save_name} for testing...")
        model.load_state_dict(torch.load(save_name))
        model.eval()
        t_preds, t_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device).unsqueeze(1)
                _, logits = model(X)
                t_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                t_targets.extend(y.numpy())
        
        acc = balanced_accuracy_score(t_targets, t_preds)
        f1 = f1_score(t_targets, t_preds, average="macro")
        kappa = cohen_kappa_score(t_targets, t_preds)
        
        print(f"Seed {seed} Test Results -> BAcc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}")
        
        results.append({
            "seed": seed,
            "balanced_accuracy": acc,
            "f1_macro": f1,
            "kappa": kappa,
            "train_time_sec": train_time
        })
        
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    
    return pd.DataFrame(results)

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    overall_start = time.time()
    
    CONFIG_PATH = "best_config_mscformer_tuev.json"
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Could not find {CONFIG_PATH}. Run Stage 1 tuning first.")
    
    with open(CONFIG_PATH, "r") as f:
        best_config = json.load(f)
    
    TUEV_ROOT = best_config["root_path"]
    SEEDS = [42, 3407, 6, 16, 66]
    
    print(f">>> Optimal Config Loaded:\n{json.dumps(best_config, indent=4)}")
    
    # Run 5-seed Evaluation
    final_df = run_final_evaluation(best_config, TUEV_ROOT, SEEDS)
    
    # Calculate Averages and Std Dev
    stats = final_df.agg(['mean', 'std']).reset_index()
    stats.iloc[0, 0] = "AVERAGE"
    stats.iloc[1, 0] = "STD_DEV"
    
    final_df = pd.concat([final_df, stats], ignore_index=True)
    
    print("\n" + "="*50)
    print("FINAL MSCFORMER BENCHMARK RESULTS (5 SEEDS)")
    print("="*50)
    print(final_df)
    
    output_csv = "TUEV_MSCFormer_Final_Benchmark.csv"
    final_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    total_elapsed = time.time() - overall_start
    print(f"Total Execution Time: {int(total_elapsed // 3600)}h {int((total_elapsed % 3600) // 60)}m {total_elapsed % 60:.2f}s")