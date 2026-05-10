import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# ==========================================================
# MODEL DEFINITION (Must match Stage 1 exactly)
# ==========================================================
class TunableEEGNet(nn.Module):
    def __init__(self, n_classes, in_chans=23, input_len=2000, F1=8, D=2, 
                 dropout=0.25, temp_kernel=101, sep_kernel=25):
        super(TunableEEGNet, self).__init__()
        self.F2 = F1 * D
        self.conv1 = nn.Conv2d(1, F1, (1, temp_kernel), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, self.F2, (in_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.F2)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        
        self.conv3 = nn.Conv2d(self.F2, self.F2, (1, sep_kernel), padding='same', groups=self.F2, bias=False)
        self.conv4 = nn.Conv2d(self.F2, self.F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_chans, input_len)
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
# DATA LOADING (Updated for TUAB)
# ==========================================================
class TUABLoader(Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        # TUAB X shape is usually (23, 2000), y is 0 or 1
        X = torch.tensor(sample["X"], dtype=torch.float32)
        y = torch.tensor(sample["y"], dtype=torch.long)
        return X, y

# ==========================================================
# REPRODUCIBILITY FUNCTION
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
    test_dir = os.path.join(root_path, "test") # TUAB Test set
    
    # Pre-load file lists for speed
    train_files = sorted(os.listdir(train_dir))
    test_files = sorted(os.listdir(test_dir))

    for seed in seeds:
        print(f"\n>>> Training & Evaluating Seed: {seed}")
        set_seed(seed)
        
        # In TUAB Stage 2, we often split a portion of train for validation 
        # or use the test set as validation if following the Stage 1 logic.
        # Here we use the test set for validation-based saving to match your request.
        train_ds = TUABLoader(train_dir, train_files)
        test_ds = TUABLoader(test_dir, test_files)
        
        train_loader = DataLoader(train_ds, batch_size=best_config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        
        model = TunableEEGNet(
            n_classes=2, # TUAB Binary
            input_len=best_config["input_len"],
            F1=best_config["F1"], 
            D=best_config["D"], 
            dropout=best_config["dropout"],
            temp_kernel=best_config["temp_kernel"],
            sep_kernel=best_config["sep_kernel"]
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda')
        
        best_val_bacc = 0
        save_name = f"EEGNet_TUAB_seed{seed}_best.pth" 

        for epoch in range(best_config["epochs"]):
            model.train()
            train_loss = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(X)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            # Validation (using Test set to find best weights as per user request)
            model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for X, y in test_loader:
                    X = X.to(device)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        out = model(X)
                    v_preds.extend(torch.argmax(out, 1).cpu().numpy())
                    v_targets.extend(y.numpy())
            
            val_bacc = balanced_accuracy_score(v_targets, v_preds)
            if val_bacc > best_val_bacc:
                best_val_bacc = val_bacc
                torch.save(model.state_dict(), save_name)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{best_config['epochs']} | Val BAcc: {val_bacc:.4f} | Best: {best_val_bacc:.4f}")

        # Final Test Evaluation on Best Weights for this seed
        print(f"Loading best weights from {save_name} for final test...")
        model.load_state_dict(torch.load(save_name))
        model.eval()
        t_preds, t_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(X)
                t_preds.extend(torch.argmax(out, 1).cpu().numpy())
                t_targets.extend(y.numpy())
        
        s_bacc = balanced_accuracy_score(t_targets, t_preds)
        s_f1 = f1_score(t_targets, t_preds, average="macro")
        s_kappa = cohen_kappa_score(t_targets, t_preds)
        
        print(f"Seed {seed} Result -> BAcc: {s_bacc:.4f}, F1: {s_f1:.4f}, Kappa: {s_kappa:.4f}")

        results.append({
            "seed": seed,
            "balanced_accuracy": s_bacc,
            "f1_macro": s_f1,
            "kappa": s_kappa
        })
        
        del model, optimizer, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    return pd.DataFrame(results)

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. Load the optimal configuration from Stage 1 TUAB JSON
    CONFIG_PATH = "best_config_eegnet_tuab.json"
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Could not find {CONFIG_PATH}. Please run Stage 1 for TUAB first.")
    
    with open(CONFIG_PATH, "r") as f:
        best_config = json.load(f)
    
    SEEDS = [42, 3407, 6, 16, 66]
    
    print(f">>> Best TUAB Config Loaded")
    print(json.dumps(best_config, indent=4))
    
    # 2. Run Evaluation
    final_df = run_final_evaluation(best_config, SEEDS)
    
    # 3. Process and Save Results
    # Calculate Mean and STD
    metrics = ["balanced_accuracy", "f1_macro", "kappa"]
    avg_stats = final_df[metrics].mean().to_dict()
    std_stats = final_df[metrics].std().to_dict()
    
    avg_row = {**avg_stats, "seed": "MEAN"}
    std_row = {**std_stats, "seed": "STD"}
    
    final_df = pd.concat([final_df, pd.DataFrame([avg_row, std_row])], ignore_index=True)
    
    print("\n" + "="*50)
    print("FINAL TUAB BENCHMARK RESULTS (5 SEEDS)")
    print("="*50)
    print(final_df.to_string(index=False))
    print("="*50)
    
    final_df.to_csv("TUAB_EEGNet_Final_Benchmark.csv", index=False)

    elapsed = time.time() - start_time
    print(f"\nTotal Evaluation Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.2f}s")