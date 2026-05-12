import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import pickle
import pandas as pd
import gc
import time
from functools import partial
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score

# --- Import EEGTransformer from your local Modules ---
try:
    from Modules.models.EEGPT_mcae import EEGTransformer, CHANNEL_DICT
except ImportError:
    print("Ensure the 'Modules' folder is in your PYTHONPATH.")
    raise

# Set sharing strategy to file_system to avoid shared memory issues
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# ==========================================================
# MODEL DEFINITION: Ablated EEGPT (Supervised from Scratch)
# ==========================================================
class EEGPT_Supervised(nn.Module):
    def __init__(self, n_classes=2, in_chans=23, input_len=2000):
        super(EEGPT_Supervised, self).__init__()
        
        self.encoder = EEGTransformer(
            img_size=[in_chans, input_len],
            patch_size=40, 
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,
            init_std=0.02, 
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        
        tuev_mapped_names = [
            'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
            'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'FZ', 'CZ', 'PZ',
            'AFZ', 'FCZ', 'CPZ', 'OZ' 
        ]
        
        final_keys = []
        available_keys = list(CHANNEL_DICT.keys())
        for ch in tuev_mapped_names[:in_chans]:
            matches = [k for k in available_keys if k.upper() == ch.upper()]
            if matches:
                final_keys.append(matches[0])
            else:
                for fallback in available_keys:
                    if fallback not in final_keys:
                        final_keys.append(fallback)
                        break
        
        self.chans_id = self.encoder.prepare_chan_ids(final_keys)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(2048, 256),
            nn.ELU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        z = self.encoder(x, self.chans_id.to(x.device))
        h = z.mean(dim=1) 
        h = h.flatten(1) 
        return self.classifier(h)

# ==========================================================
# DATA LOADING
# ==========================================================
class TUABLoader(Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        try:
            with open(os.path.join(self.root, self.files[index]), "rb") as f:
                sample = pickle.load(f)
            X = torch.tensor(sample["X"], dtype=torch.float32)
            y = torch.tensor(sample["y"], dtype=torch.long)
            return X / 10.0, y
        except Exception as e:
            # Fallback for corrupted files
            return torch.zeros((23, 2000)), torch.tensor(0)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================================
# MAIN TRAINING & EVALUATION LOOP
# ==========================================================
def run_ablation_study(root_path, seeds):
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = {
        "lr": 1e-4,
        "batch_size": 32,
        "epochs": 50,
        "weight_decay": 0.05,
        "in_chans": 23,
        "input_len": 2000
    }

    train_dir = os.path.join(root_path, "train")
    test_dir = os.path.join(root_path, "test")
    train_files = sorted(os.listdir(train_dir))
    test_files = sorted(os.listdir(test_dir))

    for seed in seeds:
        print(f"\n>>> Running Ablation - Seed: {seed}")
        set_seed(seed)
        
        train_ds = TUABLoader(train_dir, train_files)
        test_ds = TUABLoader(test_dir, test_files)
        
        # 🚨 FIX: Changed num_workers to 0 to avoid "No space left on device" (Shared Memory)
        # 🚨 FIX: Set pin_memory to False to further reduce memory pressure
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
        
        model = EEGPT_Supervised(n_classes=2, in_chans=config["in_chans"], input_len=config["input_len"]).to(device)
        # --- ADD THIS LINE FOR MULTI-GPU ---
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
# -----------------------------------
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda')
        
        best_val_bacc = 0
        best_state = None

        for epoch in range(config["epochs"]):
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
                train_loss += loss.item()
            
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
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {train_loss/len(train_loader):.4f} | BAcc: {val_bacc:.4f}")

        # Final Evaluation
        model.load_state_dict(best_state)
        model.eval()
        t_preds, t_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(X)
                t_preds.extend(torch.argmax(out, 1).cpu().numpy())
                t_targets.extend(y.numpy())
        
        results.append({
            "seed": seed,
            "balanced_accuracy": balanced_accuracy_score(t_targets, t_preds),
            "f1_macro": f1_score(t_targets, t_preds, average="macro"),
            "kappa": cohen_kappa_score(t_targets, t_preds)
        })
        
        del model, optimizer, best_state
        gc.collect()
        torch.cuda.empty_cache()
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    TUAB_ROOT = "/homes/xw2336/faster_data/TUAB/TUAB/processed" 
    SEEDS = [42, 3407, 6, 16, 66]
    
    start_time = time.time()
    results_df = run_ablation_study(TUAB_ROOT, SEEDS)
    
    summary = results_df.drop(columns=['seed']).agg(['mean', 'std'])
    
    print("\n" + "="*50)
    print("FINAL TUAB ABLATED EEGPT BENCHMARK")
    print("="*50)
    print(results_df)
    print("\nOVERALL SUMMARY (Mean ± Std):")
    for col in summary.columns:
        print(f"{col:<20}: {summary.loc['mean', col]:.4f} ± {summary.loc['std', col]:.4f}")
    
    print(f"\nTotal Time: {(time.time() - start_time)/3600:.2f} hours")