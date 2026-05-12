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

# ==========================================================
# MODEL DEFINITION: Ablated EEGPT (Supervised from Scratch)
# ==========================================================
class EEGPT_Supervised(nn.Module):
    def __init__(self, n_classes=6, in_chans=23, input_len=2000):
        super(EEGPT_Supervised, self).__init__()
        
        # Architecture configuration matching EEGPT defaults
        # LLM-rep Ablation: init_std=0.02 ensures we start from a random distribution
        self.encoder = EEGTransformer(
            img_size=[in_chans, input_len],
            patch_size=40, # 2000 / 40 = 50 temporal patches
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
        
        # --- Channel Mapping for TUEV ---
        tuev_mapped_names = [
            'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
            'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'FZ', 'CZ', 'PZ',
            'AFZ', 'FCZ', 'CPZ', 'OZ' 
        ]
        
        # Ensure name compatibility with CHANNEL_DICT
        final_keys = []
        available_keys = list(CHANNEL_DICT.keys())
        for ch in tuev_mapped_names[:in_chans]:
            matches = [k for k in available_keys if k.upper() == ch.upper()]
            if matches:
                final_keys.append(matches[0])
            else:
                # Fallback to unused key to maintain channel count
                for fallback in available_keys:
                    if fallback not in final_keys:
                        final_keys.append(fallback)
                        break
        
        self.chans_id = self.encoder.prepare_chan_ids(final_keys)

        # Classification Head
        # embed_num (4) * embed_dim (512) = 2048
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(2048, 256),
            nn.ELU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # x shape: [Batch, 23, 2000]
        # encoder output z shape: [Batch, 50, 4, 512] (Temporal x Groups x Embed)
        z = self.encoder(x, self.chans_id.to(x.device))
        
        # POOLING FIX: Average across the 50 temporal patches (dim 1)
        # New shape: [Batch, 4, 512]
        h = z.mean(dim=1) 
        
        # Flatten to [Batch, 2048] for the Linear layers
        h = h.flatten(1) 
        
        return self.classifier(h)

# ==========================================================
# DATA LOADING
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
        # Scaling by 10 to match signal range from pre-training logic
        return torch.tensor(X, dtype=torch.float32) / 10.0, torch.tensor(y, dtype=torch.long)

# ==========================================================
# TRAINING & EVALUATION LOGIC
# ==========================================================
def run_ablation_benchmark(root_path, seeds):
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dir = os.path.join(root_path, "processed_train")
    val_dir = os.path.join(root_path, "processed_eval")
    test_dir = os.path.join(root_path, "processed_test")
    
    config = {
        "lr": 1e-4,
        "batch_size": 32,
        "epochs": 50,
        "weight_decay": 0.05
    }

    for seed in seeds:
        print(f"\n>>> Starting Seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        train_ds = TUEVLoader(train_dir, sorted(os.listdir(train_dir)))
        val_ds = TUEVLoader(val_dir, sorted(os.listdir(val_dir)))
        test_ds = TUEVLoader(test_dir, sorted(os.listdir(test_dir)))
        
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
        
        model = EEGPT_Supervised(n_classes=6).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        criterion = nn.CrossEntropyLoss()
        
        best_val_bacc = 0
        best_state = None

        for epoch in range(config["epochs"]):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            
            # Validation at end of epoch
            model.eval()
            v_preds, v_targets = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    out = model(X.to(device))
                    v_preds.extend(torch.argmax(out, 1).cpu().numpy())
                    v_targets.extend(y.numpy())
            
            val_bacc = balanced_accuracy_score(v_targets, v_preds)
            if val_bacc > best_val_bacc:
                best_val_bacc = val_bacc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                print(f"Epoch {epoch}: Val B-Acc = {val_bacc:.4f} (New Best)")

        # Final Evaluation on Test Set using Best Val Weights
        print(f"Seed {seed} complete. Evaluating test set with best validation weights...")
        model.load_state_dict(best_state)
        model.eval()
        t_preds, t_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                out = model(X.to(device))
                t_preds.extend(torch.argmax(out, 1).cpu().numpy())
                t_targets.extend(y.numpy())
        
        results.append({
            "seed": seed,
            "balanced_accuracy": balanced_accuracy_score(t_targets, t_preds),
            "f1_macro": f1_score(t_targets, t_preds, average="macro"),
            "kappa": cohen_kappa_score(t_targets, t_preds)
        })
        
        del model, best_state
        gc.collect()
        torch.cuda.empty_cache()
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Standard TUEV path from your environment
    TUEV_ROOT = "/homes/xw2336/fast0/resampled_2000" 
    SEEDS = [42, 3407, 6, 16, 66]
    
    start_time = time.time()
    results_df = run_ablation_benchmark(TUEV_ROOT, SEEDS)
    
    # Calculate Mean and Std
    summary = results_df.drop(columns=['seed']).agg(['mean', 'std'])
    
    print("\n" + "="*60)
    print("FINAL LLM-REP ABLATION SUMMARY (TUEV)")
    print("="*60)
    print(results_df)
    print("\nAVERAGE OVER 5 SEEDS (Mean ± Std):")
    for col in summary.columns:
        m, s = summary.loc['mean', col], summary.loc['std', col]
        print(f"{col:<20}: {m:.4f} ± {s:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/3600:.2f} hours")