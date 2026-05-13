import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
import pandas as pd
from functools import partial
import gc

# --- Local Module Imports ---
# Ensure the 'Modules' folder is in your PYTHONPATH
from Modules.models.EEGPT_mcae import EEGTransformer, CHANNEL_DICT

# ==========================================================
# 1. MODEL DEFINITION (Ablated/Scratch)[cite: 4]
# ==========================================================
class EEGPT_Supervised(nn.Module):
    def __init__(self, n_classes=2, in_chans=3, input_len=500, drop=0.1):
        super(EEGPT_Supervised, self).__init__()
        # Initialized from scratch to ablate pre-training impact[cite: 4]
        self.encoder = EEGTransformer(
            img_size=[in_chans, input_len],
            patch_size=25, 
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=drop,      # ASHA-optimized dropout
            attn_drop_rate=drop,
            drop_path_rate=0.1,
            init_std=0.02, 
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        
        # Select BCI IV 2b Channels: C3, Cz, C4
        bcic_channels = ['C3', 'CZ', 'C4']
        final_keys = []
        available_keys = list(CHANNEL_DICT.keys())
        for ch in bcic_channels:
            matches = [k for k in available_keys if k.upper() == ch.upper()]
            final_keys.append(matches[0] if matches else available_keys[0])
        self.chans_id = self.encoder.prepare_chan_ids(final_keys)

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(2048, 256),
            nn.ELU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        z = self.encoder(x, self.chans_id.to(x.device))
        h = z.mean(dim=1).flatten(1) 
        return self.classifier(h)

# ==========================================================
# 2. DATA UTILITIES[cite: 4]
# ==========================================================
def preload_raw_data(subject_ids, base_path):
    Subj_map = {'S01':1,'S02':2,'S03':3,'S04':4,'S05':5,'S06':6,'S07':7,'S08':8,'S09':9}
    data_dict = {}
    for subj in subject_ids:
        s_idx = Subj_map[subj]
        try:
            x = np.concatenate([np.load(os.path.join(base_path, f"Subj{s_idx}_1_X.npy")), 
                                np.load(os.path.join(base_path, f"Subj{s_idx}_2_X.npy"))], axis=0) / 10.0
            y = np.concatenate([np.load(os.path.join(base_path, f"Subj{s_idx}_1_y.npy")), 
                                np.load(os.path.join(base_path, f"Subj{s_idx}_2_y.npy"))], axis=0)
            data_dict[subj] = (x, y)
        except: continue
    return data_dict

class BCICIV2b_Dataset(Dataset):
    def __init__(self, raw_data_dict, subject_ids, window_size=500):
        X_list, y_list = [], []
        for subj in subject_ids:
            x, y = raw_data_dict[subj]
            X_list.append(torch.tensor(x[:, :, :window_size], dtype=torch.float32))
            y_list.append(torch.tensor(y, dtype=torch.long))
        self.X, self.y = torch.cat(X_list, 0), torch.cat(y_list, 0)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==========================================================
# 3. STAGE 1: HPO TRAINABLE
# ==========================================================
def train_ray(config, raw_data=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = BCICIV2b_Dataset(raw_data, config["train_subs"], config["input_len"])
    val_ds = BCICIV2b_Dataset(raw_data, [config["val_sub"]], config["input_len"])
    
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(config["batch_size"]), shuffle=False)

    model = EEGPT_Supervised(drop=config["drop"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                out = model(X.to(device))
                preds.extend(torch.argmax(out, 1).cpu().numpy())
                targets.extend(y.numpy())
        
        # FIXED: Dictionary format for Ray 2.x+
        tune.report({"balanced_accuracy": balanced_accuracy_score(targets, preds)})

# ==========================================================
# 4. STAGE 2: FINAL EVALUATION[cite: 3]
# ==========================================================
def evaluate_final(config, raw_data, test_subj, train_pool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = BCICIV2b_Dataset(raw_data, train_pool, config["input_len"])
    test_ds = BCICIV2b_Dataset(raw_data, [test_subj], config["input_len"])
    
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=int(config["batch_size"]), shuffle=False)

    model = EEGPT_Supervised(drop=config["drop"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    for _ in range(config["epochs"]):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(); criterion(model(X), y).backward(); optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            out = model(X.to(device))
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            targets.extend(y.numpy())

    return {
        "Balanced_Accuracy": balanced_accuracy_score(targets, preds),
        "F1_Macro": f1_score(targets, preds, average="macro"),
        "Kappa": cohen_kappa_score(targets, preds)
    }

# ==========================================================
# 5. EXECUTION PIPELINE
# ==========================================================
if __name__ == "__main__":
    subjects = ['S01','S02','S03','S04','S05','S06','S07','S08','S09']
    root_path = "/homes/xw2336/data/BCICIV_2b"
    raw_data = preload_raw_data(subjects, root_path)
    
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-3),
        "weight_decay": tune.choice([0.01, 0.05]),
        "batch_size": tune.choice([16, 32]),
        "drop": tune.choice([0.1, 0.3, 0.5]), # Dropout search[cite: 3]
        "epochs": 40, # Faster Stage 1
        "input_len": 500
    }

    results = []
    ray.init(ignore_reinit_error=True)

    for test_subj in subjects:
        print(f"\n>>> LOSO Fold: Testing on {test_subj}")
        train_pool = [s for s in subjects if s != test_subj]
        val_sub = train_pool[-1]
        train_subs = train_pool[:-1]

        asha = ASHAScheduler(metric="balanced_accuracy", mode="max", max_t=40, grace_period=10, reduction_factor=2)
        
        config = search_space.copy()
        config.update({"test_sub": test_subj, "val_sub": val_sub, "train_subs": train_subs})

        analysis = tune.run(
            tune.with_parameters(train_ray, raw_data=raw_data),
            config=config,
            scheduler=asha,
            num_samples=20, # Search intensity[cite: 3]
            resources_per_trial={"cpu": 2, "gpu": 1},
            name=f"Ablation_HPO_{test_subj}",
            verbose=1
        )

        best_config = analysis.get_best_config(metric="balanced_accuracy", mode="max")
        final_metrics = evaluate_final(best_config, raw_data, test_subj, train_pool)
        results.append({"Subject": test_subj, **final_metrics})
        gc.collect()

    df = pd.DataFrame(results)
    print("\nFINAL RESULTS:\n", df)
    print("\nAVERAGE:\n", df.mean(numeric_only=True))
    df.to_csv("Ablated_EEGPT_ASHA_Final.csv", index=False)