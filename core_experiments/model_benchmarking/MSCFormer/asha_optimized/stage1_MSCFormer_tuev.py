import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import os
import pickle
import json
import ray
from ray import tune 
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import balanced_accuracy_score
import gc
import time
import math

# 🚨 Ensure MSCFormer.py and its dependencies are in the same folder 🚨
from MSCFormer import MSCFormer, Parameters

# ==========================================================
# ARCHITECTURE PATCHES
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
    return model

def patch_mscformer_classifier(model, target_classes=6):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.out_features in [1, 2]:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[-1]
                parent = model if parent_name == '' else model.get_submodule(parent_name)
                new_layer = nn.Linear(module.in_features, target_classes)
                setattr(parent, child_name, new_layer)
    return model

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
        X = sample["signal"] 
        y = int(sample["label"][0] - 1) 
        del sample
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return self.resampler(X_tensor), torch.tensor(y, dtype=torch.long)

# ==========================================================
# TUNING FUNCTION (CRASH-PROOF)
# ==========================================================
def train_tune_mscformer(config): 
    # Helps prevent memory fragmentation across 16 processes
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        torch.manual_seed(42)
        train_dir = os.path.join(config["root_path"], "processed_train")
        val_dir = os.path.join(config["root_path"], "processed_eval")
        
        # num_workers=1 is safer for high concurrency to avoid OS file handle limits
        train_loader = DataLoader(TUEVLoader(train_dir, sorted(os.listdir(train_dir))), 
                                  batch_size=config["batch_size"], shuffle=True, num_workers=1, pin_memory=True)
        val_loader = DataLoader(TUEVLoader(val_dir, sorted(os.listdir(val_dir))), 
                                batch_size=config["batch_size"], shuffle=False, num_workers=1, pin_memory=True)
        
        params = Parameters(dropout_rate=config["dropout_rate"])
        params.f1, params.depth, params.pooling_size = config["f1"], config["depth"], config["pooling_size"]
        params.emb_size = 3 * config["f1"]
        params.num_classes, params.in_channels = 6, 23
        
        model = MSCFormer(params)
        patch_mscformer_pe(model, max_len=5000)
        patch_mscformer_classifier(model, target_classes=6)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        criterion = nn.CrossEntropyLoss()
        
        # 🚨 USE MIXED PRECISION TO SAVE VRAM 🚨
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(config["epochs"]):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device).unsqueeze(1), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast():
                    _, logits = model(X)
                    loss = criterion(logits, y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            # Evaluation
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device).unsqueeze(1), y.to(device)
                    with torch.cuda.amp.autocast():
                        _, logits = model(X)
                    preds.extend(torch.argmax(logits, 1).cpu().numpy())
                    targets.extend(y.cpu().numpy())
            
            acc = balanced_accuracy_score(targets, preds)
            tune.report({"balanced_accuracy": acc})
            
            # Manual cleanup each epoch
            del X, y, logits
            torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        # If the trial fails (e.g., CUDA OOM), report a 0 score so Ray continues
        print(f"Trial Failed with Error: {e}")
        tune.report({"balanced_accuracy": 0.0, "error": str(e)})

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    start_time = time.time()
    
    TUEV_ROOT = "/homes/xw2336/fast0/resampled_2000"
    RESULT_STORAGE = "/homes/xw2336/fast0/ray_results_MSCFormer1"
    CONFIG_SAVE_PATH = "best_config_mscformer_tuev1.json"

    # Initialize Ray for 2 GPUs
    ray.init(num_gpus=2, ignore_reinit_error=True)

    search_space = {
        "root_path": TUEV_ROOT,
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]), 
        "epochs": tune.choice([50, 80, 100]), 
        "f1": tune.choice([8, 16, 32]), 
        "depth": tune.choice([1, 3, 5, 10, 12]), 
        "pooling_size": tune.choice([45, 52, 60]), 
        "dropout_rate": tune.choice([0.25, 0.5]),
    }

    analysis = tune.run(
        train_tune_mscformer, 
        storage_path=RESULT_STORAGE,
        config=search_space,
        scheduler=ASHAScheduler(metric="balanced_accuracy", mode="max", max_t=100, grace_period=20),
        num_samples=80, 

        resources_per_trial={"cpu": 2, "gpu": 1}, 
        max_concurrent_trials=2, 
        fail_fast=False, # prevents one failure from stopping everything
        name="MSCFormer_TUEV",
        resume="AUTO"
    )

    best_config = analysis.get_best_config(metric="balanced_accuracy", mode="max")
    
    print("\n" + "="*30)
    print(f"BEST CONFIG FOUND:\n{json.dumps(best_config, indent=4)}")
    print("="*30)

    with open(CONFIG_SAVE_PATH, "w") as f:
        json.dump(best_config, f)
    
    elapsed = time.time() - start_time
    print(f"\nTotal Tuning Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m")