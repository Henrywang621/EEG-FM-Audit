import torch
import torch.nn as nn
import torch.optim as optim
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
import math
import time
import gc

# ==========================================================
# 1. ENVIRONMENT & MEMORY CONFIGURATION
# ==========================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# 🚨 IMPORT MODEL DEFINITIONS 🚨
from MSCFormer import MSCFormer, Parameters

# ==========================================================
# 2. PATCH FUNCTIONS
# ==========================================================
def patch_mscformer_pe(model, max_len=5000):
    """Ensures Positional Encoding fits the TUAB sequence length."""
    patched = False
    for name, module in model.named_modules():
        if hasattr(module, 'encoding'):
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
                patched = True
    return patched

def patch_mscformer_classifier(model, target_classes=2):
    """Updates the final layer for TUAB Binary Classification."""
    patched = False
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features in [1, 2, 6]:
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1]
            parent = model if parent_name == '' else model.get_submodule(parent_name)
            setattr(parent, child_name, nn.Linear(module.in_features, target_classes))
            patched = True
    return patched

# ==========================================================
# 3. DATA LOADER
# ==========================================================
class TUABTuningLoader(Dataset):
    def __init__(self, root, files, orig_freq=2000, target_freq=1000):
        self.root = root
        self.files = files

        self.resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=target_freq)

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        
        # X shape: [Channels, Time]
        X = torch.FloatTensor(sample["X"])
        X_resampled = self.resampler(X)
        y = int(sample["y"])
        return X_resampled, torch.tensor(y, dtype=torch.long)

# ==========================================================
# 4. TRAINING TASK
# ==========================================================
def train_tune_tuab_mscformer(config): 

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset Setup
    train_path = os.path.join(config["root_path"], "train")
    test_path = os.path.join(config["root_path"], "test")
    
    train_ds = TUABTuningLoader(train_path, sorted(os.listdir(train_path)))
    val_ds = TUABTuningLoader(test_path, sorted(os.listdir(test_path)))
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    
    # Model Initialization
    params = Parameters(dropout_rate=config["dropout_rate"])
    params.f1, params.emb_size = config["f1"], 3 * config["f1"]
    params.depth, params.pooling_size = config["depth"], config["pooling_size"]
    params.num_classes, params.in_channels = 2, 23
    
    model = MSCFormer(params)
    patch_mscformer_pe(model, max_len=5000)
    patch_mscformer_classifier(model, target_classes=2)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    
    try:
        for epoch in range(config["epochs"]):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, logits = model(X.unsqueeze(1))
                    loss = criterion(logits, y)
                
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, logits = model(X.unsqueeze(1))
                    preds.extend(torch.argmax(logits, 1).cpu().numpy())
                    targets.extend(y.cpu().numpy())
            
            bacc = balanced_accuracy_score(targets, preds)
            tune.report({"balanced_accuracy": bacc}) 
            
    except Exception as e:
        print(f"[Error in Trial] {e}")
    finally:
        # Strict memory cleanup for Ray Tune
        del model, optimizer, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

# ==========================================================
# 5. MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    start_time = time.time()
    
    TUAB_ROOT = "/mnt/faster0/xw2336/TUAB/TUAB/processed" 
    LOCAL_TEMP_STORAGE = "/mnt/fast0/xw2336/ray_results_temp"
    FINAL_STORAGE_PORTAL = "/mnt/fast0/xw2336/TUAB/ray_results_MSCFormer"
    CONFIG_SAVE_PATH = "best_config_mscformer_tuab.json"

    os.makedirs(LOCAL_TEMP_STORAGE, exist_ok=True)
    os.makedirs(FINAL_STORAGE_PORTAL, exist_ok=True)

    # Initialize Ray for 1 GPU
    ray.init(ignore_reinit_error=True, num_gpus=1)

    search_space = {
        "root_path": TUAB_ROOT,
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "epochs": tune.choice([50, 80]), 
        "f1": tune.choice([8, 16, 32]), 
        "depth": tune.choice([1, 3, 5]), 
        "pooling_size": tune.choice([45, 52, 60]), 
        "dropout_rate": tune.choice([0.25, 0.5]),
    }

    asha_scheduler = ASHAScheduler(
        metric="balanced_accuracy", 
        mode="max", 
        max_t=80, 
        grace_period=15, 
        reduction_factor=3
    )

    print(f"Starting MSCFormer Tuning on A5000 (24GB Limit enforced).")
    
    analysis = tune.run(
        train_tune_tuab_mscformer, 
        storage_path=LOCAL_TEMP_STORAGE,
        config=search_space,
        scheduler=asha_scheduler,
        num_samples=50, 
        resources_per_trial={"cpu": 4, "gpu": 1}, 
        max_concurrent_trials=1, 
        name="MSCFormer_TUAB_A5000_Sequential"
    )
    
    best_config = analysis.get_best_config(metric="balanced_accuracy", mode="max")
    
    # Save optimized config
    with open(CONFIG_SAVE_PATH, "w") as f:
        json.dump(best_config, f, indent=4)
    
    # Sync results
    print(f"\nMoving results to {FINAL_STORAGE_PORTAL}...")
    os.system(f"cp -r {LOCAL_TEMP_STORAGE}/* {FINAL_STORAGE_PORTAL}/")
    
    elapsed = time.time() - start_time
    print(f"Success. Total Tuning Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m")