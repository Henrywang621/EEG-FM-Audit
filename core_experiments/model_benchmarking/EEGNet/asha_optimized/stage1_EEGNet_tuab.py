import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import json
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
import gc
import time

# ==========================================================
# MODEL DEFINITION (Optimized EEGNet)
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
# DATA LOADING (TUAB Specific)
# ==========================================================
class TUABLoader(Dataset):
    def __init__(self, root, files):
        self.root = root
        self.files = files

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        X = torch.tensor(sample["X"], dtype=torch.float32)
        y = torch.tensor(sample["y"], dtype=torch.long)
        return X, y

# ==========================================================
# TUNING FUNCTION
# ==========================================================
def train_tune_tuab(config):
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TUAB Data paths
    root = config["root_path"]
    train_dir = os.path.join(root, "train")
    # TUAB often uses the 'test' set for validation during tuning
    val_dir = os.path.join(root, "test") 
    
    train_ds = TUABLoader(train_dir, os.listdir(train_dir))
    val_ds = TUABLoader(val_dir, os.listdir(val_dir))
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2)
    
    model = TunableEEGNet(
        n_classes=2, # TUAB Binary: Normal vs Abnormal
        input_len=config["input_len"],
        F1=config["F1"],
        D=config["D"],
        dropout=config["dropout"],
        temp_kernel=config["temp_kernel"],
        sep_kernel=config["sep_kernel"]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(config["epochs"]):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(X)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Evaluation
        model.eval()
        preds, targets, probs = [], [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(X)
                    p = F.softmax(logits, dim=1)
                
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                targets.extend(y.cpu().numpy())
                probs.extend(p[:, 1].cpu().numpy())

        bacc = balanced_accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro')
        kappa = cohen_kappa_score(targets, preds)

        # Report to Ray Tune
        tune.report({
            "balanced_accuracy": bacc,
            "f1": f1,
            "kappa": kappa
        })
        
    # Cleanup
    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    start_time = time.time()
    
    TUAB_ROOT = "/homes/xw2336/data_portal/TUAB/processed"
    RESULT_STORAGE = "/homes/xw2336/data_portal/ASHA/TUAB/ray_results_EEGNet"
    CONFIG_SAVE_PATH = "best_config_eegnet_tuab.json"

    ray.init(ignore_reinit_error=True)

    print(f"--- Starting TUAB EEGNet ASHA Search ---")

    search_space = {
        "root_path": TUAB_ROOT,
        "input_len": 2000,
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "epochs": tune.choice([50, 80, 100]),
        "dropout": tune.choice([0.25, 0.5]),
        "F1": tune.choice([8, 16]),
        "D": tune.choice([1, 2]),
        "temp_kernel": tune.choice([51, 101, 125, 151, 201]),
        "sep_kernel": tune.choice([8, 15, 25, 35, 45])
    }

    asha_scheduler = ASHAScheduler(
        metric="balanced_accuracy", 
        mode="max", 
        max_t=100, 
        grace_period=15, 
        reduction_factor=3
    )

    analysis = tune.run(
        train_tune_tuab,
        storage_path=RESULT_STORAGE,
        config=search_space,
        scheduler=asha_scheduler,
        num_samples=50,
        resources_per_trial={"cpu": 2, "gpu": 1},
        verbose=1
    )

    best_config = analysis.get_best_config(metric="balanced_accuracy", mode="max")
    
    print("\n" + "="*30)
    print(f"OPTIMAL TUAB CONFIG:\n{json.dumps(best_config, indent=4)}")
    print("="*30)

    with open(CONFIG_SAVE_PATH, "w") as f:
        json.dump(best_config, f)
    
    elapsed = time.time() - start_time
    print(f"\nTotal Tuning Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.2f}s")