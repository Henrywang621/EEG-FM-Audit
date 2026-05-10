import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, cohen_kappa_score
import json
import time

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# ==========================================================
# HYPERPARAMETER CONFIGURATION
# ==========================================================

MODELS_TO_SEARCH = ["EEGNet"]
NUM_SAMPLES = 50
GLOBAL_SEED = 42

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Searching across models: {MODELS_TO_SEARCH}")


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_reproducibility_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


# ==========================================================
# EEGNET MODEL (Dynamic flatten sizing)
# ==========================================================

class TunableEEGNet(nn.Module):
    def __init__(self, n_classes, in_chans=3, input_len=500, 
                 F1=8, D=2, dropout=0.25, temp_kernel=51, sep_kernel=15):
        super(TunableEEGNet, self).__init__()
        
        self.F1 = F1
        self.D = D
        self.F2 = D * F1
        self.in_chans = in_chans
        self.input_len = input_len
        
        # Use Search Space Kernel Sizing
        temporal_kernel = temp_kernel
        separable_kernel = sep_kernel
        
        # Ensure kernels are odd numbers for 'same' padding
        if temporal_kernel % 2 == 0: temporal_kernel -= 1
        if separable_kernel % 2 == 0: separable_kernel -= 1
        
        self.kernel_size_1 = (1, temporal_kernel) 
        self.kernel_size_2 = (in_chans, 1)
        self.kernel_size_3 = (1, separable_kernel)
        
        # Layer 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, self.F1, self.kernel_size_1, padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1)
        
        # Layer 2: Depthwise Spatial Convolution
        self.conv2 = nn.Conv2d(self.F1, self.F1 * self.D, self.kernel_size_2, groups=self.F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F1 * self.D)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 3: Separable Convolution
        self.conv3_depth = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, self.kernel_size_3, padding='same', groups=self.F1 * self.D, bias=False)
        self.conv3_point = nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        
        # Dynamically calculate the flatten size
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, self.in_chans, self.input_len)
            dummy_x = self.conv1(dummy_x)
            dummy_x = self.batchnorm1(dummy_x)
            dummy_x = self.conv2(dummy_x)
            dummy_x = self.batchnorm2(dummy_x)
            dummy_x = self.elu(dummy_x)
            dummy_x = self.avg_pool1(dummy_x)
            dummy_x = self.dropout1(dummy_x)
            dummy_x = self.conv3_depth(dummy_x)
            dummy_x = self.conv3_point(dummy_x)
            dummy_x = self.batchnorm3(dummy_x)
            dummy_x = self.elu(dummy_x)
            dummy_x = self.avg_pool2(dummy_x)
            dummy_x = self.dropout2(dummy_x)
            self.flatten_size = dummy_x.view(1, -1).size(1)

        self.classifier = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x):
        # Accommodate both 3D and 4D tensor inputs cleanly
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        
        x = self.conv3_depth(x)
        x = self.conv3_point(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 


# ==========================================================
# DATA PRELOAD
# ==========================================================

def preload_raw_data(subject_ids,
                     base_path="/homes/xw2336/data/BCICIV_2b/"):

    Subj_map = {
        'S01':1,'S02':2,'S03':3,'S04':4,'S05':5,
        'S06':6,'S07':7,'S08':8,'S09':9
    }

    raw_data_dict = {}
    print("Preloading BCI data...")

    for subj in subject_ids:
        if subj not in Subj_map:
            continue

        subj_int = Subj_map[subj]

        try:
            x1 = np.load(f"{base_path}Subj{subj_int}_1_X.npy")
            x2 = np.load(f"{base_path}Subj{subj_int}_2_X.npy")
            y1 = np.load(f"{base_path}Subj{subj_int}_1_y.npy")
            y2 = np.load(f"{base_path}Subj{subj_int}_2_y.npy")

            x_raw = np.concatenate([x1,x2],axis=0)
            y_raw = np.concatenate([y1,y2],axis=0)

            raw_data_dict[subj] = (x_raw,y_raw)

        except Exception as e:
            print(f"Could not load {subj}: {e}")

    return raw_data_dict


# ==========================================================
# DATASET
# ==========================================================

class BCICIV2b_Dataset(Dataset):
    def __init__(self, raw_data_dict, subj_list):
        x_list = []
        y_list = []

        for subj in subj_list:
            if subj in raw_data_dict:
                x_list.append(raw_data_dict[subj][0])
                y_list.append(raw_data_dict[subj][1])

        self.X = np.concatenate(x_list, axis=0)
        self.y = np.concatenate(y_list, axis=0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor, y_tensor


# ==========================================================
# TRAIN / TUNE FUNCTION
# ==========================================================

def train_eval_model(config, global_data, train_subjs, val_subj):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_reproducibility_seeds(GLOBAL_SEED)

    model_name = config["model_class"]

    # 1. Instantiate Datasets FIRST
    train_ds = BCICIV2b_Dataset(global_data, train_subjs)
    val_ds = BCICIV2b_Dataset(global_data, [val_subj])
    
    # 2. Dynamically extract the true time dimension
    actual_input_len = train_ds.X.shape[-1]

    # 3. Pass it to the model
    if model_name == "EEGNet":
        model = TunableEEGNet(
            n_classes=2, 
            in_chans=3, 
            input_len=actual_input_len,
            F1=config["F1"], 
            D=config["D"], 
            dropout=config["dropout"],
            temp_kernel=config["temp_kernel"],
            sep_kernel=config["sep_kernel"]
        ).to(device)

    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(config["batch_size"]), shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            loss.backward()
            optimizer.step()

        # Validation per epoch
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                
                probs = torch.softmax(logits, dim=1)
                preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                targets.extend(y_batch.cpu().numpy())

        # Report to Ray Tune
        bacc = balanced_accuracy_score(targets, preds)
        tune.report({"balanced_accuracy": bacc})


# ==========================================================
# EVALUATE BEST MODEL (Inner CV -> Test)
# ==========================================================

def eval_best_model_on_test(config, global_data, full_train_subjs, test_subj):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_reproducibility_seeds(GLOBAL_SEED)

    model_name = config["model_class"]

    # 1. Instantiate Datasets FIRST
    train_ds = BCICIV2b_Dataset(global_data, full_train_subjs)
    test_ds = BCICIV2b_Dataset(global_data, [test_subj])
    
    # 2. Dynamically extract the true time dimension
    actual_input_len = train_ds.X.shape[-1]

    # 3. Pass it to the model
    model = TunableEEGNet(
        n_classes=2, 
        in_chans=3, 
        input_len=actual_input_len,
        F1=config["F1"], 
        D=config["D"], 
        dropout=config["dropout"],
        temp_kernel=config["temp_kernel"],
        sep_kernel=config["sep_kernel"]
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=int(config["batch_size"]), shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    # Full Train
    for epoch in range(config["epochs"]):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            loss.backward()
            optimizer.step()

    # Test
    model.eval()
    preds, targets, probs_class_1 = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            
            probs = torch.softmax(logits, dim=1)
            preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            probs_class_1.extend(probs[:, 1].cpu().numpy())
            targets.extend(y_batch.cpu().numpy())

    acc = np.mean(np.array(preds) == np.array(targets))
    bacc = balanced_accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    try:
        auc = roc_auc_score(targets, probs_class_1)
    except ValueError:
        auc = np.nan
    kappa = cohen_kappa_score(targets, preds)

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1": f1,
        "auc": auc,
        "kappa": kappa
    }


# ==========================================================
# MAIN EXECUTION: NESTED LOOCV
# ==========================================================

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    set_reproducibility_seeds(GLOBAL_SEED)

    # 1. Subject setup
    all_subjects = [f'S{i:02d}' for i in range(1, 10)]

    # 2. Preload Data
    global_data = preload_raw_data(all_subjects)

    # EXPANDED EEGNET SEARCH SPACE 
    search_space = {
        "model_class": tune.choice(MODELS_TO_SEARCH),
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

    results_list=[]

    print("Starting Nested LOOCV...")
    for test_subj in all_subjects:
        if test_subj not in global_data:
            continue

        print(f"\n{'='*50}\nEvaluating Test Subject: {test_subj}\n{'='*50}")

        remaining_subjs = [s for s in all_subjects if s != test_subj and s in global_data]
        val_subj = remaining_subjs[-1]  # Simple validation holdout for inner loop
        inner_train_subjs = [s for s in remaining_subjs if s != val_subj]

        # Inner Loop: Ray Tune ASHA Hyperparameter Optimization
        asha_scheduler = ASHAScheduler(
            metric="balanced_accuracy",
            mode="max",
            max_t=100,
            grace_period=20,
            reduction_factor=2
        )

        analysis = tune.run(
            tune.with_parameters(
                train_eval_model,
                global_data=global_data,
                train_subjs=inner_train_subjs,
                val_subj=val_subj
            ),
            config=search_space,
            scheduler=asha_scheduler,
            num_samples=NUM_SAMPLES,
            resources_per_trial={"cpu": 2, "gpu": 0.5},
            verbose=1
        )

        best_config = analysis.get_best_config(metric="balanced_accuracy", mode="max")
        print(f"Optimal Config for {test_subj}: {best_config}")

        # Outer Loop: Test Set Evaluation using the Best Config
        metrics = eval_best_model_on_test(
            best_config, 
            global_data, 
            full_train_subjs=remaining_subjs, 
            test_subj=test_subj
        )

        row = {"Subject": test_subj}
        row.update(metrics)
        row.update(best_config)
        results_list.append(row)

    ray.shutdown()

    # ==========================================================
    # SAVE RESULTS
    # ==========================================================
    import pandas as pd
    
    df = pd.DataFrame(results_list)
    
    if not df.empty:
        # Calculate Average Row
        avg_row = {"Subject": "AVERAGE"}
        for col in df.columns:
            if col != "Subject" and pd.api.types.is_numeric_dtype(df[col]):
                avg_row[col] = df[col].mean()
            elif col != "Subject":
                avg_row[col] = "N/A"
        
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        
        print("\n" + "="*50)
        print("FINAL NESTED LOO RESULTS:")
        print(df[["Subject", "accuracy", "balanced_accuracy", "f1", "auc", "kappa"]].to_string())
        print("="*50 + "\n")
        
        df.to_csv("Nested_LOO_results_EEGNet.csv", index=False)
        print("Saved detailed configuration and results to Nested_LOO_results_EEGNet.csv")