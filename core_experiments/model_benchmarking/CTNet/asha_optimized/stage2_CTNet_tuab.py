import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np

import random

import os

import pickle

import pandas as pd

import json

import gc

import time

from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score

from sklearn.model_selection import train_test_split



# 🚨 IMPORT CTNET 🚨

from CTNet import EEGTransformer



# ==========================================================

# DATA LOADING (TUAB Specific)

# ==========================================================

class TUABLoaderResampled(Dataset):

    def __init__(self, root, files, target_len=1000):

        self.root = root

        self.files = files

        self.target_len = target_len



    def __len__(self):

        return len(self.files)



    def __getitem__(self, index):

        try:

            with open(os.path.join(self.root, self.files[index]), "rb") as f:

                sample = pickle.load(f)

            # TUAB format: X shape (Channels, Time), y is label

            X = torch.tensor(sample["X"], dtype=torch.float32)

            y = int(sample["y"])

           

            # Resampling to target_len

            X = X.unsqueeze(0)

            X = torch.nn.functional.interpolate(X, size=self.target_len, mode='linear', align_corners=False)

            X = X.squeeze(0)

            return X, torch.tensor(y, dtype=torch.long)

        except Exception:

            return torch.zeros((23, self.target_len)), torch.tensor(0)



# ==========================================================

# EVALUATION LOGIC

# ==========================================================

def run_final_evaluation(best_config, root_path, seeds):

    results = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

   

    train_dir = os.path.join(root_path, "train")

    test_dir = os.path.join(root_path, "test")

   

    # Get file lists

    all_train_files = sorted(os.listdir(train_dir))

    test_files = sorted(os.listdir(test_dir))



    for seed in seeds:

        print(f"\n>>> Evaluating Seed: {seed}")

        # Reproducibility settings

        torch.manual_seed(seed)

        np.random.seed(seed)

        random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = True

            torch.backends.cudnn.benchmark = False

       

        # Split train into train/val (80/20) for model selection

        train_idx, val_idx = train_test_split(

            np.arange(len(all_train_files)), test_size=0.2, random_state=seed

        )

       

        full_train_ds = TUABLoaderResampled(train_dir, all_train_files)

        train_ds = Subset(full_train_ds, train_idx)

        val_ds = Subset(full_train_ds, val_idx)

        test_ds = TUABLoaderResampled(test_dir, test_files)

       

        train_loader = DataLoader(train_ds, batch_size=best_config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

        val_loader = DataLoader(val_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

        test_loader = DataLoader(test_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

       

        # Initialize CTNet (n_classes=2 for TUAB)

        model = EEGTransformer(

            depth=best_config["depth"],

            eeg1_kernel_size=best_config["eeg1_kernel_size"],

            eeg1_dropout_rate=best_config["eeg1_dropout_rate"],

            number_channel=23,  

            n_classes=2        

        )



        if torch.cuda.device_count() > 1:

            model = nn.DataParallel(model)

        model = model.to(device)

       

        optimizer = optim.Adam(model.parameters(), lr=best_config["lr"])

        criterion = nn.CrossEntropyLoss()

        scaler = torch.amp.GradScaler('cuda')

       

        best_val_bacc = 0

        save_name = f"CTNet_TUAB_seed{seed}_best.pth"



        # Training Loop

        for epoch in range(best_config["epochs"]):

            model.train()

            for X, y in train_loader:

                X, y = X.to(device).unsqueeze(1), y.to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):

                    logits = model(X)

                    loss = criterion(logits, y)

                scaler.scale(loss).backward()

                scaler.step(optimizer)

                scaler.update()

           

            # Validation

            model.eval()

            v_preds, v_targets = [], []

            with torch.no_grad():

                for X, y in val_loader:

                    X = X.unsqueeze(1).to(device)

                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):

                        logits = model(X)

                    v_preds.extend(torch.argmax(logits, 1).cpu().numpy())

                    v_targets.extend(y.numpy())

           

            val_bacc = balanced_accuracy_score(v_targets, v_preds)

            if val_bacc > best_val_bacc:

                best_val_bacc = val_bacc

                state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

                torch.save(state_to_save, save_name)



        # Final Test Evaluation

        test_model = EEGTransformer(

            depth=best_config["depth"],

            eeg1_kernel_size=best_config["eeg1_kernel_size"],

            eeg1_dropout_rate=best_config["eeg1_dropout_rate"],

            number_channel=23,  

            n_classes=2        

        ).to(device)

       

        test_model.load_state_dict(torch.load(save_name))

        test_model.eval()

       

        t_preds, t_targets = [], []

        with torch.no_grad():

            for X, y in test_loader:

                X = X.unsqueeze(1).to(device)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):

                    logits = test_model(X)

                t_preds.extend(torch.argmax(logits, 1).cpu().numpy())

                t_targets.extend(y.numpy())

       

        seed_bacc = balanced_accuracy_score(t_targets, t_preds)

        seed_f1 = f1_score(t_targets, t_preds, average="macro")

        seed_kappa = cohen_kappa_score(t_targets, t_preds)

       

        print(f"Seed {seed} Test Results -> BAcc: {seed_bacc:.4f}, F1: {seed_f1:.4f}, Kappa: {seed_kappa:.4f}")

       

        results.append({

            "seed": seed,

            "balanced_accuracy": seed_bacc,

            "f1_macro": seed_f1,

            "kappa": seed_kappa

        })

       

        del model, test_model

        gc.collect()

        torch.cuda.empty_cache()

   

    return pd.DataFrame(results)



# ==========================================================

# MAIN EXECUTION

# ==========================================================

if __name__ == "__main__":

    start_time = time.time()

   

    # Load the config from the path where Stage 1 actually saved it

    CONFIG_PATH = "/homes/xw2336/data_portal/ASHA/TUAB/best_config_ctnet_tuab.json"

   

    if not os.path.exists(CONFIG_PATH):

        # Fallback to local if path differs

        CONFIG_PATH = "best_config_CTNet1.json"



    with open(CONFIG_PATH, "r") as f:

        best_config = json.load(f)

   

    TUAB_ROOT = best_config.get("root_path", "/homes/xw2336/data_portal/TUAB/processed")

    SEEDS = [42, 3407, 6, 16, 66]

   

    final_df = run_final_evaluation(best_config, TUAB_ROOT, SEEDS)

   

    # Calculate Average and STD

    stats_df = final_df.drop(columns=['seed']).agg(['mean', 'std']).T

    print("\n" + "="*60)

    print("FINAL TUAB CTNET BENCHMARK (5 SEEDS)")

    print("="*60)

    print(final_df.to_string(index=False))

    print("-"*60)

    print("Summary (Mean ± STD):")

    for metric in ['balanced_accuracy', 'f1_macro', 'kappa']:

        m = stats_df.loc[metric, 'mean']

        s = stats_df.loc[metric, 'std']

        print(f"{metric.replace('_', ' ').title()}: {m:.4f} ± {s:.4f}")

    print("="*60)

   

    final_df.to_csv("TUAB_CTNet_Final.csv", index=False)

    print(f"\nTotal Time: {(time.time() - start_time) / 3600:.2f} hours")