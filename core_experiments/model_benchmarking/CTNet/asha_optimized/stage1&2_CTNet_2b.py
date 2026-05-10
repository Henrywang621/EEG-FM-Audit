import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import random
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score, balanced_accuracy_score
import pandas as pd
import gc

from CTNet import EEGTransformer


# ==========================================================
# USER CONFIGURATION
# ==========================================================

MODELS_TO_SEARCH = ["CTNet"]

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
    random.seed(seed)


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

    def __init__(self, raw_data_dict, subject_ids,
                 window_index=8, window_size=500,
                 step_size=100, target_len=1000):

        self.resampler = torchaudio.transforms.Resample(
            orig_freq=window_size,
            new_freq=target_len
        )

        X_list = []
        y_list = []

        for subj in subject_ids:
            if subj not in raw_data_dict:
                continue

            x_raw, y_raw = raw_data_dict[subj]

            start = (window_index - 1) * step_size
            end = start + window_size

            if end > x_raw.shape[2]:
                continue

            x_seg = x_raw[:,:,start:end]

            x_seg = torch.tensor(x_seg,dtype=torch.float32)
            x_seg = self.resampler(x_seg)

            X_list.append(x_seg)
            y_list.append(torch.tensor(y_raw,dtype=torch.long))

        if not X_list:
            raise ValueError("No valid windows.")

        self.X = torch.cat(X_list,dim=0)
        self.y = torch.cat(y_list,dim=0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]


# ==========================================================
# TRAIN FUNCTION FOR RAY
# ==========================================================

# 🚨 FIX: Function accepts ONLY config to avoid pickling crash
def train_any_model(config):

    # 🚨 FIX: Extract loop variables directly from config
    train_subjs = config["train_subjs"]
    val_subj = config["val_subj"]
    global_data_ref = config["global_data_ref"]

    global_data = ray.get(global_data_ref)
    set_reproducibility_seeds(GLOBAL_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 🚨 FIX: model_name now correctly pulls from config
    model_name = config["model_class"]

    if model_name == "CTNet":
        model = EEGTransformer(
            depth=config["depth"],
            eeg1_kernel_size=config["eeg1_kernel_size"],
            eeg1_dropout_rate=config["eeg1_dropout_rate"]
        ).to(device)

    train_ds = BCICIV2b_Dataset(
        global_data, train_subjs,
    )

    val_ds = BCICIV2b_Dataset(
        global_data, [val_subj],
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        shuffle=True, num_workers=1, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=1, pin_memory=True
    )

    optimizer = optim.Adam(model.parameters(),lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):

        model.train()
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            optimizer.zero_grad()
            X = X.unsqueeze(1)
            logits = model(X)
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds=[]
        all_labels=[]
        all_probs=[]

        with torch.no_grad():
            for X,y in val_loader:
                X,y = X.to(device),y.to(device)
                X = X.unsqueeze(1)
                logits = model(X)
                probs = torch.softmax(logits,dim=1)
                preds = torch.argmax(probs,dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs[:,1].cpu().numpy())

        acc = accuracy_score(all_labels,all_preds)
        bacc = balanced_accuracy_score(all_labels,all_preds)
        f1 = f1_score(all_labels,all_preds,average="macro")
        kappa = cohen_kappa_score(all_labels,all_preds)

        try:
            auc = roc_auc_score(all_labels,all_probs)
        except:
            auc = 0.5

        tune.report({
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1": f1,
            "auc": auc,
            "kappa": kappa
        })


# ==========================================================
# FINAL EVALUATION
# ==========================================================

def evaluate_final_model(config,full_train_subjs,test_subj,global_data):

    set_reproducibility_seeds(GLOBAL_SEED)
    device="cuda" if torch.cuda.is_available() else "cpu"

    model = EEGTransformer(
        depth=config["depth"],
        eeg1_kernel_size=config["eeg1_kernel_size"],
        eeg1_dropout_rate=config["eeg1_dropout_rate"]
    ).to(device)

    train_ds = BCICIV2b_Dataset(
        global_data, full_train_subjs
    )

    test_ds = BCICIV2b_Dataset(
        global_data, [test_subj]
    )

    train_loader = DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
    test_loader = DataLoader(test_ds,batch_size=config["batch_size"],shuffle=False)

    optimizer = optim.Adam(model.parameters(),lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):

        model.train()
        for X,y in train_loader:
            X,y = X.to(device),y.to(device)
            optimizer.zero_grad()
            X = X.unsqueeze(1)
            logits = model(X)
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()

    # 🚨 NEW: Save the model weights to disk after training finishes 🚨
    save_path = f"CTNet_best_model_{test_subj}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[Save] Final trained model for {test_subj} saved to {save_path}")

    model.eval()
    all_preds=[]
    all_labels=[]
    all_probs=[]

    with torch.no_grad():
        for X,y in test_loader:
            X,y = X.to(device),y.to(device)
            X = X.unsqueeze(1)
            logits = model(X)
            probs = torch.softmax(logits,dim=1)
            preds = torch.argmax(probs,dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())

    acc = accuracy_score(all_labels,all_preds)
    bacc = balanced_accuracy_score(all_labels,all_preds)
    f1 = f1_score(all_labels,all_preds,average="macro")
    kappa = cohen_kappa_score(all_labels,all_preds)

    try:
        auc = roc_auc_score(all_labels,all_probs)
    except:
        auc = 0.5

    return dict(
        accuracy=acc,
        balanced_accuracy=bacc,
        f1=f1,
        auc=auc,
        kappa=kappa
    )


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main():

    all_subjects=['S01','S02','S03','S04','S05','S06','S07','S08','S09']

    ray.init(ignore_reinit_error=True)

    global_raw_data = preload_raw_data(all_subjects)
    global_data_ref = ray.put(global_raw_data)

    search_space = {
        # 🚨 FIX: Re-added model_class so train_any_model can find it
        "model_class": tune.choice(MODELS_TO_SEARCH),
        "lr": tune.loguniform(1e-5,1e-2),
        "batch_size": tune.choice([16,32,64,128]),
        "epochs":tune.choice([20, 30, 40, 50]),
        "depth": tune.choice([2,4,6]),
        "eeg1_kernel_size": tune.choice([16,32,64]),
        "eeg1_dropout_rate": tune.choice([0.25,0.5]),
    }

    results_list=[]

    print("Starting Nested LOO search")

    for i,test_subj in enumerate(all_subjects):

        val_idx=(i+1)%len(all_subjects)
        hpo_val_subj = all_subjects[val_idx]
        hpo_train_pool=[s for s in all_subjects if s!=test_subj and s!=hpo_val_subj]
        full_train_pool=[s for s in all_subjects if s!=test_subj]

        asha = ASHAScheduler(
            metric="balanced_accuracy",
            mode="max",
            max_t=100,
            grace_period=20,
            reduction_factor=2
        )

        # 🚨 FIX: Dynamic config injection (bypasses pickling errors)
        current_config = search_space.copy()
        current_config["train_subjs"] = hpo_train_pool
        current_config["val_subj"] = hpo_val_subj
        current_config["global_data_ref"] = global_data_ref

        # 🚨 FIX: tune.with_parameters is completely gone!
        analysis = tune.run(
            train_any_model, 
            config=current_config,
            scheduler=asha,
            num_samples=NUM_SAMPLES,
            resources_per_trial={"cpu":2,"gpu":1},
            max_concurrent_trials=1, 
            name=f"Nested_LOO_{test_subj}",
        )

        best_trial = analysis.get_best_trial("balanced_accuracy","max","last")

        final_metrics = evaluate_final_model(
            best_trial.config,
            full_train_pool,
            test_subj,
            global_raw_data
        )

        row={"Subject":test_subj,**final_metrics}

        # 🚨 FIX: Clean out the object refs so CSV export doesn't crash
        clean_config = {k: v for k, v in best_trial.config.items() if k not in ["train_subjs", "val_subj", "global_data_ref"]}
        row.update(clean_config)

        results_list.append(row)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df=pd.DataFrame(results_list)
    metric_cols=["accuracy","balanced_accuracy","f1","auc","kappa"]
    avg_row=df[metric_cols].mean().to_dict()
    avg_row["Subject"]="AVERAGE"

    df=pd.concat([df,pd.DataFrame([avg_row])],ignore_index=True)

    print("\nFINAL RESULTS")
    print(df)

    df.to_csv("Nested_LOO_results.csv",index=False)


if __name__=="__main__":
    main()