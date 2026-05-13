"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

#from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import math
import numpy as np
import os
from downstream_dataset import TUABLoader, TUEVLoader, TUSLLoader, HMCLoader, WorkloadLoader, CCDLoader, BCICIV2bLoader
from metrics import binary_metrics_fn, multiclass_metrics_fn
from sklearn.metrics import confusion_matrix


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def prepare_TUEV_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test data loader
    train_dataset = TUEVLoader(
        os.path.join(
            root, "processed_train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    test_dataset = TUEVLoader(
        os.path.join(
            root, "processed_test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    val_dataset = TUEVLoader(
        os.path.join(
            root, "processed_eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUAB_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset

def prepare_BCI2b_dataset(root, train_subjects, val_subjects, test_subjects, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    """
    Args:
        root (str): Root directory containing the BCI IV 2b .npy files.
        train_subjects (list): List of Subject IDs for training (e.g. ['S01', 'S02', ...])
        val_subjects (list): List of Subject IDs for validation.
        test_subjects (list): List of Subject IDs for testing.
        is_instruct (bool): Whether to use instruction tuning (text prompts).
    """
    
    print(f"Splitting Subjects -> Train: {len(train_subjects)} | Val: {len(val_subjects)} | Test: {len(test_subjects)}")

    # 1. Prepare Training Dataset
    # is_val=False ensures we generate the full text targets for training
    train_dataset = BCICIV2bLoader(
        subject_ids=train_subjects,
        root_path=root,
        is_instruct=is_instruct,
        is_val=False, 
        eeg_max_len=eeg_max_len,
        text_max_len=text_max_len
    )

    # 2. Prepare Validation Dataset
    # is_val=True hides the answer in the text prompt (for evaluation)
    val_dataset = BCICIV2bLoader(
        subject_ids=val_subjects,
        root_path=root,
        is_instruct=is_instruct,
        is_val=True,
        eeg_max_len=eeg_max_len,
        text_max_len=text_max_len
    )

    # 3. Prepare Test Dataset
    test_dataset = BCICIV2bLoader(
        subject_ids=test_subjects,
        root_path=root,
        is_instruct=is_instruct,
        is_val=True,
        eeg_max_len=eeg_max_len,
        text_max_len=text_max_len
    )

    print(f"Samples Loaded -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    return train_dataset, test_dataset, val_dataset

def prepare_CCD_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    
    tr_dir = "processed_data4tr"
    val_dir = "processed_data4val"
    test_dir = "processed_data4test"

    # tr_dir = "AT_processed_tr"
    # val_dir = "AT_processed_val"
    # test_dir = "AT_processed_test"
    
    train_files = os.listdir(os.path.join(root, tr_dir))
    val_files = os.listdir(os.path.join(root, val_dir))
    test_files = os.listdir(os.path.join(root, test_dir))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = CCDLoader(os.path.join(root, tr_dir), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = CCDLoader(os.path.join(root, test_dir), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = CCDLoader(os.path.join(root, val_dir), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUSL_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUSLLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = TUSLLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = TUSLLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_HMC_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = HMCLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = HMCLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = HMCLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_Workload_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = WorkloadLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = WorkloadLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = WorkloadLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


# def get_metrics(output, target, metrics, is_binary):
#     # 1. Ensure inputs are CPU Numpy arrays (Safeguard)
#     if hasattr(output, 'detach'): output = output.detach().cpu().numpy()
#     if hasattr(target, 'detach'): target = target.detach().cpu().numpy()

#     if is_binary:
#         # ... (Binary logic remains unchanged) ...
#         if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:
#             results = binary_metrics_fn(target, output, metrics=metrics)
#         else:
#             results = {"accuracy": 0.0, "balanced_accuracy": 0.0, "pr_auc": 0.0, "roc_auc": 0.0}
#     else:
#         # multiclass_metrics_fn usually handles raw probabilities internally for things like AUC
#         results = multiclass_metrics_fn(target, output, metrics=metrics)

#     # --- FIX STARTS HERE ---
    
#     # 2. Prepare data for Confusion Matrix (Must be 1D Class IDs)
    
#     # Process Prediction: Convert Probabilities (N, C) -> Class Indices (N,)
#     if output.ndim > 1:
#         y_pred_cm = np.argmax(output, axis=1)
#     else:
#         y_pred_cm = output # Already indices (rare for models) or binary thresholds needed

#     # Process Target: Convert One-Hot (N, C) -> Class Indices (N,) if necessary
#     if target.ndim > 1:
#         y_true_cm = np.argmax(target, axis=1)
#     else:
#         y_true_cm = target

#     # 3. Calculate Confusion Matrix with the processed variables
#     cm = confusion_matrix(y_true_cm, y_pred_cm)
    
#     # --- FIX ENDS HERE ---

#     per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)  # avoid div by 0

#     # add them into results
#     for i, acc in enumerate(per_class_acc):
#         results[f"class_{i}_accuracy"] = acc

#     results["confusion_matrix"] = cm.tolist()
#     return results

def get_metrics(output, target, metrics, is_binary):
    if hasattr(output, 'detach'): output = output.detach().cpu().numpy()
    if hasattr(target, 'detach'): target = target.detach().cpu().numpy()

    if is_binary:
        # Standard binary metrics (ROC, PR, etc.)
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:
            results = binary_metrics_fn(target, output, metrics=metrics)
        else:
            results = {"accuracy": 0.0, "balanced_accuracy": 0.0, "pr_auc": 0.0, "roc_auc": 0.0}
        
        # FIX: For binary (N, 1), predictions are based on a 0.0 threshold (logits)
        y_pred_cm = (output.flatten() > 0).astype(int)
        y_true_cm = target.astype(int)
    else:
        results = multiclass_metrics_fn(target, output, metrics=metrics)
        # Standard multiclass argmax
        y_pred_cm = np.argmax(output, axis=1)
        y_true_cm = np.argmax(target, axis=1) if target.ndim > 1 else target

    # 3. Calculate Confusion Matrix using corrected variables
    cm = confusion_matrix(y_true_cm, y_pred_cm)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    for i, acc in enumerate(per_class_acc):
        results[f"class_{i}_accuracy"] = acc

    results["confusion_matrix"] = cm.tolist()
    return results