import random 
import os
import torch
from torch import nn
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import random
import os 
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
from scipy.signal import resample
import pickle

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)

from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from utils_eval import get_metrics

# ============================================================================
# Spectral Ablation Configuration
# ============================================================================
FREQ_BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 75),
    'Low_Gamma': (30, 50), 
    'High_Gamma': (50, 75)
}

def apply_spectral_ablation(x, fs, band_name):
    """
    Zeros out frequency coefficients in the specified band using FFT.
    """
    # 0. Check for "Baseline" or invalid band
    if band_name not in FREQ_BANDS:
        return x
    
    low_cut, high_cut = FREQ_BANDS[band_name]
    
    # 1. Perform Real FFT
    fft_x = torch.fft.rfft(x, dim=-1)
    
    # 2. Compute Frequency Bins
    n = x.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1/fs).to(x.device)
    
    # 3. Create Mask (1 = Keep, 0 = Kill)
    band_mask = (freqs >= low_cut) & (freqs <= high_cut)
    keep_mask = ~band_mask 
    
    # Broadcast mask to match batch/channel dims
    keep_mask = keep_mask.view(1, 1, -1)
    
    # 4. Apply Ablation
    fft_ablated = fft_x * keep_mask
    
    # 5. Inverse FFT to get back to time domain
    x_ablated = torch.fft.irfft(fft_ablated, n=n, dim=-1)
    
    return x_ablated

# ============================================================================
# TUAB Dataset Loader
# ============================================================================
class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200, target_length=1024):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.target_length = target_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]  # Shape: [23, 2000]
        
        # Resample from 2000 to target_length (1024)
        if X.shape[-1] != self.target_length:
            X = resample(X, self.target_length, axis=-1)
        
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y
    
def prepare_TUAB_dataset(root, target_length=1024):
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(f"Dataset sizes - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files, target_length=target_length)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files, target_length=target_length)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files, target_length=target_length)
    
    return train_dataset, val_dataset, test_dataset

# ============================================================================
# Model Definition - WITHOUT REMAPPING (17 channels)
# ============================================================================
class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, load_path="/users/yyang/EEGPT/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()    
        
        print("=" * 80)
        print("VERSION: WITHOUT REMAPPING - 17 CHANNELS")
        print("NO remapping of old nomenclature (T3, T4, T5, T6)")
        print("Excluded: A1, A2, T3, T4, T5, T6 (not in EEGPT vocabulary)")
        print("=" * 80)
        
        self.channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 21, 22]
        self.chans_num = len(self.channel_indices)  # 17 channels
        
        use_channels_names = [
            "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "FZ", "CZ", "PZ", "FPZ", "OZ"
        ]

        target_encoder = EEGTransformer(
            img_size=[17, 1024],
            patch_size=32*2,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        self.target_encoder = target_encoder
        self.chans_id = target_encoder.prepare_chan_ids(use_channels_names)
        
        pretrain_ckpt = torch.load(load_path, weights_only=False)
        
        target_encoder_stat = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]] = v
        
        self.target_encoder.load_state_dict(target_encoder_stat)

        self.chan_conv = Conv1dWithConstraint(17, self.chans_num, 1, max_norm=1)
        
        self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(16*16, 2, max_norm=0.25)
       
        self.drop = torch.nn.Dropout(p=0.50)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train": [], "valid": [], "test": []}
        self.is_sanity = True
    
    def forward(self, x):
        B, C, T = x.shape
        
        x = x[:, self.channel_indices, :]  # [B, 17, T]
        
        x = x / 10
        x = self.chan_conv(x)
        self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x))
        
        h = z.flatten(2)
        h = self.linear_probe1(self.drop(h))
        h = h.flatten(1)
        h = self.linear_probe2(h)
        
        return x, h

    def training_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label) * 1.0).mean()
        
        y_score = logit.clone().detach().cpu()
        y_score = torch.softmax(y_score, dim=-1)[:, 1]
        self.running_scores["train"].append((label.clone().detach().cpu(), y_score))
        
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)
        
        return loss
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"] = []
        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity = False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x, y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)
        
        for key, value in results.items():
            self.log('valid_' + key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label) * 1.0).mean()
        
        y_score = logit
        y_score = torch.softmax(y_score, dim=-1)[:, 1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)
        
        return loss
    
    def on_train_epoch_start(self) -> None:
        self.running_scores["train"] = []
        return super().on_train_epoch_start()
    
    def on_train_epoch_end(self) -> None:
        from sklearn import metrics
        
        label, y_score = [], []
        for x, y in self.running_scores["train"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        rocauc = metrics.roc_auc_score(label, y_score)
        self.log('train_rocauc', rocauc, on_epoch=True, on_step=False)
        return super().on_train_epoch_end()
    
    def on_test_epoch_start(self) -> None:
        self.running_scores["test"] = []
        return super().on_test_epoch_start()
    
    def on_test_epoch_end(self) -> None:
        from sklearn import metrics
        
        label, y_score = [], []
        for x, y in self.running_scores["test"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        rocauc = metrics.roc_auc_score(label, y_score)
        self.log('test_rocauc', rocauc, on_epoch=True, on_step=False)
        return super().on_test_epoch_end()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label) * 1.0).mean()
        
        y_score = logit
        y_score = torch.softmax(y_score, dim=-1)[:, 1]
        self.running_scores["test"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        
        self.log('test_loss', loss, on_epoch=True, on_step=False)
        self.log('test_acc', accuracy, on_epoch=True, on_step=False)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters()) +
            list(self.linear_probe1.parameters()) +
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr, 
            steps_per_epoch=steps_per_epoch, 
            epochs=max_epochs, 
            pct_start=0.2)
        
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': None,
        }
      
        return {'optimizer': optimizer, 'lr_scheduler': lr_dict}

# ============================================================================
# Evaluation with Spectral Ablation
# ============================================================================
if __name__ == "__main__":
    import math
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, roc_auc_score
    import numpy as np
    
    seed_torch(8)
    
    # Hyperparameters
    batch_size = 64
    
    # TUAB effective sampling rate after resampling: 
    # Original: 200 Hz, 2000 samples -> resampled to 1024 samples
    # Effective fs = 200 * (1024 / 2000) = 102.4 Hz
    FS = 102.4  # Effective sampling rate after resampling to 1024
    
    # Data path
    data_root = "/mnt/scratch2/users/xw2336/LLM_Eva/Data/TUAB/v3.0.1/edf/processed"
    
    # Load test dataset
    print("Loading TUAB test dataset...")
    _, _, test_dataset = prepare_TUAB_dataset(data_root, target_length=1024)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Test batches: {len(test_loader)}")
    print(f"Test samples: {len(test_dataset)}")
    
    checkpoint_paths = ['/users/yyang/EEGPT/EEGPT/downstream/checkpoints/EEGPT_TUAB_17ch_seed_42/best.ckpt',
                       '/users/yyang/EEGPT/EEGPT/downstream/checkpoints/EEGPT_TUAB_17ch_seed_3407/best.ckpt',
                       '/users/yyang/EEGPT/EEGPT/downstream/checkpoints/EEGPT_TUAB_17ch_seed_6/best.ckpt',
                       '/users/yyang/EEGPT/EEGPT/downstream/checkpoints/EEGPT_TUAB_17ch_seed_16/best.ckpt',
                       '/users/yyang/EEGPT/EEGPT/downstream/checkpoints/EEGPT_TUAB_17ch_seed_66/best.ckpt']
    for path in checkpoint_paths:
        # Load model from checkpoint
        checkpoint_path = path
        print(f"\nLoading model from: {checkpoint_path}")
        model = LitEEGPTCausal.load_from_checkpoint(checkpoint_path)
        model.eval()
        model.cuda()
    
        # Ablation conditions
        ablation_conditions = ['Baseline'] + list(FREQ_BANDS.keys())
    
        # Store results: {condition: {metric: value}}
        all_results = {}
        class_names = ['Normal', 'Abnormal']
    
        for cond in ablation_conditions:
            all_preds = []
            all_labels = []
            all_probs = []
        
            print(f"\nRunning inference: {cond}...")
            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch
                    x, y = x.cuda(), y.cuda()
                
                    # Apply spectral ablation BEFORE forward pass
                    # Ablation on raw input (all 23 channels), before channel selection in forward()
                    x_ablated = apply_spectral_ablation(x, FS, cond)
                
                    output = model(x_ablated)
                    if isinstance(output, tuple):
                        logits = output[1]
                    else:
                        logits = output
                
                    preds = torch.argmax(logits, dim=1)
                    probs = torch.softmax(logits, dim=-1)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(y.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
        
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)
        
            accuracy = accuracy_score(all_labels, all_preds)
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            kappa = cohen_kappa_score(all_labels, all_preds)
            try:
                auroc = roc_auc_score(all_labels, all_probs[:, 1])  # Binary: prob of positive class
            except ValueError:
                auroc = float('nan')
            cm = confusion_matrix(all_labels, all_preds)
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
            all_results[cond] = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'f1': f1,
                'kappa': kappa,
                'auroc': auroc,
                'confusion_matrix': cm,
                'per_class_accuracy': per_class_acc,
            }
        
            print(f"  {cond}: Acc={accuracy*100:.2f}%, Bal_Acc={balanced_acc*100:.2f}%, F1={f1:.4f}, Kappa={kappa:.4f}, AUC-ROC={auroc:.4f}")
    
        # ========================================================================
        # Summary Table
        # ========================================================================
        print("\n" + "=" * 80)
        print("SPECTRAL ABLATION RESULTS SUMMARY - TUAB DATASET")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Effective Sampling Rate: {FS} Hz")
        print(f"Total Test Samples: {len(test_dataset)}")
        print("=" * 80)
    
        header = f"{'Condition':>12s} | {'Accuracy':>10s} | {'Bal. Acc':>10s} | {'F1':>8s} | {'Kappa':>8s} | {'AUC-ROC':>8s} | {'Normal':>10s} | {'Abnormal':>10s}"
        print(header)
        print("-" * len(header))
    
        for cond in ablation_conditions:
            r = all_results[cond]
            pca = r['per_class_accuracy']
            row = (f"{cond:>12s} | {r['accuracy']*100:>9.2f}% | {r['balanced_accuracy']*100:>9.2f}% | "
                   f"{r['f1']:>8.4f} | {r['kappa']:>8.4f} | {r['auroc']:>8.4f} | "
                   f"{pca[0]*100:>9.2f}% | {pca[1]*100:>9.2f}%")
            print(row)
    
        # ========================================================================
        # Drop from Baseline
        # ========================================================================
        print("\n" + "=" * 80)
        print("ACCURACY DROP FROM BASELINE (Baseline - Ablated)")
        print("=" * 80)
    
        header = f"{'Condition':>12s} | {'Acc Drop':>10s} | {'Bal Acc Drop':>12s} | {'F1 Drop':>10s} | {'Kappa Drop':>11s} | {'AUROC Drop':>11s}"
        print(header)
        print("-" * len(header))
    
        baseline_acc = all_results['Baseline']['accuracy']
        baseline_bal = all_results['Baseline']['balanced_accuracy']
        baseline_f1 = all_results['Baseline']['f1']
        baseline_kappa = all_results['Baseline']['kappa']
        baseline_auroc = all_results['Baseline']['auroc']
    
        for cond in ablation_conditions:
            if cond == 'Baseline':
                continue
            acc_drop = baseline_acc - all_results[cond]['accuracy']
            bal_drop = baseline_bal - all_results[cond]['balanced_accuracy']
            f1_drop = baseline_f1 - all_results[cond]['f1']
            kappa_drop = baseline_kappa - all_results[cond]['kappa']
            auroc_drop = baseline_auroc - all_results[cond]['auroc']
            row = (f"{cond:>12s} | {acc_drop*100:>+9.2f}% | {bal_drop*100:>+11.2f}% | "
                   f"{f1_drop:>+10.4f} | {kappa_drop:>+11.4f} | {auroc_drop:>+11.4f}")
            print(row)
    
        # ========================================================================
        # Confusion Matrices
        # ========================================================================
        print("\n" + "=" * 80)
        print("CONFUSION MATRICES")
        print("=" * 80)
    
        for cond in ablation_conditions:
            print(f"\n{cond}:")
            print(all_results[cond]['confusion_matrix'])
    
        print("\n" + "=" * 80)

