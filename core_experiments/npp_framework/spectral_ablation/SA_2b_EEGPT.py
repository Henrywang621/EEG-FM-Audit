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

# ============================================================
# Spectral Ablation Configuration
# ============================================================
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


class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, load_path="/users/yyang/EEGPT/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()    
        self.chans_num = 3

        use_channels_names = ['C3', 'CZ', 'C4']

        # init model
        target_encoder = EEGTransformer(
            img_size=[3, 1024],
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
        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        
        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path, weights_only=False)
        
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
        
                
        self.target_encoder.load_state_dict(target_encoder_stat)

        self.chan_conv       = Conv1dWithConstraint(3, self.chans_num, 1, max_norm=1)
        
        self.linear_probe1   =   LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2   =   LinearWithConstraint(16*16, 2, max_norm=0.25)
       
        self.drop           = torch.nn.Dropout(p=0.50)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True
        
    def mixup_data(self, x, y, alpha=None):
        lam = torch.rand(1).to(x) if alpha is None else alpha
        lam = torch.max(lam, 1 - lam)

        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]

        return mixed_x, mixed_y
    
    def forward(self, x):
        B, C, T = x.shape
        x = x/10
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
        y = F.one_hot(y.long(), num_classes=2).float()
        
        label = y
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==torch.argmax(label, dim=-1))*1.0).mean()
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False)
        self.log('data_max', x.max(), on_epoch=True, on_step=False)
        self.log('data_min', x.min(), on_epoch=True, on_step=False)
        self.log('data_std', x.std(), on_epoch=True, on_step=False)
        
        return loss
        
        
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1)==label)*1.0).mean()
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)
        
        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))

        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters())+
            list(self.linear_probe1.parameters())+
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': None,
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
        
# ============================================================
# Evaluation with Spectral Ablation
# ============================================================
from dataloader import *
from pytorch_lightning.callbacks import ModelCheckpoint
import math
seed_torch(8)

import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
import numpy as np

# Sampling frequency for BCICIV2b
FS = 250  # Hz — adjust if your data uses a different sampling rate

# Ablation conditions: Baseline (no ablation) + each frequency band
ablation_conditions = ['Baseline'] + list(FREQ_BANDS.keys())

# Store results: {condition: {subject: {metric: value}}}
all_results = {cond: {} for cond in ablation_conditions}
METRIC_NAMES = ['accuracy', 'f1', 'kappa', 'auroc']

for i in range(1, 10):
    batch_size = 64
    current_subj = f'S{i:02d}'
    
    # Load validation dataset
    valid_dataset = BCICIV2bLoader(
        subject_ids=[current_subj],
        window_index=8,
        window_size=500,
        step_size=100,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model from checkpoint
    checkpoint_path = f'/users/yyang/EEGPT/checkpoints/subject_{i:02d}/last.ckpt'
    model = LitEEGPTCausal.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda()
    
    # Run each ablation condition
    for cond in ablation_conditions:
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in valid_loader:
                x, y = batch
                x, y = x.cuda(), y.cuda()
                
                # Apply spectral ablation (Baseline returns x unchanged)
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
        f1 = f1_score(all_labels, all_preds, average='macro')
        kappa = cohen_kappa_score(all_labels, all_preds)
        try:
            auroc = roc_auc_score(all_labels, all_probs[:,1])
        except ValueError:
            auroc = float('nan')
        
        all_results[cond][current_subj] = {
            'accuracy': accuracy, 'f1': f1, 'kappa': kappa, 'auroc': auroc
        }

    # Print per-subject results
    print(f"\n--- Subject {i:02d} ({current_subj}) ---")
    for cond in ablation_conditions:
        r = all_results[cond][current_subj]
        print(f"  {cond:>12s}: Acc={r['accuracy']:.4f} | F1={r['f1']:.4f} | Kappa={r['kappa']:.4f} | AUC-ROC={r['auroc']:.4f}")

# ============================================================
# Summary Tables (one per metric)
# ============================================================
for metric in METRIC_NAMES:
    print("\n" + "="*80)
    print(f"SPECTRAL ABLATION RESULTS — {metric.upper()}")
    print("="*80)

    header = f"{'Condition':>12s} | " + " | ".join([f'S{i:02d}' for i in range(1, 10)]) + " |  Mean"
    print(header)
    print("-" * len(header))

    for cond in ablation_conditions:
        vals = [all_results[cond][f'S{i:02d}'][metric] for i in range(1, 10)]
        row = f"{cond:>12s} | " + " | ".join([f"{v:.3f}" for v in vals]) + f" | {np.nanmean(vals):.3f}"
        print(row)

# ============================================================
# Drop from Baseline (one per metric)
# ============================================================
for metric in METRIC_NAMES:
    print("\n" + "="*80)
    print(f"{metric.upper()} DROP FROM BASELINE (Baseline - Ablated)")
    print("="*80)
    header = f"{'Condition':>12s} | " + " | ".join([f'S{i:02d}' for i in range(1, 10)]) + " |  Mean"
    print(header)
    print("-" * len(header))

    for cond in ablation_conditions:
        if cond == 'Baseline':
            continue
        drops = [all_results['Baseline'][f'S{i:02d}'][metric] - all_results[cond][f'S{i:02d}'][metric] for i in range(1, 10)]
        row = f"{cond:>12s} | " + " | ".join([f"{d:+.3f}" for d in drops]) + f" | {np.nanmean(drops):+.3f}"
        print(row)
