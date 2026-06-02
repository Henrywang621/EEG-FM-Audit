import random 
import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import os 
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
from scipy.signal import resample
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score, MulticlassAUROC, MulticlassCohenKappa

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
from utils import temporal_interpolation
from sklearn import metrics
from utils_eval import get_metrics


# EEGPT original channels
eegpt_channels = [
    'FP1', 'FPZ', 'FP2',
    "AF7", 'AF3', 'AF4', "AF8",
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8',
    'O1', 'OZ', 'O2'
]

# TUEV channels (23 total)
tuev_channel_names = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'
]

# Only keep channels that directly match (no conversions)
# 15 overlapping channels: FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, FZ, CZ, PZ
use_channels_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'FZ', 'CZ', 'PZ']

# Get indices of channels to keep from TUEV data
tuev_channels_to_keep = [i for i, ch in enumerate(tuev_channel_names) if ch in use_channels_names]

print(f"Keeping {len(tuev_channels_to_keep)} channels: {[tuev_channel_names[i] for i in tuev_channels_to_keep]}")

class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, load_path="/users/yyang/EEGPT/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt", num_classes=6):
        super().__init__()    
        self.chans_num = len(use_channels_names)
        self.num_classes = num_classes
        
        # Initialize balanced accuracy metrics (macro-averaged accuracy)
        self.train_bal_acc = Accuracy(task='multiclass', num_classes=num_classes, average='macro')
        self.valid_bal_acc = Accuracy(task='multiclass', num_classes=num_classes, average='macro')
        self.test_bal_acc = Accuracy(task='multiclass', num_classes=num_classes, average='macro')
        
        # F1 Score (macro-averaged)
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.valid_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        
        # AUC-ROC (macro-averaged, needs softmax probabilities)
        self.train_auroc = MulticlassAUROC(num_classes=num_classes, average='macro')
        self.valid_auroc = MulticlassAUROC(num_classes=num_classes, average='macro')
        self.test_auroc = MulticlassAUROC(num_classes=num_classes, average='macro')
        
        # Cohen's Kappa
        self.train_kappa = MulticlassCohenKappa(num_classes=num_classes)
        self.valid_kappa = MulticlassCohenKappa(num_classes=num_classes)
        self.test_kappa = MulticlassCohenKappa(num_classes=num_classes)
        
        # init model
        target_encoder = EEGTransformer(
            img_size=[self.chans_num, 2*256],
            patch_size=32*2,
            patch_stride = 32,
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
        
                
        self.target_encoder.load_state_dict(target_encoder_stat, strict=False)

        self.chan_conv       = Conv1dWithConstraint(self.chans_num, self.chans_num, 1, max_norm=1)
        
        self.linear_probe1   =   LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2   =   LinearWithConstraint(240, self.num_classes, max_norm=0.25)
       
        self.drop           = torch.nn.Dropout(p=0.50)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True
        
    
    def forward(self, x):
        B, C, T = x.shape
        x = x/10
        x = x - x.mean(dim=-2, keepdim=True)
        x = temporal_interpolation(x, 2*256)
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
        accuracy = ((preds==label)*1.0).mean()
        
        # Update and log balanced accuracy
        self.train_bal_acc(preds, label)
        self.log('train_bal_acc', self.train_bal_acc, on_epoch=True, on_step=False, prog_bar=True)
        
        # Update and log F1, AUC-ROC, Kappa
        self.train_f1(preds, label)
        self.log('train_f1', self.train_f1, on_epoch=True, on_step=False)
        self.train_auroc(torch.softmax(logit, dim=-1), label)
        self.log('train_auroc', self.train_auroc, on_epoch=True, on_step=False)
        self.train_kappa(preds, label)
        self.log('train_kappa', self.train_kappa, on_epoch=True, on_step=False)
        
        self.running_scores["train"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        
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
            
        label, logits = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            logits.append(y)
        label = torch.cat(label, dim=0)
        logits = torch.cat(logits, dim=0)
        
        y_score = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        
        print(label.shape, logits.shape)
        
        accuracy = (preds == label).float().mean()
        self.log('valid_accuracy', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        
        # Update and log balanced accuracy
        self.valid_bal_acc(preds, label)
        self.log('valid_bal_acc', self.valid_bal_acc, on_epoch=True, on_step=False, prog_bar=True)
        
        # Update and log F1, AUC-ROC, Kappa
        self.valid_f1(preds, label)
        self.log('valid_f1', self.valid_f1, on_epoch=True, on_step=False, prog_bar=True)
        self.valid_auroc(torch.softmax(logit, dim=-1), label)
        self.log('valid_auroc', self.valid_auroc, on_epoch=True, on_step=False, prog_bar=True)
        self.valid_kappa(preds, label)
        self.log('valid_kappa', self.valid_kappa, on_epoch=True, on_step=False, prog_bar=True)
        
        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))

        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)
        
        return loss
        
    def on_train_epoch_start(self) -> None:
        self.running_scores["train"]=[]
        return super().on_train_epoch_start()
        
    def on_train_epoch_end(self) -> None:
        label, logits = [], []
        for x,y in self.running_scores["train"]:
            label.append(x)
            logits.append(y)
        label = torch.cat(label, dim=0)
        logits = torch.cat(logits, dim=0)
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == label).float().mean()
        self.log('train_epoch_acc', accuracy, on_epoch=True, on_step=False)
        return super().on_train_epoch_end()
        
    def on_test_epoch_start(self) -> None:
        self.running_scores["test"]=[]
        return super().on_test_epoch_start()
        
    def on_test_epoch_end(self) -> None:
        label, logits = [], []
        for x,y in self.running_scores["test"]:
            label.append(x)
            logits.append(y)
        label = torch.cat(label, dim=0)
        logits = torch.cat(logits, dim=0)
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == label).float().mean()
        self.log('test_accuracy', accuracy, on_epoch=True, on_step=False)
        return super().on_test_epoch_end()
    
    def test_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        
        # Update and log balanced accuracy
        self.test_bal_acc(preds, label)
        self.log('test_bal_acc', self.test_bal_acc, on_epoch=True, on_step=False)
        
        # Update and log F1, AUC-ROC, Kappa
        self.test_f1(preds, label)
        self.log('test_f1', self.test_f1, on_epoch=True, on_step=False)
        self.test_auroc(torch.softmax(logit, dim=-1), label)
        self.log('test_auroc', self.test_auroc, on_epoch=True, on_step=False)
        self.test_kappa(preds, label)
        self.log('test_kappa', self.test_kappa, on_epoch=True, on_step=False)
        
        self.running_scores["test"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        
        self.log('test_loss', loss, on_epoch=True, on_step=False)
        self.log('test_acc', accuracy, on_epoch=True, on_step=False)
        
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


# Dataset loader for TUEV
class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, target_length=1024):
        self.root = root
        self.files = files
        self.target_length = target_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        import pickle
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        
        X = sample["signal"]
        
        if X.dtype.byteorder == '>' or (X.dtype.byteorder == '|' and X.dtype.str.endswith('f4') == False):
            X = X.byteswap().newbyteorder('=')
        
        X = np.array(X, dtype=np.float32, copy=True)
        X = np.ascontiguousarray(X)
        
        # Keep only overlapping channels (15 channels)
        X = X[tuev_channels_to_keep, :]
        
        # Downsample from 2000 to 1024
        X = resample(X, self.target_length, axis=-1)
        X = np.ascontiguousarray(X, dtype=np.float32)
            
        Y = int(sample["label"][0] - 1)  # Labels are 1-6, convert to 0-5
        
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        
        return X_tensor, Y


# Load configs
import math
seed_torch(9)

root = '/mnt/scratch2/users/xw2336/LLM_Eva/Data/TUEV/v2.0.1/edf/processed'

global max_epochs
global steps_per_epoch
global max_lr

batch_size = 64
max_epochs = 100
max_lr = 4e-4

# Load file lists
train_files = os.listdir(os.path.join(root, "processed_train"))
val_files = os.listdir(os.path.join(root, "processed_eval"))
test_files = os.listdir(os.path.join(root, "processed_test"))

# Create datasets
train_dataset = TUEVLoader(os.path.join(root, "processed_train"), train_files, target_length=1024)
val_dataset = TUEVLoader(os.path.join(root, "processed_eval"), val_files, target_length=1024)
test_dataset = TUEVLoader(os.path.join(root, "processed_test"), test_files, target_length=1024)

print(f"Dataset sizes - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

steps_per_epoch = math.ceil(len(train_loader))

# Initialize model
model = LitEEGPTCausal(num_classes=6)

# Setup callbacks
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor]



# Setup trainer
trainer = pl.Trainer(
    accelerator='cuda',
    max_epochs=max_epochs,
    callbacks=callbacks,  # Add to existing callbacks
    enable_checkpointing=False,          # Enable this!
    logger=[
        pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_TUEV_Baseline_Task1_tb", version="v1"),
        pl_loggers.CSVLogger('./logs/', name="EEGPT_TUEV_Baseline_Task1_csv")
    ]
)

# Train
trainer.fit(model, train_loader, valid_loader)

# Test
trainer.test(model, test_loader)
