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
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(f"Dataset sizes - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    train_dataset = TUABLoader(os.path.join(root, "train"), train_files, target_length=target_length)
    val_dataset   = TUABLoader(os.path.join(root, "val"),   val_files,   target_length=target_length)
    test_dataset  = TUABLoader(os.path.join(root, "test"),  test_files,  target_length=target_length)
    
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
        
        # TUAB has 23 channels, but we only use channels that directly match EEGPT
        # EEGPT channels: FP1, FPZ, FP2, AF3, AF4, F7, F5, F3, F1, FZ, F2, F4, F6, F8,
        #                 FT7, FC5, FC3, FC1, FCZ, FC2, FC4, FC6, FT8, T7, C5, C3, C1, 
        #                 CZ, C2, C4, C6, T8, TP7, CP5, CP3, CP1, CPZ, CP2, CP4, CP6, 
        #                 TP8, P7, P5, P3, P1, PZ, P2, P4, P6, P8, PO7, PO3, POZ, PO4, 
        #                 PO8, O1, OZ, O2
        
        # TUAB channels in order: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2,
        #                         F7, F8, T3, T4, T5, T6, Fz, Cz, Pz, A1, A2, Fpz, Oz
        #                         0    1   2   3   4   5   6   7   8   9
        #                         10  11  12  13  14  15  16  17  18  19  20  21   22
        
        # Channel indices to keep (only direct EEGPT matches):
        # Keep: 0,1,2,3,4,5,6,7,8,9,10,11,16,17,18,21,22
        # Exclude: 12(T3), 13(T4), 14(T5), 15(T6), 19(A1), 20(A2)
        self.channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 21, 22]
        self.chans_num = len(self.channel_indices)  # 17 channels
        
        # EEGPT-compatible channel names (direct matches only, no remapping)
        use_channels_names = [
            "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "FZ", "CZ", "PZ", "FPZ", "OZ"
        ]

        # init model - input size [17, 1024]
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
        
        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path, weights_only=False)
        
        target_encoder_stat = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]] = v
        
        self.target_encoder.load_state_dict(target_encoder_stat)

        self.chan_conv = Conv1dWithConstraint(17, self.chans_num, 1, max_norm=1)
        
        self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(16*16, 2, max_norm=0.25)  # Binary classification for TUAB
       
        self.drop = torch.nn.Dropout(p=0.50)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # idx 0 = val, idx 1 = test (matches order in trainer.fit val dataloaders list)
        self.running_scores = {"train": [], "valid": [], "test": []}
        self.is_sanity = True
    
    def forward(self, x):
        B, C, T = x.shape
        
        # Select only the channels that directly match EEGPT (no remapping)
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

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label) * 1.0).mean()
        
        y_score = torch.softmax(logit.clone().detach().cpu(), dim=-1)[:, 1]
        self.running_scores["train"].append((label.clone().detach().cpu(), y_score))
        
        self.log('train_loss', loss,       on_epoch=True, on_step=False)
        self.log('train_acc',  accuracy,   on_epoch=True, on_step=False)
        self.log('data_avg',   x.mean(),   on_epoch=True, on_step=False)
        self.log('data_max',   x.max(),    on_epoch=True, on_step=False)
        self.log('data_min',   x.min(),    on_epoch=True, on_step=False)
        self.log('data_std',   x.std(),    on_epoch=True, on_step=False)
        
        return loss

    def on_train_epoch_start(self) -> None:
        self.running_scores["train"] = []
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")
        return super().on_train_epoch_end()

    # -----------------------------------------------------------------------
    # Validation + Test (both fed as val dataloaders, idx 0 = val, idx 1 = test)
    # -----------------------------------------------------------------------
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"] = []
        self.running_scores["test"]  = []
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss  = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label) * 1.0).mean()
        y_score  = torch.softmax(logit, dim=-1)[:, 1]

        split = "valid" if dataloader_idx == 0 else "test"
        self.running_scores[split].append(
            (label.clone().detach().cpu(), y_score.clone().detach().cpu())
        )

        self.log(f'{split}_loss', loss,     on_epoch=True, on_step=False, add_dataloader_idx=False)
        self.log(f'{split}_acc',  accuracy, on_epoch=True, on_step=False, add_dataloader_idx=False)
        
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity = False
            return super().on_validation_epoch_end()

        self._log_epoch_metrics("valid")
        self._log_epoch_metrics("test")
        return super().on_validation_epoch_end()

    # -----------------------------------------------------------------------
    # Shared metric logging helper
    # -----------------------------------------------------------------------
    def _log_epoch_metrics(self, split: str) -> None:
        records = self.running_scores[split]
        if not records:
            return
        label   = torch.cat([r[0] for r in records], dim=0)
        y_score = torch.cat([r[1] for r in records], dim=0)
        print(f"[{split}] label: {label.shape}, y_score: {y_score.shape}")

        metric_names = ["accuracy", "balanced_accuracy", "precision", "recall",
                        "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metric_names, True)
        for key, value in results.items():
            self.log(f'{split}_{key}', value, on_epoch=True, on_step=False, sync_dist=True)

    # -----------------------------------------------------------------------
    # Optimiser
    # -----------------------------------------------------------------------
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
# Training Script
# ============================================================================
if __name__ == "__main__":
    import math
    
    seed_torch(8)
    
    global max_epochs
    global steps_per_epoch
    global max_lr
    
    # Hyperparameters
    batch_size = 64
    max_epochs = 100
    max_lr = 4e-4
    
    # Data path
    data_root = "/mnt/scratch2/users/xw2336/LLM_Eva/Data/TUAB/v3.0.1/edf/processed"
    
    print("Loading TUAB dataset...")
    train_dataset, val_dataset, test_dataset = prepare_TUAB_dataset(data_root, target_length=1024)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
    valid_loader = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)
    
    steps_per_epoch = math.ceil(len(train_loader))
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    print("Initializing model...")
    model = LitEEGPTCausal()
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/EEGPT_TUAB_17ch/',
        filename='best-{epoch:02d}-{valid_loss:.4f}',
        monitor='valid_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    callbacks = [lr_monitor, checkpoint_callback]

    trainer = pl.Trainer(
        accelerator='cuda',
        precision=16,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=[
            pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_TUAB_17ch_afterPretrain_tb", version="v1"),
            pl_loggers.CSVLogger('./logs/',         name="EEGPT_TUAB_17ch_afterPretrain_csv"),
        ]
    )

    print("Starting training...")
    # Pass [valid_loader, test_loader] so PL runs both every validation epoch.
    # dataloader_idx=0 -> valid, dataloader_idx=1 -> test inside validation_step.
    trainer.fit(model, train_loader, [valid_loader, test_loader])
    # trainer.fit(model, train_loader, [valid_loader, test_loader],
    #             ckpt_path='./checkpoints/EEGPT_TUAB_17ch/last.ckpt')
