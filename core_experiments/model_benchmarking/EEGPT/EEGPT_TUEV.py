import argparse
import math
import copy
import numpy as np
import random
import os
import pickle
from functools import partial
from scipy.signal import resample

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
# from utils import temporal_interpolation
from utils_eval import get_metrics

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Channel alignment for TUEV (15 overlapping channels)
tuev_channel_names = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'
]
use_channels_names = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'FZ', 'CZ', 'PZ'
]
tuev_channels_to_keep = [i for i, ch in enumerate(tuev_channel_names) if ch in use_channels_names]


# ============================================================================
# TUEV Dataset Loader
# ============================================================================
class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, target_length=1024):
        self.root = root
        self.files = files
        self.target_length = target_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.files[index]), "rb") as f:
            sample = pickle.load(f)
        
        X = sample["signal"]
        if X.dtype.byteorder == '>' or (X.dtype.byteorder == '|' and not X.dtype.str.endswith('f4')):
            X = X.byteswap().newbyteorder('=')
        
        X = np.array(X, dtype=np.float32, copy=True)
        X = np.ascontiguousarray(X)
        X = X[tuev_channels_to_keep, :]
        
        if X.shape[-1] != self.target_length:
            X = resample(X, self.target_length, axis=-1)
        X = np.ascontiguousarray(X, dtype=np.float32)
            
        Y = int(sample["label"][0] - 1)  # Map 1-6 -> 0-5
        return torch.as_tensor(X, dtype=torch.float32), Y


def prepare_TUEV_dataset(root, target_length=1024):
    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files   = os.listdir(os.path.join(root, "processed_eval"))
    test_files  = os.listdir(os.path.join(root, "processed_test"))

    print(f"Dataset sizes - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    train_dataset = TUEVLoader(os.path.join(root, "processed_train"), train_files, target_length=target_length)
    val_dataset   = TUEVLoader(os.path.join(root, "processed_eval"),  val_files,   target_length=target_length)
    test_dataset  = TUEVLoader(os.path.join(root, "processed_test"),  test_files,  target_length=target_length)
    
    return train_dataset, val_dataset, test_dataset


def temporal_interpolation(x, target_len):
    # x shape: (Batch, Channels, Time)
    return F.interpolate(x, size=target_len, mode='linear', align_corners=True)


# ============================================================================
# Lightning Module Definition
# ============================================================================
class LitEEGPTCausal(pl.LightningModule):
    def __init__(
        self, 
        lr=4e-4, 
        weight_decay=0.01, 
        epochs=100, 
        steps_per_epoch=100, 
        num_classes=6,
        load_path="/homes/yy2541/Henry_shine3/EEGPT/checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt"
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.chans_num = len(use_channels_names)
        self.num_classes = num_classes

        target_encoder = EEGTransformer(
            img_size=[self.chans_num, 2*256],
            patch_size=32*2,
            patch_stride=32,
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
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
            
        self.target_encoder = target_encoder
        self.chans_id = target_encoder.prepare_chan_ids(use_channels_names)
        
        pretrain_ckpt = torch.load(load_path, weights_only=False)
        target_encoder_stat = {
            k[15:]: v for k, v in pretrain_ckpt['state_dict'].items() if k.startswith("target_encoder.")
        }
        self.target_encoder.load_state_dict(target_encoder_stat, strict=False)

        self.chan_conv = Conv1dWithConstraint(self.chans_num, self.chans_num, 1, max_norm=1)
        self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(240, self.num_classes, max_norm=0.25)
       
        # Freeze target encoder for linear probing / DDP stability
        self.target_encoder.requires_grad_(False)
        self.chan_conv.requires_grad_(True)
        self.linear_probe1.requires_grad_(True)
        self.linear_probe2.requires_grad_(True)

        self.drop = nn.Dropout(p=0.50)
        self.loss_fn = nn.CrossEntropyLoss()
        self.running_scores = {"train": [], "valid": [], "test": []}
        self.is_sanity = True
    
    def forward(self, x):
        x = x / 10.0
        x = x - x.mean(dim=-2, keepdim=True)
        x = temporal_interpolation(x, 2 * 256)
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
        
        y_score = torch.softmax(logit.clone().detach().cpu(), dim=-1)
        self.running_scores["train"].append((label.clone().detach().cpu(), y_score))
        
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_start(self) -> None:
        self.running_scores["train"] = []
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"] = []
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label) * 1.0).mean()
        y_score = torch.softmax(logit, dim=-1)

        self.running_scores["valid"].append(
            (label.clone().detach().cpu(), y_score.clone().detach().cpu())
        )
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity = False
            return super().on_validation_epoch_end()
        self._log_epoch_metrics("valid")
        return super().on_validation_epoch_end()
    
    def on_test_epoch_start(self) -> None:
        self.running_scores["test"] = []
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label) * 1.0).mean()
        y_score = torch.softmax(logit, dim=-1)

        self.running_scores["test"].append(
            (label.clone().detach().cpu(), y_score.clone().detach().cpu())
        )
        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")
        return super().on_test_epoch_end()

    def _log_epoch_metrics(self, split: str) -> None:
        records = self.running_scores[split]
        if not records:
            return
        label = torch.cat([r[0] for r in records], dim=0)
        y_score = torch.cat([r[1] for r in records], dim=0)

        metric_names = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_macro"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metric_names, is_binary=False)
        for key, value in results.items():
            self.log(f'{split}_{key}', value, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters()) +
            list(self.linear_probe1.parameters()) +
            list(self.linear_probe2.parameters()),
            weight_decay=self.hparams.weight_decay
        )
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.hparams.lr, 
            steps_per_epoch=self.hparams.steps_per_epoch, 
            epochs=self.hparams.epochs, 
            pct_start=0.2
        )
        
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'valid_balanced_accuracy',
            'strict': True,
            'name': None,
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_dict}


# ============================================================================
# Main Script Execution
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--out_dir', default='./checkpoints/EEGPT_TUEV_15ch', type=str)
    parser.add_argument('--devices', default=1, type=int, help="Number of GPUs to use")
    parser.add_argument('--data_root', default='/homes/yy2541/Henry_shine2/TUEV/processed', type=str)
    parser.add_argument('--ckpt_path', default='/homes/yy2541/Henry_shine3/EEGPT/checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt', type=str)
    args = parser.parse_args()

    seeds = [42, 3407, 6, 16, 66]
    aggregate_results = {'balanced_accuracy': [], 'f1_score': [], 'cohen_kappa': []}

    print("Loading TUEV dataset...")
    train_dataset, val_dataset, test_dataset = prepare_TUEV_dataset(args.data_root, target_length=1024)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    valid_loader = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    steps_per_epoch = math.ceil(len(train_loader))

    for seed in seeds:
        print(f"\n{'='*50}\nStarting Fine-tuning Run with Random Seed: {seed}\n{'='*50}")
        seed_torch(seed)
        
        model = LitEEGPTCausal(
            lr=args.lr, 
            weight_decay=args.weight_decay, 
            epochs=args.epochs, 
            steps_per_epoch=steps_per_epoch,
            num_classes=6,
            load_path=args.ckpt_path
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.out_dir, f'seed_{seed}'),
            filename='best-{epoch:02d}-{valid_balanced_accuracy:.4f}',
            monitor='valid_balanced_accuracy',
            mode='max',
            save_top_k=1,
            save_last=False,
            verbose=True,
        )

        trainer = pl.Trainer(
            accelerator='cuda',
            devices=args.devices,
            strategy=DDPStrategy(find_unused_parameters=True) if args.devices > 1 else 'auto',
            precision='16-mixed',
            max_epochs=args.epochs,
            callbacks=[lr_monitor, checkpoint_callback],
            logger=[
                pl_loggers.TensorBoardLogger('./logs/', name=f"EEGPT_TUEV_seed{seed}_tb", version="v1"),
            ]
        )

        trainer.fit(model, train_loader, valid_loader)
        test_results = trainer.test(model, dataloaders=test_loader, ckpt_path='best')
        
        if test_results:
            res = test_results[0]
            aggregate_results['balanced_accuracy'].append(res.get('test_balanced_accuracy', 0))
            aggregate_results['f1_score'].append(res.get('test_f1', 0))
            aggregate_results['cohen_kappa'].append(res.get('test_cohen_kappa', 0))

    print("\n\n" + "="*55)
    print("FINAL TEST PERFORMANCE ACROSS 5 SEEDS (Best Val Checkpoints)")
    print("="*55)
    
    bacc_mean, bacc_std = np.mean(aggregate_results['balanced_accuracy']), np.std(aggregate_results['balanced_accuracy'])
    f1_mean, f1_std = np.mean(aggregate_results['f1_score']), np.std(aggregate_results['f1_score'])
    kappa_mean, kappa_std = np.mean(aggregate_results['cohen_kappa']), np.std(aggregate_results['cohen_kappa'])

    print(f"Balanced Accuracy: {bacc_mean:.4f} ± {bacc_std:.4f}")
    print(f"F1 Score:          {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Cohen's Kappa:     {kappa_mean:.4f} ± {kappa_std:.4f}")