import argparse
import math
import numpy as np
import random
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from functools import partial
from scipy.signal import resample

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

# Import CHANNEL_DICT to dynamically filter supported channels
from Modules.models.EEGPT_mcae import EEGTransformer, CHANNEL_DICT
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
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

class SeedIVLoader(Dataset):
    def __init__(self, subject_ids, keep_indices, data_dir="/homes/xw2336/data_portal/data/SEED_IV/", target_length=1024):
        self.subject_ids = subject_ids
        self.keep_indices = keep_indices
        self.data_dir = data_dir
        self.target_length = target_length
        self.X = []
        self.y = []

        for subj in subject_ids:
            for sess in range(1, 4):
                file_path = os.path.join(self.data_dir, f"subject_{subj}_{sess}.npy")
                if os.path.exists(file_path):
                    payload = np.load(file_path, allow_pickle=True).item()
                    self.X.append(payload['X'])
                    self.y.append(payload['y'])
                else:
                    print(f"⚠️ Warning: Missing file for Subject {subj}, Session {sess}")
        
        if len(self.X) > 0:
            self.X = np.concatenate(self.X, axis=0)
            self.y = np.concatenate(self.y, axis=0).astype(np.int64)
            
            self.X = np.nan_to_num(self.X, nan=0.0, posinf=800.0, neginf=-800.0)
            self.X = np.clip(self.X, -800.0, 800.0)
            
            mean = np.mean(self.X, axis=2, keepdims=True)
            std = np.std(self.X, axis=2, keepdims=True)
            self.X = (self.X - mean) / (std + 1e-8)
        else:
            raise ValueError("No data found for the provided SEED-IV subjects.")
        
        raw_min = np.min(self.y)
        if raw_min != 0:
            self.y = self.y - raw_min

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # Slice the channels to strictly match EEGPT's 58-channel dictionary
        x = x[self.keep_indices, :]

        if x.dtype.byteorder == '>' or (x.dtype.byteorder == '|' and not x.dtype.str.endswith('f4')):
            x = x.byteswap().newbyteorder('=')
        
        if x.shape[-1] != self.target_length:
            x = resample(x, self.target_length, axis=-1)

        x = np.ascontiguousarray(x, dtype=np.float32)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

class LitEEGPTCausal(pl.LightningModule):
    def __init__(self, use_channels_names, lr=4e-4, weight_decay=0.01, epochs=50, steps_per_epoch=100, load_path=""):
        super().__init__()
        self.save_hyperparameters()
        
        self.chans_num = len(use_channels_names)

        target_encoder = EEGTransformer(
            img_size=[self.chans_num, 1024],
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
        self.linear_probe2 = LinearWithConstraint(16*16, 4, max_norm=0.25)
       
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
        
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
        return loss
        
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

        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
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

        self.running_scores["test"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
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
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1, 'monitor': 'valid_balanced_accuracy', 'strict': True}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--out_dir', default='./checkpoints/EEGPT_SEED_IV', type=str)
    parser.add_argument('--data_dir', default='/homes/xw2336/data_portal/data/SEED_IV/', type=str)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--ckpt_path', default='/homes/yy2541/Henry_shine3/EEGPT/checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt', type=str)
    args = parser.parse_args()

    seed_torch(42)

    # 1. Define ALL 62 SEED-IV channels natively
    SEED_IV_ALL_CHANS = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 
        'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 
        'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 
        'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ]
    
    # 2. Dynamically extract only the channels that EEGPT natively supports in its dictionary
    use_channels_names = [ch for ch in SEED_IV_ALL_CHANS if ch in CHANNEL_DICT]
    keep_indices = [i for i, ch in enumerate(SEED_IV_ALL_CHANS) if ch in CHANNEL_DICT]
    
    print(f"Dynamically mapped {len(use_channels_names)} supported channels out of 62.")

    aggregate_results = {'balanced_accuracy': [], 'f1_score': [], 'cohen_kappa': []}
    all_subjects = list(range(1, 16))

    print(f"{'='*50}\nStarting SEED-IV LOSO Cross-Validation (Seed 42)\n{'='*50}")

    for test_sub in all_subjects:
        train_subjs = [s for s in all_subjects if s != test_sub]
        
        print(f"\nTraining on {len(train_subjs)} subjects, Evaluating on Subject: {test_sub}")

        # Pass keep_indices to filter arrays during __getitem__
        train_dataset = SeedIVLoader(subject_ids=train_subjs, keep_indices=keep_indices, data_dir=args.data_dir)
        valid_dataset = SeedIVLoader(subject_ids=[test_sub], keep_indices=keep_indices, data_dir=args.data_dir)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader  = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        steps_per_epoch = math.ceil(len(train_loader))
        
        # Initialize model with the dynamically filtered channel list
        model = LitEEGPTCausal(
            use_channels_names=use_channels_names,
            lr=args.lr, weight_decay=args.weight_decay, 
            epochs=args.epochs, steps_per_epoch=steps_per_epoch,
            load_path=args.ckpt_path
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.out_dir, f'subject_{test_sub:02d}'),
            filename='best-{epoch:02d}-{valid_balanced_accuracy:.4f}',
            monitor='valid_balanced_accuracy',
            mode='max',
            save_top_k=1,
            save_last=False,
            verbose=True
        )
        
        trainer = pl.Trainer(
            accelerator='cuda',
            devices=args.devices,
            strategy=DDPStrategy(find_unused_parameters=True) if args.devices > 1 else 'auto',
            precision='16-mixed',
            max_epochs=args.epochs,
            callbacks=[LearningRateMonitor(logging_interval='epoch'), checkpoint_callback],
            enable_checkpointing=True,
            logger=[pl_loggers.TensorBoardLogger('./log/', name="EEGPT_SEED_IV_tb", version=f"subject{test_sub}")],
        )
        
        trainer.fit(model, train_loader, valid_loader)
        test_results = trainer.test(model, dataloaders=test_loader, ckpt_path='best')
        
        if test_results:
            res = test_results[0]
            aggregate_results['balanced_accuracy'].append(res.get('test_balanced_accuracy', 0))
            aggregate_results['f1_score'].append(res.get('test_f1_macro', 0))
            aggregate_results['cohen_kappa'].append(res.get('test_cohen_kappa', 0))

    print("\n\n" + "="*55)
    print("FINAL TEST PERFORMANCE ACROSS 15 SUBJECTS (LOSO CV)")
    print("="*55)
    
    bacc_mean, bacc_std = np.mean(aggregate_results['balanced_accuracy']), np.std(aggregate_results['balanced_accuracy'])
    f1_mean, f1_std = np.mean(aggregate_results['f1_score']), np.std(aggregate_results['f1_score'])
    kappa_mean, kappa_std = np.mean(aggregate_results['cohen_kappa']), np.std(aggregate_results['cohen_kappa'])

    print(f"Balanced Accuracy: {bacc_mean:.4f} ± {bacc_std:.4f}")
    print(f"F1 Score:          {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Cohen's Kappa:     {kappa_mean:.4f} ± {kappa_std:.4f}")