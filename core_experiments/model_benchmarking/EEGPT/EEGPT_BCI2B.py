import argparse
import math
import numpy as np
import random
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from torch.utils.data import Dataset
from scipy.signal import resample

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

# from dataloader import *
from Modules.models.EEGPT_mcae import EEGTransformer
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

def sliding_window_eeg(data, window_index=8, window_size=500, step_size=100):
    """
    data shape: (N, Channels, Total_Time) -> (N, 3, 1500)
    Extracts a specific window slice of length window_size.
    """
    start = window_index * step_size
    end = start + window_size
    return data[:, :, start:end]


class BCICIV2bLoader(Dataset):
    def __init__(
        self, 
        subject_ids, 
        data_dir="/homes/yy2541/Henry_shine2/data/BCICIV_2b",
        window_index=8, 
        window_size=500, 
        step_size=100, 
        target_length=1024
    ):
        self.subject_ids = subject_ids
        self.data_dir = data_dir
        self.window_index = window_index
        self.window_size = window_size
        self.step_size = step_size
        self.target_length = target_length

        self.X, self.y = self.load_subject_data(self.subject_ids)

    def load_subject_data(self, subject_ids):
        subj_mapping = {
            'S01': 1, 'S02': 2, 'S03': 3, 'S04': 4, 'S05': 5,
            'S06': 6, 'S07': 7, 'S08': 8, 'S09': 9
        }

        data_list = []
        label_list = []

        for sid in subject_ids:
            s_num = subj_mapping[sid] if isinstance(sid, str) and sid in subj_mapping else sid

            f_x1 = os.path.join(self.data_dir, f"Subj{s_num}_1_X.npy")
            f_x2 = os.path.join(self.data_dir, f"Subj{s_num}_2_X.npy")
            f_y1 = os.path.join(self.data_dir, f"Subj{s_num}_1_y.npy")
            f_y2 = os.path.join(self.data_dir, f"Subj{s_num}_2_y.npy")

            x1, x2 = np.load(f_x1), np.load(f_x2)
            y1, y2 = np.load(f_y1), np.load(f_y2)

            data_list.append(np.concatenate([x1, x2], axis=0))
            label_list.append(np.concatenate([y1, y2], axis=0))

        X_all = np.concatenate(data_list, axis=0)
        y_all = np.concatenate(label_list, axis=0)

        # Slice the 500-sample temporal window (e.g. 800:1300)
        x_seg = sliding_window_eeg(X_all, self.window_index, self.window_size, self.step_size)

        # Ensure label array format
        y_all = y_all.astype(np.int64).squeeze()

        return x_seg, y_all

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # Resample from 500 samples to 1024 samples for EEGPT
        if x.shape[-1] != self.target_length:
            x = resample(x, self.target_length, axis=-1)

        x = np.ascontiguousarray(x, dtype=np.float32)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

class LitEEGPTCausal(pl.LightningModule):
    def __init__(self, lr=4e-4, weight_decay=0.01, epochs=50, steps_per_epoch=100, load_path=""):
        super().__init__()
        self.save_hyperparameters()
        self.chans_num = 3
        use_channels_names = ['C3', 'CZ', 'C4']

        # Init model
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
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
            
        self.target_encoder = target_encoder
        self.chans_id = target_encoder.prepare_chan_ids(use_channels_names)
        
        pretrain_ckpt = torch.load(load_path, weights_only=False)
        target_encoder_stat = {
            k[15:]: v for k, v in pretrain_ckpt['state_dict'].items() if k.startswith("target_encoder.")
        }
        self.target_encoder.load_state_dict(target_encoder_stat)

        self.chan_conv = Conv1dWithConstraint(3, self.chans_num, 1, max_norm=1)
        self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(16*16, 2, max_norm=0.25)
       
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
        y_score = torch.softmax(logit, dim=-1)[:, 1]

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
        y_score = torch.softmax(logit, dim=-1)[:, 1]

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

        # Binary metrics for BCIIV2b Left vs Right Hand
        metric_names = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metric_names, is_binary=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--out_dir', default='./checkpoints/EEGPT_BCIC2B', type=str)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--ckpt_path', default='/homes/yy2541/Henry_shine3/EEGPT/checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt', type=str)
    args = parser.parse_args()

    # Apply Single Seed setup for LOSO Evaluation
    seed_torch(42)

    aggregate_results = {'balanced_accuracy': [], 'f1_score': [], 'cohen_kappa': []}
    subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']

    print(f"{'='*50}\nStarting BCIIV2b LOSO Cross-Validation (Seed 42)\n{'='*50}")

    for i in range(1, 10):
        current_subj = f'S{i:02d}'
        train_subjs = [s for s in subjects if s != current_subj]
        
        print(f"\nTraining on {len(train_subjs)} subjects, Evaluating on Subject: {current_subj}")

        train_dataset = BCICIV2bLoader(subject_ids=train_subjs, window_index=8, window_size=500, step_size=100)
        valid_dataset = BCICIV2bLoader(subject_ids=[current_subj], window_index=8, window_size=500, step_size=100)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # We use the valid_dataset for both validation and testing (LOSO)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader  = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        
        steps_per_epoch = math.ceil(len(train_loader))
        
        model = LitEEGPTCausal(
            lr=args.lr, 
            weight_decay=args.weight_decay, 
            epochs=args.epochs, 
            steps_per_epoch=steps_per_epoch,
            load_path=args.ckpt_path
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Save only the best checkpoint, disable saving last to save storage
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.out_dir, f'subject_{i:02d}'),
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
            callbacks=[lr_monitor, checkpoint_callback],
            enable_checkpointing=True,
            logger=[
                pl_loggers.TensorBoardLogger('./log/', name="EEGPT_BCIC2B_tb", version=f"subject{i}"),
            ]
        )
        
        # Train on the 8 subjects, validate on the left-out subject
        trainer.fit(model, train_loader, valid_loader)
        
        # Test the best checkpoint on the left-out subject
        test_results = trainer.test(model, dataloaders=test_loader, ckpt_path='best')
        
        if test_results:
            res = test_results[0]
            aggregate_results['balanced_accuracy'].append(res.get('test_balanced_accuracy', 0))
            aggregate_results['f1_score'].append(res.get('test_f1', 0))
            aggregate_results['cohen_kappa'].append(res.get('test_cohen_kappa', 0))

    # Calculate final LOSO performance metrics
    print("\n\n" + "="*55)
    print("FINAL TEST PERFORMANCE ACROSS 9 SUBJECTS (LOSO CV)")
    print("="*55)
    
    bacc_mean, bacc_std = np.mean(aggregate_results['balanced_accuracy']), np.std(aggregate_results['balanced_accuracy'])
    f1_mean, f1_std = np.mean(aggregate_results['f1_score']), np.std(aggregate_results['f1_score'])
    kappa_mean, kappa_std = np.mean(aggregate_results['cohen_kappa']), np.std(aggregate_results['cohen_kappa'])

    print(f"Balanced Accuracy: {bacc_mean:.4f} ± {bacc_std:.4f}")
    print(f"F1 Score:          {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Cohen's Kappa:     {kappa_mean:.4f} ± {kappa_std:.4f}")