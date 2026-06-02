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
        
        # Logging to TensorBoard
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
        
        # Logging to TensorBoard
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
        
        # Logging to TensorBoard
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

def apply_spatial_preserving_phase_rand(x):
    batch_size, num_channels, time_len = x.shape
    mu_x = x.mean(dim=-1, keepdim=True)
    y = x - mu_x
    Y = torch.fft.rfft(y, dim=-1)
    magnitudes = torch.abs(Y)
    random_phases = torch.rand(batch_size, 1, Y.shape[-1], device=x.device) * 2 * torch.pi
    random_phases[..., 0] = 0.0
    Y_surrogate = magnitudes * torch.exp(1j * random_phases)
    return torch.fft.irfft(Y_surrogate, n=time_len, dim=-1) + mu_x

# ============================================================================
# Training Script
# ============================================================================
if __name__ == "__main__":
    import math
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, roc_auc_score
    import numpy as np
    
    seed_torch(8)
    
    # Hyperparameters
    batch_size = 64
    
    # Data path
    data_root = "/mnt/scratch2/users/xw2336/LLM_Eva/Data/TUAB/v3.0.1/edf/processed"
    
    # Load test dataset with downsampling to 1024
    print("Loading TUAB test dataset...")
    _, _, test_dataset = prepare_TUAB_dataset(data_root, target_length=1024)
    
    # Create test data loader
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
    
        # Collect predictions and labels
        all_preds = []
        all_labels = []
        all_probs = []
    
        n_augments = 10

        print("\nRunning inference on test set...")
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.cuda(), y.cuda()
        
                prob_accumulator = torch.zeros(x.shape[0], 2, device=x.device)
        
                for _ in range(n_augments):
                    x_aug = apply_spatial_preserving_phase_rand(x)
                    output = model(x_aug)
                    logits = output[1] if isinstance(output, tuple) else output
                    prob_accumulator += torch.softmax(logits, dim=-1)
        
                avg_probs = prob_accumulator / n_augments
                preds = torch.argmax(avg_probs, dim=1)
        
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_probs.append(avg_probs.cpu().numpy())
    
        # Concatenate all predictions and labels
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
    
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        kappa = cohen_kappa_score(all_labels, all_preds)
        try:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])  # Binary: use prob of positive class
        except ValueError:
            auroc = float('nan')
        cm = confusion_matrix(all_labels, all_preds)
    
        # Calculate per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
        # Print results
        print("\n" + "="*60)
        print("Baseline TEST RESULTS - TUAB DATASET")
        print("="*60)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Total Test Samples: {len(all_labels)}")
        print("-"*60)
        print(f"Overall Accuracy:        {accuracy*100:.2f}%")
        print(f"Balanced Accuracy:       {balanced_acc*100:.2f}%")
        print(f"F1 Score (macro):        {f1:.4f}")
        print(f"Cohen's Kappa:           {kappa:.4f}")
        print(f"AUC-ROC:                 {auroc:.4f}")
        print("-"*60)
        print("\nPer-Class Accuracy:")
        class_names = ['Normal', 'Abnormal']  # Adjust based on your dataset
        for i in range(len(per_class_acc)):
            class_name = class_names[i] if i < len(class_names) else f'Class {i}'
            n_samples = cm.sum(axis=1)[i]
            print(f"  {class_name:15s}: {per_class_acc[i]*100:.2f}% ({n_samples} samples)")
        print("="*60)
    
        # Optional: Print confusion matrix
        print("\nConfusion Matrix:")
        print(cm)
        print("="*60 + "\n")


