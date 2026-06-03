import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from engine_pretraining_Tuav_15ch import *
from configs_Tuev import *
import pickle
import os
from scipy.signal import resample
import numpy as np
torch.set_float32_matmul_precision("medium")

seed_torch(7)


# Init model
model = LitEEGPT(get_config(**(MODELS_CONFIGS[tag])),
                 USE_LOSS_A=(variant != "A"),
                 USE_LN=(variant != "B"),
                 USE_SKIP=(variant != "C"))

# Callbacks
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

# ModelCheckpoint to save ONLY the best model based on training loss
checkpoint_callback = ModelCheckpoint(
    dirpath=f'./checkpoints/EEGPT_{tag}_{variant}_drop15/',
    filename='best-{epoch:02d}-{train_loss:.4f}',
    monitor='train_loss',
    mode='min',
    save_top_k=1,
    save_last=False,
    verbose=True
)

callbacks = [lr_monitor, checkpoint_callback]

trainer = pl.Trainer(
    strategy='auto',
    devices=devices,
    max_epochs=max_epochs,
    callbacks=callbacks,
    logger=[
        pl_loggers.TensorBoardLogger('./logs/', name=f"EEGPT_{tag}_{variant}_drop15_tb"),
        pl_loggers.CSVLogger('./logs/', name=f"EEGPT_{tag}_{variant}_drop15_csv")
    ]
)

# Use combined dataset for both training and validation
trainer.fit(model, combined_loader, combined_valid_loader, ckpt_path='/users/yyang/EEGPT/EEGPT/pretrain/checkpoints/EEGPT_large_D_drop15/best-epoch=224-train_loss=0.1395.ckpt')
