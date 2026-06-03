# Training in 256Hz data and 4s
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from engine_pretraining import *
from configs_BCI2B import *
from dataloader import *

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
    dirpath=f'./checkpoints/EEGPT_{tag}_{variant}/',
    filename='best-{epoch:02d}-{train_loss:.4f}',
    monitor='train_loss',  # Monitor training loss
    mode='min',            # Save when training loss is minimized
    save_top_k=1,          # Keep only the best model
    save_last=False,       # Don't save last epoch separately
    verbose=True
)

callbacks = [lr_monitor, checkpoint_callback]

trainer = pl.Trainer(
    strategy='auto', 
    devices=devices, 
    max_epochs=max_epochs, 
    callbacks=callbacks,
    logger=[
        pl_loggers.TensorBoardLogger('./logs/', name=f"EEGPT_{tag}_{variant}_tb"),
        pl_loggers.CSVLogger('./logs/', name=f"EEGPT_{tag}_{variant}_csv")
    ]
)

trainer.fit(model, train_loader, val_loader)
