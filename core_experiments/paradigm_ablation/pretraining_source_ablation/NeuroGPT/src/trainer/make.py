#!/usr/bin/env python3

import os
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, balanced_accuracy_score, roc_auc_score
import torch
from transformers import TrainingArguments,TrainerCallback
from trainer.base import Trainer
from scipy.special import softmax


class CSVLogCallback(TrainerCallback):

    def __init__(self):
        super().__init__()
        self.train_log_filepath = None
        self.eval_log_filepath = None
        
    def on_log(
        self,
        args,
        state,
        control,
        model,
        **kwargs
        ) -> None:

        if args.local_rank not in {-1, 0}:
            return

        if self.train_log_filepath is None:
            self.train_log_filepath = os.path.join(
                args.output_dir,
                'train_history.csv'
            )

            with open(self.train_log_filepath, 'a') as f:
                f.write('step,loss,lr\n')

        if self.eval_log_filepath is None:
            self.eval_log_filepath = os.path.join(
                args.output_dir,
                'eval_history.csv'
            )

            with open(self.eval_log_filepath, 'a') as f:
                f.write('step,loss,accuracy\n')

        is_eval = any('eval' in k for k in state.log_history[-1].keys())

        if is_eval:
            with open(self.eval_log_filepath, 'a') as f:
                f.write('{},{},{}\n'.format(
                        state.global_step,
                        state.log_history[-1]['eval_loss'],
                        state.log_history[-1]['eval_accuracy'] if 'eval_accuracy' in state.log_history[-1] else np.nan
                    )
                )

        else:

            with open(self.train_log_filepath, 'a') as f:
                f.write('{},{},{}\n'.format(
                        state.global_step,
                        state.log_history[-1]['loss'] if 'loss' in state.log_history[-1] else state.log_history[-1]['train_loss'],
                        state.log_history[-1]['learning_rate'] if 'learning_rate' in state.log_history[-1] else None
                    )
                )


def _cat_data_collator(features: List) -> Dict[str, torch.tensor]:

    if not isinstance(features[0], dict):
        features = [vars(f) for f in features] 

    return {
        k: torch.cat(
            [
                f[k]
                for f in features
            ]
        )
        for k in features[0].keys()
        if not k.startswith('__')
    }


# def decoding_accuracy_metrics(eval_preds):
#     preds, labels = eval_preds
#     preds = preds.argmax(axis=-1)
#     accuracy = accuracy_score(labels, preds)
#     return {
#         "accuracy": round(accuracy, 3)
#     }

# def decoding_metrics(eval_preds):
#     preds, labels = eval_preds
    
#     # 1. Determine if Binary or Multi-class
#     if preds.ndim > 1 and preds.shape[1] > 1:
#         # Multi-class case
#         preds_labels = np.argmax(preds, axis=-1)
#         probs = softmax(preds, axis=-1)
#         is_binary = False
#     else:
#         # Binary case
#         preds_labels = (preds > 0.5).astype(int)
#         probs = preds
#         is_binary = True

#     metrics = {}
#     metrics["accuracy"] = accuracy_score(labels, preds_labels)
#     # Use binary average for binary tasks
#     metrics["f1"] = f1_score(labels, preds_labels, average="binary" if is_binary else "weighted")
#     metrics["kappa"] = cohen_kappa_score(labels, preds_labels)
#     metrics["balanced_acc"] = balanced_accuracy_score(labels, preds_labels)

#     # 2. Safe ROC-AUC Calculation
#     # Check if we have both classes present to avoid the "Only one class" error
#     if len(np.unique(labels)) < 2:
#         metrics["roc_auc"] = float("nan")
#     else:
#         try:
#             if is_binary:
#                 # Binary: No multi_class arg, expects 1D probs for positive class
#                 metrics["roc_auc"] = roc_auc_score(labels, probs)
#             else:
#                 # Multi-class: Needs multi_class arg
#                 metrics["roc_auc"] = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
#         except Exception as e:
#             print(f"ROC-AUC Error: {e}") # Print error to debug
#             metrics["roc_auc"] = float("nan")

#     # Rounding
#     metrics = {k: round(v, 3) if not np.isnan(v) else v for k, v in metrics.items()}
#     return metrics


def decoding_metrics(eval_preds):
    preds, labels = eval_preds
    
    # --- FIX SHAPES START ---
    
    # 1. Fix Labels: [Batch, 1] -> [Batch]
    # This prevents sklearn from thinking it's a "multilabel" task
    labels = np.array(labels).reshape(-1)
    
    # 2. Fix Preds: [Batch, 2, 1] -> [Batch, 2]
    # We remove the useless last dimension
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
        
    # --- FIX SHAPES END ---

    # 3. Generate Predictions
    # Since shape is [Batch, 2], we use Argmax to pick Class 0 or Class 1
    preds_labels = np.argmax(preds, axis=-1)
    
    # Generate Probabilities (Apply Softmax to get 0.0-1.0 range)
    probs_full = softmax(preds, axis=-1)
    
    # For Binary AUC, we usually want the probability of the POSITIVE class (Index 1)
    probs_positive = probs_full[:, 1] 

    # 4. Calculate Metrics
    metrics = {}
    metrics["accuracy"] = accuracy_score(labels, preds_labels)
    
    # Note: Even though preds has 2 columns, this is a Binary Task.
    # We use average="binary" to focus on the positive class performance.
    metrics["f1"] = f1_score(labels, preds_labels, average="macro")
    metrics["kappa"] = cohen_kappa_score(labels, preds_labels)
    metrics["balanced_acc"] = balanced_accuracy_score(labels, preds_labels)

    # # 5. Safe ROC-AUC Calculation
    # if len(np.unique(labels)) < 2:
    #     # If batch only has one class (all 0s or all 1s), AUC crashes
    #     metrics["roc_auc"] = float("nan")
    # else:
    #     try:
    #         # We pass the probabilities of Class 1 against the labels
    #         metrics["roc_auc"] = roc_auc_score(labels, probs_positive)
    #     except Exception as e:
    #         print(f"ROC-AUC Error: {e}")
    #         metrics["roc_auc"] = float("nan")

    n_classes = len(np.unique(labels))

    if n_classes < 2:
        metrics["roc_auc"] = float("nan")
    else:
        try:
            if n_classes == 2:
                metrics["roc_auc"] = roc_auc_score(labels, probs_positive)
            else:
                metrics["roc_auc"] = roc_auc_score(
                    labels,
                    probs_full,                  # (N, C)
                    multi_class="ovr",
                    average="macro"
                )
        except Exception as e:
            print(f"ROC-AUC Error: {e}")
            metrics["roc_auc"] = float("nan")

    # Rounding
    metrics = {k: round(v, 3) if not np.isnan(v) else v for k, v in metrics.items()}
    return metrics


def make_trainer(
    model_init,
    training_style,
    train_dataset,
    validation_dataset,
    do_train: bool = True,
    do_eval: bool = True,
    run_name: str = None,
    output_dir: str = None,
    overwrite_output_dir: bool = True,
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    optim: str='adamw_hf',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    adam_beta1: float=0.9,
    adam_beta2: float=0.999,
    adam_epsilon: float=1e-8,
    max_grad_norm: float=1.0,
    per_device_train_batch_size: int = 64,
    per_device_eval_batch_size: int = 64,
    dataloader_num_workers: int = 0,
    max_steps: int = 400000,
    num_train_epochs: int = 100,
    lr_scheduler_type: str = 'linear',
    warmup_ratio: float = 0.01,
    evaluation_strategy: str = 'steps',
    prediction_loss_only: bool = False,
    logging_strategy: str = 'steps',
    save_strategy: str = 'steps',
    save_total_limit: int = 5,
    save_steps: int = 10000,
    logging_steps: int = 10000,
    eval_steps: int = None,
    logging_first_step: bool = True,
    greater_is_better: bool = True,
    seed: int = 1,
    fp16: bool = True,
    deepspeed: str = None,
    compute_metrics = None,
    **kwargs
    ) -> Trainer:
    """
    Make a Trainer object for training a model.
    Returns an instance of transformers.Trainer.
    
    See the HuggingFace transformers documentation for more details
    on input arguments:
    https://huggingface.co/transformers/main_classes/trainer.html

    Custom arguments:
    ---
    model_init: callable
        A callable that does not require any arguments and 
        returns model that is to be trained (see scripts.train.model_init)
    training_style: str
        The training style (ie., framework) to use.
        One of: 'BERT', 'CSM', 'NetBERT', 'autoencoder',
        'decoding'.
    train_dataset: src.batcher.dataset
        The training dataset, as generated by src.batcher.dataset
    validation_dataset: src.batcher.dataset
        The validation dataset, as generated by src.batcher.dataset

    Returns
    ----
    trainer: transformers.Trainer
    """

    # TrainingArguments is a configuration class from the transformers library that stores all the settings for 
    trainer_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        do_train=do_train,
        do_eval=do_eval,
        overwrite_output_dir=overwrite_output_dir,
        prediction_loss_only=prediction_loss_only,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        eval_accumulation_steps=1,
        optim="adamw_torch",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        lr_scheduler_type=lr_scheduler_type,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        greater_is_better=greater_is_better,
        save_steps=save_steps,
        logging_strategy=logging_strategy,
        logging_first_step=logging_first_step,
        logging_steps=logging_steps,
        # evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps if eval_steps is not None else logging_steps,
        seed=seed,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        deepspeed=deepspeed,
        **kwargs
    )

    data_collator = _cat_data_collator
    is_deepspeed = deepspeed is not None
    # TODO: custom compute_metrics so far not working in multi-gpu setting
    # compute_metrics = decoding_accuracy_metrics if training_style=='decoding' and compute_metrics is None else compute_metrics

    compute_metrics = decoding_metrics if training_style=='decoding' and compute_metrics is None else compute_metrics
    trainer = Trainer(
        args=trainer_args,
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
        is_deepspeed=is_deepspeed
    )

    trainer.add_callback(CSVLogCallback)

    return trainer