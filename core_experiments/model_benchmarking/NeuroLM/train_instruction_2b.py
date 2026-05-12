"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
Modified for NeurIPS DB Track Reproducibility
"""

import os
import time
import argparse
from contextlib import nullcontext
import gc
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model.model_neurolm import NeuroLM
from model.model import GPTConfig
from pathlib import Path
import tiktoken
from utils import (
    prepare_TUAB_dataset, prepare_CCD_dataset, prepare_TUEV_dataset, 
    prepare_TUSL_dataset, prepare_HMC_dataset, prepare_Workload_dataset, 
    cosine_scheduler, get_metrics, prepare_BCI2b_dataset
)
from downstream_dataset import SEEDDataset
from torch.utils.data.dataset import ConcatDataset

# Global variables for DDP and Device management
master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None
is_initialized = False 

def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank, is_initialized
    
    if is_initialized:
        seed_offset = ddp_rank if ddp else 0
        torch.manual_seed(args.seed + seed_offset)
        return

    backend = 'nccl'
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
    
    ddp = int(os.environ.get('RANK', -1)) != -1 
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 
        seed_offset = ddp_rank 
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    device_type = 'cuda' if 'cuda' in device else 'cpu' 
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    is_initialized = True

def get_instruct_datasets(args, downstream_dataset: str, eeg_max_len=-1, text_max_len=-1):
    dataset_info = {'name': downstream_dataset}
    
    if downstream_dataset == 'BCICIV2b':
        all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
        train_subs = [sub for sub in all_subjects if sub != args.test_subject]
        val_subs   = [args.test_subject]
        test_subs  = [args.test_subject] 

        dataset_train, dataset_test, dataset_val = prepare_BCI2b_dataset(
            root=Path(args.dataset_dir, 'BCICIV_2b'), 
            train_subjects=train_subs,
            val_subjects=val_subs,
            test_subjects=test_subs,
            is_instruct=True, 
            eeg_max_len=eeg_max_len, 
            text_max_len=text_max_len
        )

        dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "f1", "cohen_kappa"]
        dataset_info['is_binary'] = True
        dataset_info['result_idx'] = 7 
        dataset_info['label_dic'] = {'Yes': 1, 'No': 0}

    dataset_info['dataset_train'] = dataset_train
    dataset_info['dataset_val'] = dataset_val
    dataset_info['dataset_test'] = dataset_test

    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train, batch_size=args.eeg_batch_size,
            num_workers=3, pin_memory=True, drop_last=True,
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val, batch_size=int(args.eeg_batch_size * 1.5),
            num_workers=3, pin_memory=True, drop_last=False,
        )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test, batch_size=int(args.eeg_batch_size * 1.5),
            num_workers=3, pin_memory=True, drop_last=False,
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.eeg_batch_size, num_workers=3,
            pin_memory=True, drop_last=True, shuffle=True
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=int(args.eeg_batch_size * 1.5), num_workers=3,
            pin_memory=True, drop_last=False, shuffle=False
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=int(args.eeg_batch_size * 1.5), num_workers=3,
            pin_memory=True, drop_last=False, shuffle=False
        )
        
    dataset_info['data_loader_train'] = data_loader_train
    dataset_info['data_loader_val'] = data_loader_val
    dataset_info['data_loader_test'] = data_loader_test
    return dataset_info

def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, f'checkpoints/instruction-B_2b4/{args.test_subject}')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    def get_batch(split):
        if split == 'train':
            data = np.memmap(os.path.join('/homes/xw2336/xw2336/NeuroLM/text', 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join('/homes/xw2336/xw2336/NeuroLM/text', 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - args.block_size, (args.text_batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + args.block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    concat_datasets = True
    all_datasets = [get_instruct_datasets(args, args.dataset, eeg_max_len=276, text_max_len=80)]
    
    if concat_datasets:
        merge_datasets = ConcatDataset([dataset_info['dataset_train'] for dataset_info in all_datasets])
        if ddp:
            sampler_merge = torch.utils.data.DistributedSampler(
                merge_datasets, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
            )
            data_loader_merge = torch.utils.data.DataLoader(
                merge_datasets, sampler=sampler_merge, batch_size=args.eeg_batch_size,
                num_workers=3, pin_memory=True, drop_last=True
            )
        else:
            data_loader_merge = torch.utils.data.DataLoader(
                merge_datasets, batch_size=args.eeg_batch_size, num_workers=3,
                pin_memory=True, drop_last=True, shuffle=True
            )
            
    iter_num = 0
    init_from = 'resume' if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')) else 'pretrained'
    
    model_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=args.block_size, bias=False, vocab_size=50257, dropout=0.0) 
                    
    if init_from == 'resume':
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint['model_args'][k]
        model = NeuroLM(GPTConfig(**model_args), init_from='gpt2')
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
    else:
        ckpt_path = os.path.join(args.out_dir, args.NeuroLM_path)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint['model_args'][k]
        model = NeuroLM(GPTConfig(**model_args), init_from='scratch')
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        start_epoch = 0

    model.to(device)
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
    
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None 

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    if args.wandb_log and master_process:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(project=args.wandb_project, name=f"{args.wandb_runname}_{args.test_subject}", dir=os.path.join(args.out_dir, 'wandb'), reinit=True)

    num_training_steps_per_epoch = len(data_loader_merge) if concat_datasets else sum([len(d['data_loader_train']) for d in all_datasets])
    lr_schedule_values = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=int(args.warmup_ratio * num_training_steps_per_epoch * args.epochs)
    )

    enc = tiktoken.get_encoding("gpt2")
    decode = lambda l: enc.decode(l)
    
    datasets = [{'data_loader_train': data_loader_merge}] if concat_datasets else all_datasets
    X_text2, Y_text2 = get_batch('train') 
    raw_model = model.module if ddp else model 
    final_test_stats = None

    for epoch in range(start_epoch, args.epochs):
        if ddp and concat_datasets:
            data_loader_merge.sampler.set_epoch(epoch)
            
        for dataset_info in datasets:
            if args.eval_only: break
            for step, (batch) in enumerate(dataset_info['data_loader_train']):
                lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
                for param_group in optimizer.param_groups: param_group['lr'] = lr

                X_eeg, X_text, Y_text, input_chans, input_time, input_mask, gpt_mask = batch
                X_eeg, X_text, Y_text = X_eeg.float().to(device), X_text.to(device), Y_text.to(device)
                input_chans, input_time, gpt_mask = input_chans.to(device), input_time.to(device), gpt_mask.to(device)
                if input_mask is not None: input_mask = input_mask.to(device)

                Y_eeg = torch.full((X_eeg.size(0), X_eeg.size(1)), fill_value=-1-raw_model.GPT2.config.vocab_size).to(device)

                if ddp: model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0

                with ctx:
                    loss1, log1, _ = model(X_eeg, Y_eeg, X_text, Y_text, input_chans, input_time, input_mask, eeg_text_mask=gpt_mask)
                    loss2, log2, _ = model(None, None, X_text2, Y_text2)
                    loss = (loss1 + loss2) / args.gradient_accumulation_steps 
                    # loss = loss1 / args.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                X_text2, Y_text2 = get_batch('train')
                if (iter_num + 1) % args.log_interval == 0 and master_process:
                    print(f"Fold {args.test_subject} | Epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: train total loss {log1['train/loss'] + log2['train/loss']:.4f}")
                    # print(f"Fold {args.test_subject} | Epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: train total loss {log1['train/loss']:.4f}")
                iter_num += 1
        
        if not args.eval_only:
            for dataset_info in all_datasets:
                results_val = evaluate(raw_model, dataset_info, dataset_info['data_loader_val'], decode)
                results_test = evaluate(raw_model, dataset_info, dataset_info['data_loader_test'], decode)
                final_test_stats = results_test 
                if args.wandb_log and master_process:
                    log = {f'val_{dataset_info["name"]}/{k}': v for k, v in results_val.items() if isinstance(v, (int, float))}
                    log.update({f'test_{dataset_info["name"]}/{k}': v for k, v in results_test.items() if isinstance(v, (int, float))})
                    wandb.log(log)

    if master_process and not args.eval_only:
        checkpoint = {'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'model_args': model_args, 'iter_num': iter_num, 'epoch': args.epochs - 1}
        torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-final.pt'))

    if args.eval_only:
        for dataset_info in all_datasets:
            final_test_stats = evaluate(raw_model, dataset_info, dataset_info['data_loader_test'], decode)

    del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    if args.wandb_log and master_process: wandb.finish()
    return final_test_stats

def get_pred(pred_string, dataset_info):
    if dataset_info['name'] == 'BCICIV2b':
        s = pred_string.lower()
        if 'yes' in s: return 1 
        if 'no' in s: return 0  
        return -1 
    return -1

@torch.no_grad()
def evaluate(model, dataset_info, dataloader, decode):
    model.eval()
    preds, targets = [], []
    for _, batch in enumerate(dataloader):
        X_eeg, X_text, label, input_chans, input_time, input_mask, gpt_mask = batch
        X_eeg, X_text = X_eeg.float().to(device), X_text.to(device)
        input_chans, input_time, gpt_mask = input_chans.to(device), input_time.to(device), gpt_mask.to(device)
        if input_mask is not None: input_mask = input_mask.to(device)

        with ctx:
            # FIX: Put all the required variables back in, removing the '...'
            text = model.generate(X_eeg, X_text, input_chans, input_time, input_mask, eeg_text_mask=gpt_mask, max_new_tokens=5)
            
            for i, t in enumerate(text[:, 1:]):
                decoded_str = decode(t.tolist())
                
                # IMPORTANT DEBUG PRINT: See exactly what the model is saying
                if master_process and len(preds) < 10: # Just print the first 10 to not spam your console
                    print(f"DEBUG | Target Label: {label[i].item()} | Model Output: '{decoded_str}'")
                
                pred = get_pred(decoded_str, dataset_info)
                preds.append(pred if dataset_info['is_binary'] else np.eye(dataset_info['num_classes'])[pred])
            
            targets.append(label)
    
    model.train()
    return get_metrics(np.array(preds), torch.cat(targets, dim=0).cpu().numpy(), dataset_info['metrics'], dataset_info['is_binary'])

def get_args():
    parser = argparse.ArgumentParser('NeuroLM BCI Script', add_help=False)
    parser.add_argument('--dataset', default='BCICIV2b', type=str)
    parser.add_argument('--test_subject', default='S01', type=str)
    parser.add_argument('--out_dir', default='./')
    parser.add_argument('--dataset_dir', default='./')
    parser.add_argument('--tokenizer_path', default='checkpoints/VQ.py')
    parser.add_argument('--NeuroLM_path', default='checkpoints/NeuroLM-B.pt')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='NeuroLM')
    parser.add_argument('--wandb_runname', default='instruction-B')
    parser.add_argument('--wandb_api_key', type=str)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--eeg_batch_size', default=64, type=int)
    parser.add_argument('--text_batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--block_size', default=1024, type=int)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.dataset == 'BCICIV2b':
        all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
        all_fold_metrics = []
        is_main = int(os.environ.get('RANK', 0)) == 0
        
        for i, test_sub in enumerate(all_subjects):
            args.test_subject = test_sub
            if is_main:
                print(f"\n{'='*60}\n🚀 STRICT LOSO CV: FOLD {i+1} / 9 | Test: {test_sub}\n{'='*60}")
            
            fold_metrics = main(args)
            
            if fold_metrics and is_main:
                print(f"\n✅ RESULTS FOR TEST SUBJECT {test_sub}:")
                for key, val in fold_metrics.items():
                    # TYPE-SAFE PRINTING: Only use :.4f for numbers
                    if isinstance(val, (int, float, np.float32, np.float64)):
                        print(f"   {key.replace('_', ' ').title()}: {val:.4f}")
                    else:
                        print(f"   {key.replace('_', ' ').title()}:")
                        print(np.array(val))
                all_fold_metrics.append(fold_metrics)
                
        if is_main and all_fold_metrics:
            print("\n" + "🌟"*30 + "\nFINAL STRICT LOSO CV AVERAGES (9 FOLDS)\n" + "🌟"*30)
            for metric in ['accuracy', 'balanced_accuracy', 'f1', 'cohen_kappa']:
                values = [m.get(metric, 0) for m in all_fold_metrics if isinstance(m.get(metric), (int, float))]
                if values:
                    print(f"Average {metric.replace('_', ' ').title()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
            print("🌟"*30 + "\n")
            
        if int(os.environ.get('RANK', -1)) != -1: destroy_process_group()
    else:
        main(args)