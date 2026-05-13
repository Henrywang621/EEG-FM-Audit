#!/usr/bin/env python3
import torch
import os
import sys
import gc  # Added for manual garbage collection

# Set environment variables to manage CUDA memory fragmentation BEFORE torch loads
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure imports work from your source directory
from train_GPT_THU import get_config, train

def run_ablation_pretraining():
    """
    Main execution for No-GPT ablation pretraining with memory management.
    Matches the architecture to 3 randomly initialized attention layers.
    """
    # 1. Load the standard pretraining config
    config = get_config()
    
    # 2. Apply No-GPT Ablation Constraints
    config["architecture"] = "GPT"        
    config["num_hidden_layers"] = 3       
    config["training_style"] = "CSM_causal" 
    config["pretrained_model"] = None     
    config["run_name"] = "NeuroGPT_Ablation_NoGPT_3L"
    
    # Update log directory for the ablation run
    config["log_dir"] = os.path.join("results/pretraining/", config["run_name"])
    os.makedirs(config["log_dir"], exist_ok=True)

    # Pre-training memory flush
    gc.collect()
    torch.cuda.empty_cache()

    try:
        # 3. Execute pretraining using the original trainer logic
        print(f"🚀 Starting ablation pretraining: {config['run_name']}")
        trainer = train(config)
        
        # 4. Save weights for the finetuning script
        save_path = os.path.join(config["log_dir"], 'model_no_gpt_3l')
        os.makedirs(save_path, exist_ok=True)
        
        # Explicitly move to CPU before saving to prevent VRAM spikes
        state_dict = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
        torch.save(state_dict, os.path.join(save_path, 'full_weights.pth'))
        
        print(f"✅ Ablation pretraining complete.")
        print(f"✅ Weights saved: {save_path}/full_weights.pth")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ CUDA Out of Memory. Clearing cache...")
            torch.cuda.empty_cache()
        raise e
    finally:
        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    run_ablation_pretraining()