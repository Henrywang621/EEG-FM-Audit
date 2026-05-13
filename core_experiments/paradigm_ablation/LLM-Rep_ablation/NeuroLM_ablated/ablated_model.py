import torch
import torch.nn as nn
from model.model_neurolm import NeuroLM

class SupervisedNeuroLM(nn.Module):
    """
    Supervised NeuroLM for LLM-Rep Ablation.
    Replaces generative paradigm with a standard classification head and masked pooling.
    """
    def __init__(self, gpt_conf, num_classes=2):
        super().__init__()
        # Initialize from scratch to ablate foundation pre-training impact
        self.backbone = NeuroLM(gpt_conf, init_from='scratch')
        
        # Ensure all parameters are trainable for scratch-based supervised learning
        for p in self.backbone.parameters():
            p.requires_grad = True
        
        # Traditional Classification Head
        self.classifier = nn.Linear(gpt_conf.n_embd, num_classes)
        
        # Initialize classifier weights using the backbone's initialization logic
        self.classifier.apply(self.backbone._init_weights)

    def forward(self, x_eeg, chans, t_steps, eeg_mask):
        # 1. Tokenization and Projection
        # eeg_mask shape: [batch, seq_len]
        input_mask = eeg_mask.unsqueeze(1).repeat(1, x_eeg.size(1), 1).unsqueeze(1)
        x = self.backbone.tokenizer(x_eeg, chans, t_steps, input_mask, return_all_tokens=True)
        x = self.backbone.encode_transform_layer(x)
        x += self.backbone.pos_embed(chans)
        
        # 2. Transformer Backbone
        # Iterating through the GPT blocks
        for block in self.backbone.GPT2.transformer.h:
            x = block(x)
        x = self.backbone.GPT2.transformer.ln_f(x)
        
        # 3. Masked Global Average Pooling
        # mask shape: [B, T, 1]
        mask = eeg_mask.unsqueeze(-1).float() 
        masked_x = x * mask
        
        # Sum only the non-masked tokens and divide by the actual signal length
        sum_feat = torch.sum(masked_x, dim=1)
        count_feat = torch.clamp(mask.sum(dim=1), min=1e-9)
        feat = sum_feat / count_feat
        
        # 4. Final Logits for TUAB classification
        return self.classifier(feat)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return self.backbone.configure_optimizers(weight_decay, learning_rate, betas, device_type)