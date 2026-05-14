#!/usr/bin/env python3 
import torch
from typing import Dict
import warnings
from safetensors.torch import load_file as load_safetensors
import os


class Model(torch.nn.Module):
    """
    Create Model object from embedder, decoder,
    and unembedder (if not None).

    Args
    ----
    embedder: src.embedder.make_embedder
        Instance of embedder class.
    decoder: src.decoder.make_decoder
        Instance of decoder class.
    unembedder: src.unembedder.make_unembedder
        Instance of unembedder class.
        Only added to model if not None.
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        embedder: torch.nn.Module,
        decoder: torch.nn.Module,
        unembedder: torch.nn.Module = None
        ) -> torch.nn.Module:
        
        super().__init__()
        self.name = f'Embedder-{embedder.name}_Decoder-{decoder.name}'
        self.encoder = encoder
        self.embedder = embedder
        self.decoder = decoder
        self.unembedder = unembedder
        self.is_decoding_mode = False
        self.ft_only_encoder = False

    def from_pretrained(
        self,
        pretrained_path: str
        ) -> None:
        """Load pretrained model from pretrained_path.
        Needs to point to pytorch_model.bin file.
        """
        print(
            f'Loading pretrained model from {pretrained_path}'
        )

        ext = os.path.splitext(pretrained_path)[1].lower()

        if ext == ".safetensors":
            pretrained = load_safetensors(pretrained_path)
        else:
            if next(self.parameters()).is_cuda:
                pretrained = torch.load(pretrained_path)
            else:
                pretrained = torch.load(pretrained_path, map_location="cpu")
        
        for k in self.state_dict():
            if k in pretrained:
                assert pretrained[k].shape == self.state_dict()[k].shape,\
                    f'{k} shape mismatch between pretrained model and current model '+\
                    f'{pretrained[k].shape} vs {self.state_dict()[k].shape}'
        
        for k in pretrained:     
            if k not in self.state_dict():
                warnings.warn(
                    f'Warning: /!\ Skipping {k} from {pretrained_path} '\
                    'because it is not part of the current model'
                )

        self.load_state_dict(pretrained, strict=False)
        
    def switch_ft_mode(self, ft_encoder_only=False):
        self.ft_only_encoder = ft_encoder_only

    def switch_decoding_mode(
        self,
        is_decoding_mode: bool = False,
        num_decoding_classes: int = None
        ) -> None:
        """Switch model to decoding model or back to training mode."""
        self.is_decoding_mode = is_decoding_mode
        
        self.embedder.switch_decoding_mode(is_decoding_mode=is_decoding_mode)
        self.decoder.switch_decoding_mode(
            is_decoding_mode=is_decoding_mode,
            num_decoding_classes=num_decoding_classes
        )

    def compute_loss(
        self,
        batch: Dict[str, torch.tensor],
        return_outputs: bool = False
        ) -> Dict[str, torch.tensor]:
        """
        Compute training loss, based on embedder's training-style.
        """
        (outputs, batch) = self.forward(
            batch=batch,
            return_batch=True
        )

        # =========================================================
        # FAILSAFE: PREVENT CUDA DEVICE-SIDE ASSERTS
        # Solves label out-of-bounds errors mathematically.
        # =========================================================
        if 'labels' in batch and 'decoding_logits' in outputs:
            labels = batch['labels'].long()
            num_classes = outputs['decoding_logits'].size(-1)
            
            # Identify labels that are not PyTorch's default padding index (-100)
            valid_mask = labels != -100
            if valid_mask.any():
                valid_labels = labels[valid_mask]
                
                # If labels are 1-based (e.g., 1 and 2), auto-shift them to 0 and 1
                if valid_labels.min() == 1:
                    labels[valid_mask] = valid_labels - 1
                
                # Hard clamp any remaining rogue values to strictly fit [0, num_classes - 1]
                labels[valid_mask] = torch.clamp(labels[valid_mask], min=0, max=num_classes - 1)
                
            batch['labels'] = labels

        losses = self.embedder.loss(
            batch=batch,
            outputs=outputs
        )

        return (losses, outputs) if return_outputs else losses

    def prep_batch(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict[str, torch.tensor]:
        """Prepare input batch for forward pass."""
        return self.embedder.prep_batch(batch=dict(batch))

    def forward(
        self,
        batch: Dict[str, torch.tensor],
        prep_batch: bool = True,
        return_batch: bool = False
        ) -> torch.tensor:
        """
        Forward pass of model.
        """
        
        if self.encoder is not None:
            # Let the splitted chunks of raw input through the encoder
            features = self.encoder(batch['inputs'])
            
            # =========================================================
            # LLM-REP ABLATION (EARLY EXIT)
            # =========================================================
            if self.is_decoding_mode and self.ft_only_encoder:
                b = features.size(0)
                nchunks = batch['inputs'].size(1) if batch['inputs'].dim() > 1 else 1
                
                true_batch_size = batch['labels'].size(0) if 'labels' in batch else b // nchunks
                
                # Dynamic Check: Pool inflated chunks (128 -> 32) safely.
                if b > true_batch_size and nchunks > 1:
                    features = features.view(true_batch_size, nchunks, -1).mean(dim=1)

                if features.dim() == 2:
                    features = features.unsqueeze(1)
                
                outputs={'outputs': features, 'decoding_logits': features}
                return (outputs, batch) if return_batch else outputs

            # =========================================================
            # STANDARD PRETRAINING / FULL MODEL BEHAVIOR
            # =========================================================
            if features.dim() == 2:
                features = features.unsqueeze(-1)

            b, f1, f2 = features.size()
            nchunks = batch['inputs'].size()[1]
            batch['inputs'] = features.view(b//nchunks, nchunks, f1*f2)
        
        if prep_batch:
            if len(batch['inputs'].size()) > 3:
                bsize, chunk, chann, time = batch['inputs'].size() 
                batch['inputs'] = batch['inputs'].view(bsize, chunk, chann*time)
            batch = self.prep_batch(batch=batch)
        else:
            assert 'inputs_embeds' in batch, 'inputs_embeds not in batch'

        batch['inputs_embeds'] = self.embedder(batch=batch)
        outputs = self.decoder(batch=batch)
        
        if self.unembedder is not None and not self.is_decoding_mode:
            outputs['outputs'] = self.unembedder(inputs=outputs['outputs'])['outputs']

        return (outputs, batch) if return_batch else outputss