#/usr/bin/env python3

import pdb
import torch
from typing import Dict
from einops import rearrange

class EmbeddingModel(torch.nn.Module):

    def __init__(
        self,
        in_dim: int = 1024,
        embed_dim: int = 768,
        num_hidden_layers: int = 1,
        dropout: int = 0.1,
        ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        layer_stack = []
        for _ in range(self.num_hidden_layers-1):
            layer_stack.extend(
                [
                    torch.nn.Linear(
                        in_features=self.in_dim,
                        out_features=self.embed_dim
                    ),
                    torch.nn.LayerNorm(self.embed_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(p=self.dropout)
                ]
            )
        layer_stack.extend(
            [
                torch.nn.Linear(
                    in_features=self.embed_dim if self.num_hidden_layers>1 else self.in_dim,
                    out_features=self.embed_dim
                ),
                torch.nn.LayerNorm(self.embed_dim),
                torch.nn.Dropout(p=self.dropout)
            ]
        )
        self.model = torch.nn.Sequential(*layer_stack)

    def _stack_inputs(
        self,
        tensor
        ) -> torch.tensor:
        
        return rearrange(
            tensor=tensor,
            pattern='b s e -> (b s) e'
        )

    def _unstack_inputs(
        self,
        tensor,
        b
        ) -> torch.tensor:
        
        return rearrange(
            tensor=tensor,
            pattern='(b s) e -> b s e',
            b=b
        )

    def forward(
        self,
        inputs,
        **kwargs
        ) -> torch.tensor:
        inputs_stacked = self._stack_inputs(tensor=inputs)
        
        return self._unstack_inputs(
            tensor=self.model(inputs_stacked),
            b=inputs.size()[0]
        )


class BaseEmbedder(torch.nn.Module):
    def __init__(self,
        in_dim: int = 1024,
        embed_dim: int = 768,
        num_hidden_layers: int = 1,
        dropout: float = 0.1,
        **kwargs
        ) -> None:
        super().__init__()
        self.name = 'BaseEmbedder'
        self.training_style = 'base'
        self._root_training_style = 'base'
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.xe_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.bxe_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.l2_loss = torch.nn.MSELoss(reduction='mean') # for L2 loss
        # self.huber_loss = torch.nn.HuberLoss(reduction='mean', delta=1.0) # for Huber loss
        
        self.embed_model = EmbeddingModel(
            in_dim=self.in_dim,
            embed_dim=self.embed_dim,
            num_hidden_layers=self.num_hidden_layers,
            dropout=self.dropout
        )
        self.is_decoding_mode = False

    def switch_decoding_mode(self, is_decoding_mode: bool=False) -> None:
        self.is_decoding_mode = is_decoding_mode
        
        if self.is_decoding_mode:
            self.training_style = 'decoding'
        else:
            self.training_style = self._root_training_style
    
    @staticmethod
    def _pad_tensor_left_by_n(
        tensor,
        n,
        pad_value
        ) -> torch.tensor:
        filling = torch.ones(
            (
                tensor.size()[0],
                n,
                *tensor.size()[2:]
            ),
            device=tensor.device
        ) * pad_value
        
        return torch.cat(
            [
                filling,
                tensor
            ],
            dim=1
        ).to(torch.long)

    @staticmethod
    def _round_to_precision(
        x: torch.tensor,
        precision: float,
        ) -> torch.tensor:
        return torch.round(x / precision) * precision


    def embed_inputs(
        self,
        inputs: torch.tensor
        ) -> torch.tensor:
        return self.embed_model(inputs)
    
    def forward(
        self,
        batch: Dict[str, torch.tensor]
        ) -> torch.tensor:
        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        
        if self.in_dim == self.embed_dim:
            inputs_embeds = batch[inputs_key]
        else:
            inputs_embeds = self.embed_inputs(inputs=batch[inputs_key])
        
        return inputs_embeds

    # def decoding_loss(
    #     self,
    #     decoding_logits,
    #     labels,
    #     **kwargs
    #     ) -> Dict[str, torch.tensor]:
    #     # pdb.set_trace()
    #     return {
    #         'decoding_loss': self.xe_loss(
    #             input=decoding_logits,
    #             target=labels.to(dtype=torch.long)
    #         )
    #     }
    # def decoding_loss(
    #         self,
    #         decoding_logits,
    #         labels,
    #         **kwargs
    #         ) -> Dict[str, torch.tensor]:
            
    #         # 1. Ensure logits are [32, 1] and labels are [32, 1]
    #         # 2. Ensure both are Float (BCE requires Float)
            
    #         target = labels.float()
    #         if target.ndim == 1:
    #             target = target.unsqueeze(1) # Reshape [32] -> [32, 1]

    #         # Use Binary Cross Entropy instead of Standard Cross Entropy
    #         # Note: Ideally, define self.bce_loss = nn.BCEWithLogitsLoss() in __init__
    #         # If you must use a functional call here:
    #         loss = torch.nn.functional.binary_cross_entropy_with_logits(
    #             input=decoding_logits, 
    #             target=target
    #         )

    #         return { 'decoding_loss': loss }

    # def decoding_loss(self, decoding_logits, labels, **kwargs) -> Dict[str, torch.tensor]:
            
    #         # 1. Select only the second class (Index 1)
    #         # Input: [32, 2, 1] -> [32, 1]
    #         decoding_logits = decoding_logits[:, 1, :] 
            
    #         # 2. Ensure labels are Float for BCE
    #         target = labels.float()
    #         if target.ndim == 1:
    #             target = target.unsqueeze(1)

    #         return {
    #             'decoding_loss': torch.nn.functional.binary_cross_entropy_with_logits(
    #                 input=decoding_logits,
    #                 target=target
    #             )
    #         }

    # def decoding_loss(self, decoding_logits, labels, **kwargs) -> Dict[str, torch.tensor]:
            
    #         # 1. Handle dimensionality dynamically
    #         if decoding_logits.ndim == 3:
    #             # If sequence length exists (e.g., [32, seq_len, 2]), select the appropriate token
    #             # Often, classification heads look at the first or last token. 
    #             # If your model specifically uses index 1 for classification, keep it as:
    #             decoding_logits = decoding_logits[:, 1, :] 
            
    #         # If it's 2D (e.g., [32, 2]), we don't need to slice the sequence dimension!
    #         # It already contains just the class probabilities for the batch.

    #         # 2. Extract the positive class probability (assuming binary classification)
    #         # If your logits are shape [batch_size, 2], you want the predictions for class 1
    #         if decoding_logits.shape[-1] == 2:
    #             decoding_logits = decoding_logits[:, 1]
            
    #         # 3. Ensure labels are Float for BCE and match the shape
    #         target = labels.float()
            
    #         # Ensure both are 1D arrays of shape [batch_size] for BCEWithLogitsLoss
    #         decoding_logits = decoding_logits.squeeze()
    #         target = target.squeeze()

    #         return {
    #             'decoding_loss': torch.nn.functional.binary_cross_entropy_with_logits(
    #                 input=decoding_logits,
    #                 target=target
    #             )
    #         }

    def decoding_loss(self, decoding_logits, labels, **kwargs) -> Dict[str, torch.tensor]:
        # Handle sequence dimension if it exists [Batch, Seq, Classes] -> [Batch, Classes]
        if decoding_logits.ndim == 3:
            # Take the CLS token or the relevant token for classification
            decoding_logits = decoding_logits[:, 0, :] 

        # --- CRITICAL FIX: Flatten labels to 1D to match logits ---
        # Flattens [Batch, Chunks] (e.g., [2, 4]) -> [8]
        labels_1d = labels.to(dtype=torch.long).view(-1)
        # ----------------------------------------------------------

        # 1. MULTI-CLASS CASE (TUEV - 6 classes)
        if decoding_logits.shape[-1] > 2:
            loss = self.xe_loss(
                input=decoding_logits,
                target=labels_1d
            )
        
        # 2. BINARY CASE (TUAB/BCI - 2 classes)
        else:
            # If shape is [Batch, 2], CrossEntropy is actually safer and easier
            loss = self.xe_loss(
                input=decoding_logits,
                target=labels_1d
            )
            # Alternatively, if you insist on BCE:
            # loss = torch.nn.functional.binary_cross_entropy_with_logits(...)

        return {'decoding_loss': loss}

    def reconstruction_loss(
        self,
        input,
        target,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        
        return {
            'reconstruction_loss': self.l2_loss(
                input=input,
                target=target
            )
        }

    def prep_batch(
        self,
        batch: Dict[str, torch.tensor]
        ) -> Dict:
        batch_out = {}
        
        for key in batch:
            
            if (
                torch.is_tensor(batch[key])
                and key != 'labels'
            ):
                batch_out[key] = batch[key].to(torch.float)
            
            elif key == 'labels':
                batch_out[key] = batch['labels'].to(torch.int)

            else:
                batch_out[key] = torch.clone(batch[key])
        
        # dummy copy of inputs to be used in forward pass
        batch_out['inputs_embeds'] = torch.clone(batch_out['inputs'])
        
        return batch_out

    def _root_loss(
        self,
        inputs,
        outputs,
        attention_mask,
        **kwargs
        ) -> Dict[str, torch.tensor]:
        attention_mask = torch.unsqueeze(attention_mask, -1).repeat(1,1,self.in_dim)
        
        return  self.reconstruction_loss(
            input=torch.masked_select(outputs, attention_mask.to(torch.bool)),
            target=torch.masked_select(inputs, attention_mask.to(torch.bool))
        )

    def loss(
        self,
        batch,
        outputs
        ) -> Dict[str, torch.tensor]:

        if self.is_decoding_mode:
            losses = self.decoding_loss(
                **batch,
                **outputs
            )
        
        else:
            losses = self._root_loss(
                **batch,
                **outputs
            )

        if 'loss' not in losses:
            losses['loss'] = sum(losses.values())

        return losses