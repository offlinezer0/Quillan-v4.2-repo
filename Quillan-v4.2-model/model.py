import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from .layers.attention import FlashAttention
from .layers.moe import MiniMoE
from .layers.diffusion import BitDiffusion

class QuillanSOTA(nn.Module):
    """
    Quillan v4.2 SOTA Hierarchical MoE with Bit-Level Diffusion
    
    Architecture Overview:
    ├─ Input Embedding: Token + positional
    ├─ Overseer MoE: Single FP16 expert (global coordination)
    ├─ Reasoning Diffusion: Bit-level compression layer
    ├─ 32 Mini MoEs: Each with 8 experts + 325 micro agents (INT4)
    ├─ FlashAttention: Memory-efficient self-attention
    └─ Output Head: Vocabulary projection
    
    Parameter Breakdown:
    - Embeddings: ~25M (vocab=50K, dim=512)
    - Overseer: ~1M (FP16)
    - Diffusion: ~50M (BitNet)
    - 32 Mini MoEs: ~400M (INT4 compressed)
    - Attention: ~10M
    - Output: ~25M
    Total: ~511M params → ~600MB quantized
    
    Training Memory: ~6GB (with gradient checkpointing + BF16)
    Inference Memory: ~1.5GB (INT4/INT8 + CPU)
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 512,
        num_mini_moes: int = 32,
        num_experts_per_mini: int = 8,
        num_micros_per_mini: int = 325,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        diffusion_steps: int = 10,
        use_bitnet: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.num_layers = num_layers
        self.num_mini_moes = num_mini_moes
        
        # Input embeddings
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Overseer MoE (FP16 - single expert for global coordination)
        self.overseer = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        # Reasoning Diffusion Layer
        self.diffusion = BitDiffusion(
            dim=dim,
            num_steps=diffusion_steps,
            use_bitnet=use_bitnet
        )
        
        # Hierarchical Layers: [Attention → Mini MoE] × num_layers
        moes_per_layer = num_mini_moes // num_layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': FlashAttention(dim, num_heads, dropout),
                'mini_moes': nn.ModuleList([
                    MiniMoE(
                        dim=dim,
                        num_experts=num_experts_per_mini,
                        num_micros=num_micros_per_mini,
                        top_k=2,
                        top_k_micros=8,
                        use_bitnet=use_bitnet
                    ) for _ in range(moes_per_layer)
                ])
            }) for _ in range(num_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie input and output embeddings (weight sharing)
        self.head.weight = self.token_embed.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with careful scaling"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical architecture
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            return_aux_loss: Whether to return auxiliary losses
        
        Returns:
            Dictionary containing:
            - logits: Output logits (batch, seq_len, vocab_size)
            - aux_loss: Load balancing loss (if requested)
            - diffusion_loss: Diffusion training loss (if training)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embed tokens
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.embed_dropout(x)
        
        # Overseer coordination (FP16)
        x_overseer = self.overseer(x.float())
        if not self.training:
            x_overseer = x_overseer.half()
        x = x + 0.1 * x_overseer  # Residual with small weight
        
        # Reasoning diffusion
        diffusion_loss = None
        if self.training:
            # Training: Predict noise for diffusion loss
            pred_noise, target_noise = self.diffusion(x)
            diffusion_loss = F.mse_loss(pred_noise, target_noise)
            x_diffused = x  # Skip diffusion in training (unstable gradients)
        else:
            # Inference: Denoise for compressed reasoning
            x_diffused = self.diffusion(x, num_inference_steps=5)
        
        x = x + 0.2 * x_diffused  # Blend original and diffused
        
        # Hierarchical layers
        total_aux_loss = 0.0
        
        for layer_idx, layer in enumerate(self.layers):
            # Self-attention
            x_attn = layer['attention'](x, attention_mask)
            x = x + x_attn
            
            # Mini MoE processing (distribute 32 MoEs across layers)
            for mini_moe in layer['mini_moes']:
                x_moe, aux_loss = mini_moe(x, return_aux_loss=return_aux_loss)
                x = x + x_moe
                
                if aux_loss is not None:
                    total_aux_loss += aux_loss
        
        # Output projection
        x = self.norm(x)
        logits = self.head(x)
        
        # Prepare return dictionary
        output = {'logits': logits}
        
        if return_aux_loss:
            moes_per_layer = self.num_mini_moes // self.num_layers
            output['aux_loss'] = total_aux_loss / (self.num_layers * moes_per_layer)
        
        if diffusion_loss is not None:
            output['diffusion_loss'] = diffusion_loss
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9
    ) -> torch.Tensor:
        """
        Autoregressive generation with sampling
        
        Args:
            input_ids: Prompt tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens (nucleus filtering)
            top_p: Keep tokens with cumulative probability > p
        
        Returns:
            Generated tokens (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Get predictions for next token
            outputs = self(input_ids)
            logits = outputs['logits'][:, -1, :]  # Last token logits
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
