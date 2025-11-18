#!/usr/bin/env python3
"""
Quick Start Script for Quillan v4.2 Bit-Level SOTA Model

This script demonstrates:
1. Building the model with bit-level optimizations
2. Setting up GRPO/DAPO training
3. Quantization for CPU inference
4. Basic usage examples

Run on Colab T4 GPU for training, local CPU for inference.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any

# Import from main model file
import sys
sys.path.append('.')
from 🧠 Quillan v4 import (
    QuillanBitSOTA,
    GRPODAPOTrainer,
    RLConfig,
    quantize_model_for_inference,
    BitOps
)

def build_model_for_training(device: torch.device) -> QuillanBitSOTA:
    """Build model optimized for T4 GPU training"""
    model = QuillanBitSOTA(
        vocab_size=50257,
        dim=512,
        num_mini_moes=32,
        num_experts_per_mini=8,
        num_micros_per_mini=325,
        num_layers=6,
        num_heads=8,
        max_seq_len=2048,
        diffusion_steps=10,
        use_bitnet=True,
        dropout=0.1
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model built: {total_params:,} parameters ({total_params/1e6:.2f}M)")
    print(f"   Estimated size (FP16): {total_params * 2 / 1e6:.2f} MB")
    
    return model

def setup_trainer(model: QuillanBitSOTA, device: torch.device) -> GRPODAPOTrainer:
    """Setup GRPO/DAPO trainer with optimal config for T4 GPU"""
    config = RLConfig(
        learning_rate=3e-4,
        batch_size=4,  # Adjust based on memory
        num_trajectories=4,
        max_trajectory_len=128,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        aux_loss_coef=0.01,
        diffusion_loss_coef=0.1,
        max_grad_norm=1.0,
        num_epochs=1000,
        warmup_steps=1000,
        use_mixed_precision=True  # BF16 for T4
    )
    
    trainer = GRPODAPOTrainer(model, config, device)
    print(f"✅ Trainer setup complete")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Mixed precision: {config.use_mixed_precision}")
    
    return trainer

def example_training_step(trainer: GRPODAPOTrainer, device: torch.device):
    """Example training step with mock data"""
    # Mock data (replace with your actual dataset)
    batch_size = 4
    seq_len = 128
    vocab_size = 50257
    
    # Questions (prompts)
    questions = [
        torch.randint(0, vocab_size, (seq_len,)).to(device)
        for _ in range(batch_size)
    ]
    
    # Trajectories (generated token sequences)
    trajectories = [
        list(torch.randint(0, vocab_size, (20,)).cpu().numpy())
        for _ in range(batch_size)
    ]
    
    # Rewards (task success scores)
    rewards = [0.8, 0.9, 0.7, 0.85]  # Mock rewards
    
    # Training step
    losses = trainer.train_step(questions, trajectories, rewards)
    
    print(f"\n📊 Training Step Results:")
    print(f"   Policy Loss: {losses['policy_loss']:.4f}")
    print(f"   Aux Loss: {losses['aux_loss']:.4f}")
    print(f"   Diffusion Loss: {losses['diffusion_loss']:.4f}")
    print(f"   Total Loss: {losses['total_loss']:.4f}")
    
    return losses

def test_bit_operations():
    """Test bit-level operations"""
    print("\n🔬 Testing Bit-Level Operations:")
    
    # Create test tensor
    x = torch.randn(2, 10, 512)
    bit_ops = BitOps()
    
    # Test bit packing
    x_binary = (x > 0).float()
    packed = bit_ops.pack_bits(x_binary, bits_per_value=1)
    print(f"   Original shape: {x_binary.shape}")
    print(f"   Packed shape: {packed.shape}")
    print(f"   Compression ratio: {x_binary.numel() / packed.numel():.1f}x")
    
    # Test bitwise XOR noise
    x_noisy = bit_ops.bitwise_xor_noise(x_binary, noise_prob=0.1)
    print(f"   ✅ Bitwise operations working")

def quantize_for_cpu_inference(model: QuillanBitSOTA, device: torch.device):
    """Quantize model for CPU inference"""
    print("\n⚙️  Quantizing model for CPU inference...")
    
    quantized_model = quantize_model_for_inference(model, device)
    
    # Estimate quantized size
    total_params = sum(p.numel() for p in quantized_model.parameters())
    print(f"   ✅ Quantization complete")
    print(f"   Estimated size (INT4/INT8): ~{total_params * 0.5 / 1e6:.2f} MB")
    
    return quantized_model

def test_inference(model: QuillanBitSOTA, device: torch.device):
    """Test inference on model"""
    print("\n🎯 Testing Inference...")
    
    model.eval()
    batch_size = 1
    seq_len = 128
    vocab_size = 50257
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, return_aux_loss=False)
        logits = outputs['logits']
        
        print(f"   ✅ Inference successful")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Output logits shape: {logits.shape}")
        
        # Test generation
        generated = model.generate(
            input_ids[:, :10],  # Short prompt
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        print(f"   Generated shape: {generated.shape}")

def main():
    """Main execution"""
    print("=" * 80)
    print("🧠 Quillan v4.2 Bit-Level SOTA - Quick Start")
    print("=" * 80)
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print(f"\n💻 Using CPU")
    
    # 1. Build model
    print("\n" + "=" * 80)
    print("STEP 1: Building Model")
    print("=" * 80)
    model = build_model_for_training(device)
    
    # 2. Setup trainer
    print("\n" + "=" * 80)
    print("STEP 2: Setting Up Trainer")
    print("=" * 80)
    trainer = setup_trainer(model, device)
    
    # 3. Test bit operations
    print("\n" + "=" * 80)
    print("STEP 3: Testing Bit Operations")
    print("=" * 80)
    test_bit_operations()
    
    # 4. Test inference
    print("\n" + "=" * 80)
    print("STEP 4: Testing Inference")
    print("=" * 80)
    test_inference(model, device)
    
    # 5. Example training step
    print("\n" + "=" * 80)
    print("STEP 5: Example Training Step")
    print("=" * 80)
    example_training_step(trainer, device)
    
    # 6. Quantization (for CPU deployment)
    print("\n" + "=" * 80)
    print("STEP 6: Quantization for CPU")
    print("=" * 80)
    quantized_model = quantize_for_cpu_inference(model, device)
    
    print("\n" + "=" * 80)
    print("✅ Quick Start Complete!")
    print("=" * 80)
    print("\n📝 Next Steps:")
    print("   1. Replace mock data with your actual dataset")
    print("   2. Train on Colab T4 GPU using trainer.train_step()")
    print("   3. Quantize after training for CPU inference")
    print("   4. Deploy quantized model on local CPU")
    print("\n📚 See BIT_LEVEL_OPTIMIZATION_GUIDE.md for detailed documentation")

if __name__ == "__main__":
    main()

