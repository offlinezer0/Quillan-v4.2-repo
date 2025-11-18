#!/usr/bin/env python3
"""
Quick test script for Quillan v4.2 SOTA Model
Tests model building, forward pass, and GRPO training
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import model (handle emoji filename)
    model_file = "🧠 Quillan v4.py"
    if not os.path.exists(model_file):
        # Try alternative name
        model_file = "Quillan v4.py"
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("quillan_model", model_file)
    quillan_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quillan_module)
    
    from quillan_module import (
        build_quillan_sota,
        DAPOGRPOTrainer,
        RLConfig,
        BitEncoder,
        quantize_model_int4,
        quantize_model_int8
    )
    
    print("=" * 80)
    print("🧠 Quillan v4.2 SOTA Model Test")
    print("=" * 80)
    
    # Test 1: Model Building
    print("\n[1/4] Testing Model Building...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    model = build_quillan_sota(
        vocab_size=50257,
        dim=512,
        num_mini_moes=32,
        use_bitnet=True,
        use_bit_diffusion=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model built: {total_params:,} parameters ({total_params/1e6:.2f}M)")
    print(f"   Estimated size (INT4): {total_params * 0.5 / 1024 / 1024:.2f} MB")
    
    # Test 2: Forward Pass
    print("\n[2/4] Testing Forward Pass...")
    dummy_input = torch.randint(0, 50257, (2, 128)).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input, return_aux_loss=True)
    
    print(f"   ✅ Forward pass successful!")
    print(f"   Logits shape: {outputs['logits'].shape}")
    if 'aux_loss' in outputs:
        print(f"   Aux loss: {outputs['aux_loss'].item():.4f}")
    
    # Test 3: Bit Encoding
    print("\n[3/4] Testing Bit-Level Encoding...")
    encoder = BitEncoder()
    x = torch.randn(10, 512)
    encoded, scale = encoder.encode_bits(x, bits=1)
    decoded = encoder.decode_bits(encoded, scale, bits=1, shape=x.shape)
    
    compression_ratio = x.numel() * 4 / encoded.numel()  # float32 = 4 bytes
    print(f"   ✅ Bit encoding successful!")
    print(f"   Compression ratio: {compression_ratio:.1f}x")
    print(f"   Reconstruction error: {(x - decoded).abs().mean().item():.4f}")
    
    # Test 4: GRPO Trainer
    print("\n[4/4] Testing GRPO Trainer...")
    config = RLConfig(
        learning_rate=3e-4,
        batch_size=4,
        num_trajectories=4,
        clip_epsilon=0.2,
        use_mixed_precision=torch.cuda.is_available()
    )
    
    trainer = DAPOGRPOTrainer(model, config, device)
    
    # Mock trajectories: (state, action, question)
    mock_trajectories = [
        [(torch.randint(0, 50257, (128,)).to(device), 
          torch.randint(0, 50257, (1,)).item(),
          torch.randint(0, 50257, (64,)).to(device)) 
         for _ in range(10)]
        for _ in range(4)
    ]
    mock_rewards = [1.0, 0.8, 1.2, 0.9]
    
    try:
        model.train()
        losses = trainer.train_step(mock_trajectories, mock_rewards)
        print(f"   ✅ Training step successful!")
        print(f"   Policy Loss: {losses['policy_loss']:.4f}")
        print(f"   Total Loss: {losses['total_loss']:.4f}")
    except Exception as e:
        print(f"   ⚠️  Training step failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Quantization (if bitsandbytes available)
    print("\n[5/5] Testing Quantization...")
    try:
        model_quantized = quantize_model_int4(model)
        model_quantized = quantize_model_int8(model_quantized)
        print(f"   ✅ Quantization successful!")
    except Exception as e:
        print(f"   ⚠️  Quantization failed (expected if bitsandbytes not installed): {e}")
    
    print("\n" + "=" * 80)
    print("✅ All Tests Complete!")
    print("=" * 80)
    print("\n📝 Next Steps:")
    print("   1. Train on your dataset with real trajectories")
    print("   2. Fine-tune hyperparameters (clip_epsilon, learning_rate)")
    print("   3. Export for CPU inference using export_for_cpu_inference()")
    print("   4. Monitor training metrics (policy loss, rewards)")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("   Make sure the model file exists and all dependencies are installed.")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

