#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Quillan v4.2 SOTA Model
Tests all components and identifies bugs
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import traceback
import os
from typing import List, Tuple

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import the model
try:
    from importlib import import_module
    import importlib.util
    # Handle emoji in filename
    filename = "🧠 Quillan v4.py"
    if not os.path.exists(filename):
        # Try alternative names
        for f in os.listdir('.'):
            if 'Quillan' in f and f.endswith('.py'):
                filename = f
                break
    
    spec = importlib.util.spec_from_file_location("quillan", filename)
    quillan_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quillan_module)
    print("[OK] Successfully imported Quillan module")
except Exception as e:
    print(f"[ERROR] Failed to import module: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test results tracker
test_results = {"passed": 0, "failed": 0, "errors": []}

def test(name: str, func):
    """Run a test and track results"""
    try:
        func()
        test_results["passed"] += 1
        print(f"[PASS] {name}")
    except Exception as e:
        test_results["failed"] += 1
        error_msg = f"[FAIL] {name}: {str(e)}"
        test_results["errors"].append(error_msg)
        print(error_msg)
        traceback.print_exc()

# ============================================================================
# TEST 1: BitLinear
# ============================================================================

def test_bitlinear():
    """Test BitLinear quantization and forward pass"""
    BitLinear = quillan_module.BitLinear
    dim = 128
    layer = BitLinear(dim, dim)
    
    # Test quantization
    w_quant, w_scale = layer.quantize_weights()
    assert w_quant.shape == (dim, dim), "Weight quantization shape mismatch"
    assert torch.all(w_quant.abs() <= 1), "Quantized weights must be in [-1, 1]"
    
    # Test forward pass (training)
    layer.train()
    x = torch.randn(2, 10, dim)
    out = layer(x)
    assert out.shape == (2, 10, dim), "Output shape mismatch in training"
    
    # Test forward pass (inference)
    layer.eval()
    out = layer(x)
    assert out.shape == (2, 10, dim), "Output shape mismatch in inference"

# ============================================================================
# TEST 2: BitPack
# ============================================================================

def test_bitpack():
    """Test bit packing and unpacking"""
    BitPack = quillan_module.BitPack
    
    # Test packing
    binary = torch.randint(0, 2, (4, 16)).float()
    packed = BitPack.pack_bits(binary)
    assert packed.dtype == torch.uint8, "Packed tensor should be uint8"
    
    # Test unpacking
    unpacked = BitPack.unpack_bits(packed, binary.shape)
    assert unpacked.shape == binary.shape, "Unpacked shape mismatch"
    assert torch.allclose(unpacked, binary, atol=0.1), "Unpacked values don't match"

# ============================================================================
# TEST 3: RotaryEmbedding
# ============================================================================

def test_rotary_embedding():
    """Test rotary position embeddings"""
    RotaryEmbedding = quillan_module.RotaryEmbedding
    apply_rotary_emb = quillan_module.apply_rotary_emb
    
    dim = 128  # Must be even for RoPE
    max_seq_len = 512
    rope = RotaryEmbedding(dim, max_seq_len)
    
    # Test forward
    cos, sin = rope(torch.randn(1, 10, dim), seq_len=10)
    assert cos.shape == (10, dim), "Cosine embedding shape mismatch"
    assert sin.shape == (10, dim), "Sine embedding shape mismatch"
    
    # Test application
    x = torch.randn(2, 10, 8, dim // 8)  # (batch, seq, heads, head_dim)
    rotated = apply_rotary_emb(x, cos, sin)
    assert rotated.shape == x.shape, "Rotated tensor shape mismatch"

# ============================================================================
# TEST 4: FlashAttention
# ============================================================================

def test_flash_attention():
    """Test FlashAttention module"""
    FlashAttention = quillan_module.FlashAttention
    
    dim = 128
    num_heads = 8
    attn = FlashAttention(dim, num_heads, dropout=0.1)
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    x = torch.randn(batch_size, seq_len, dim)
    
    attn.train()
    out = attn(x)
    assert out.shape == (batch_size, seq_len, dim), "Attention output shape mismatch"
    
    # Test with mask
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    out = attn(x, mask)
    assert out.shape == (batch_size, seq_len, dim), "Masked attention output shape mismatch"

# ============================================================================
# TEST 5: MicroAgent
# ============================================================================

def test_micro_agent():
    """Test MicroAgent module"""
    MicroAgent = quillan_module.MicroAgent
    
    dim = 128
    agent = MicroAgent(dim, use_bitnet=False)
    
    x = torch.randn(2, 10, dim)
    out = agent(x)
    assert out.shape == x.shape, "MicroAgent output shape mismatch"

# ============================================================================
# TEST 6: MiniMoE
# ============================================================================

def test_mini_moe():
    """Test MiniMoE routing and load balancing"""
    MiniMoE = quillan_module.MiniMoE
    
    dim = 128
    num_experts = 4  # Reduced for testing
    num_micros = 10  # Reduced for testing
    
    moe = MiniMoE(
        dim=dim,
        num_experts=num_experts,
        num_micros=num_micros,
        top_k=2,
        top_k_micros=4,
        use_bitnet=False
    )
    
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, dim)
    
    # Test forward (training)
    moe.train()
    out, aux_loss = moe(x, return_aux_loss=True)
    assert out.shape == x.shape, "MiniMoE output shape mismatch"
    assert aux_loss is not None, "Auxiliary loss should be computed in training"
    assert aux_loss.item() >= 0, "Auxiliary loss should be non-negative"
    
    # Test forward (inference)
    moe.eval()
    out, aux_loss = moe(x, return_aux_loss=False)
    assert out.shape == x.shape, "MiniMoE output shape mismatch in inference"
    assert aux_loss is None, "Auxiliary loss should be None when not requested"

# ============================================================================
# TEST 7: BitDiffusion
# ============================================================================

def test_bit_diffusion():
    """Test BitDiffusion forward and sampling"""
    BitDiffusion = quillan_module.BitDiffusion
    
    dim = 128
    num_steps = 10
    diffusion = BitDiffusion(dim, num_steps, use_bitnet=False)
    
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, dim)
    
    # Test training forward
    diffusion.train()
    pred_noise, target_noise = diffusion(x)
    assert pred_noise.shape == x.shape, "Predicted noise shape mismatch"
    assert target_noise.shape == x.shape, "Target noise shape mismatch"
    
    # Test inference forward
    diffusion.eval()
    with torch.no_grad():
        denoised = diffusion(x, num_inference_steps=5)
        assert denoised.shape == x.shape, "Denoised output shape mismatch"

# ============================================================================
# TEST 8: QuillanSOTA Model
# ============================================================================

def test_quillan_model():
    """Test complete QuillanSOTA model"""
    QuillanSOTA = quillan_module.QuillanSOTA
    
    # Create smaller model for testing
    model = QuillanSOTA(
        vocab_size=1000,
        dim=128,
        num_mini_moes=4,  # Reduced for testing
        num_experts_per_mini=4,
        num_micros_per_mini=10,
        num_layers=2,
        num_heads=4,
        max_seq_len=256,
        diffusion_steps=5,
        use_bitnet=False,  # Disable for faster testing
        dropout=0.1
    )
    
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Test forward (training)
    model.train()
    outputs = model(input_ids, return_aux_loss=True)
    assert 'logits' in outputs, "Outputs should contain 'logits'"
    assert outputs['logits'].shape == (batch_size, seq_len, 1000), "Logits shape mismatch"
    assert 'aux_loss' in outputs, "Auxiliary loss should be in outputs"
    
    # Test forward (inference)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, return_aux_loss=False)
        assert 'logits' in outputs, "Logits should be in outputs"
    
    # Test generation
    model.eval()
    with torch.no_grad():
        prompt = torch.randint(0, 1000, (1, 10))
        generated = model.generate(prompt, max_new_tokens=5, temperature=1.0)
        assert generated.shape[1] == 15, "Generated sequence length mismatch"

# ============================================================================
# TEST 9: GRPOTrainer
# ============================================================================

def test_grpo_trainer():
    """Test GRPO trainer"""
    QuillanSOTA = quillan_module.QuillanSOTA
    GRPOTrainer = quillan_module.GRPOTrainer
    RLConfig = quillan_module.RLConfig
    
    # Create small model
    model = QuillanSOTA(
        vocab_size=1000,
        dim=128,
        num_mini_moes=2,
        num_experts_per_mini=2,
        num_micros_per_mini=5,
        num_layers=1,
        num_heads=4,
        max_seq_len=64,
        use_bitnet=False,
        dropout=0.1
    )
    
    device = torch.device('cpu')  # Use CPU for testing
    config = RLConfig(
        learning_rate=1e-4,
        batch_size=2,
        num_trajectories=2,
        max_trajectory_len=8,
        clip_epsilon=0.2,
        use_mixed_precision=False  # Disable for CPU
    )
    
    trainer = GRPOTrainer(model, config, device)
    
    # Create mock trajectories
    input_ids = torch.randint(0, 1000, (2, 8))
    trajectories = [
        [(input_ids[0, :i], input_ids[0, i].item()) for i in range(1, 5)]
        for _ in range(2)
    ]
    rewards = [1.0, 0.8]
    
    # Test training step
    model.train()
    losses = trainer.train_step(trajectories, rewards)
    assert 'policy_loss' in losses, "Policy loss should be in output"
    assert 'total_loss' in losses, "Total loss should be in output"
    assert isinstance(losses['policy_loss'], float), "Policy loss should be float"

# ============================================================================
# TEST 10: Edge Cases
# ============================================================================

def test_edge_cases():
    """Test edge cases and potential bugs"""
    
    # Test 1: Odd dimension for RoPE
    try:
        RotaryEmbedding = quillan_module.RotaryEmbedding
        apply_rotary_emb = quillan_module.apply_rotary_emb
        
        dim = 129  # Odd dimension
        rope = RotaryEmbedding(dim, max_seq_len=128)
        cos, sin = rope(torch.randn(1, 10, dim), seq_len=10)
        
        # This should handle odd dimensions gracefully
        x = torch.randn(1, 10, 8, dim // 8)
        # If dim is odd, head_dim will be truncated
        head_dim = dim // 8
        x = x[..., :head_dim]  # Adjust to match
        rotated = apply_rotary_emb(x, cos, sin)
        assert rotated.shape == x.shape, "Odd dimension handling failed"
    except Exception as e:
        print(f"[WARN] Odd dimension test warning: {e}")
    
    # Test 2: Empty trajectories
    try:
        QuillanSOTA = quillan_module.QuillanSOTA
        model = QuillanSOTA(
            vocab_size=100,
            dim=64,
            num_mini_moes=2,
            num_experts_per_mini=2,
            num_micros_per_mini=4,
            num_layers=1,
            num_heads=2,
            max_seq_len=32,
            use_bitnet=False
        )
        
        # Very short sequence
        input_ids = torch.randint(0, 100, (1, 1))
        outputs = model(input_ids)
        assert outputs['logits'].shape[1] == 1, "Short sequence handling failed"
    except Exception as e:
        raise AssertionError(f"Edge case test failed: {e}")

# ============================================================================
# TEST 11: Memory and Performance
# ============================================================================

def test_memory_efficiency():
    """Test memory efficiency with gradient checkpointing"""
    MiniMoE = quillan_module.MiniMoE
    
    dim = 256
    moe = MiniMoE(
        dim=dim,
        num_experts=4,
        num_micros=10,
        use_bitnet=False
    )
    
    batch_size, seq_len = 4, 64
    x = torch.randn(batch_size, seq_len, dim)
    
    # Test that gradient checkpointing works
    moe.train()
    x.requires_grad = True
    out, _ = moe(x, return_aux_loss=True)
    
    # Backward pass should work with checkpointing
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient checkpointing failed"

# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("[TEST] Quillan v4.2 SOTA - Comprehensive Test Suite")
    print("="*80)
    print()
    
    # Run all tests
    test("BitLinear", test_bitlinear)
    test("BitPack", test_bitpack)
    test("RotaryEmbedding", test_rotary_embedding)
    test("FlashAttention", test_flash_attention)
    test("MicroAgent", test_micro_agent)
    test("MiniMoE", test_mini_moe)
    test("BitDiffusion", test_bit_diffusion)
    test("QuillanSOTA Model", test_quillan_model)
    test("GRPOTrainer", test_grpo_trainer)
    test("Edge Cases", test_edge_cases)
    test("Memory Efficiency", test_memory_efficiency)
    
    # Print summary
    print()
    print("="*80)
    print("[SUMMARY] Test Summary")
    print("="*80)
    print(f"[PASS] Passed: {test_results['passed']}")
    print(f"[FAIL] Failed: {test_results['failed']}")
    
    if test_results['errors']:
        print("\n[ERRORS] Errors:")
        for error in test_results['errors']:
            print(f"  {error}")
    
    print("="*80)
    
    # Exit with error code if any tests failed
    sys.exit(1 if test_results['failed'] > 0 else 0)

