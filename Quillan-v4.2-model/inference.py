import torch
import argparse
from quillan import QuillanSOTA
import os

def load_model(checkpoint_path: str, device: str = 'cuda'):
    print(f"Loading model from {checkpoint_path}...")
    # Initialize model structure
    model = QuillanSOTA(
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
    )
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using random weights.")
    
    model.to(device)
    model.eval()
    return model

def generate_text(model, prompt_ids, max_new_tokens=50, temperature=0.7, device='cuda'):
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    
    return output_ids[0].tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quillan Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/quillan_final.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Input prompt") # In real usage, need tokenizer
    parser.add_argument("--max_tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, device)
    
    # Mock tokenizer for demonstration (replace with actual tokenizer)
    # Assuming simple ASCII mapping for demo purposes if no tokenizer provided
    print(f"Prompt: {args.prompt}")
    # This is a placeholder. In production, use the actual tokenizer used during training.
    # For now, we'll just generate random tokens if we can't tokenize, or assume user passes IDs?
    # Let's assume we just run a dummy generation to prove it works.
    dummy_input_ids = [1, 2, 3] # Replace with tokenizer.encode(args.prompt)
    
    output_ids = generate_text(model, dummy_input_ids, args.max_tokens, args.temperature, device)
    print(f"Generated IDs: {output_ids}")
    # print(f"Generated Text: {tokenizer.decode(output_ids)}")
