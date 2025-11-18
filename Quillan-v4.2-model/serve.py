from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from quillan import QuillanSOTA
import os
from typing import List, Optional

app = FastAPI(title="Quillan API", description="API for Quillan v4.2 SOTA Model")

# Global model variable
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GenerateRequest(BaseModel):
    prompt_ids: List[int]
    max_new_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9

class GenerateResponse(BaseModel):
    generated_ids: List[int]

@app.on_event("startup")
async def load_model_event():
    global model
    checkpoint_path = "checkpoints/quillan_final.pt"
    print(f"Loading model from {checkpoint_path} on {device}...")
    
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

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_tensor = torch.tensor([request.prompt_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p
            )
        
        return GenerateResponse(generated_ids=output_ids[0].tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "device": device, "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
