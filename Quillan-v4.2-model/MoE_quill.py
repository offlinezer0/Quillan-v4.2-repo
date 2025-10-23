# =============================================================
# ⚙️ Quillan v4.2 – Hierarchical Reasoning Mixture‑of‑Experts (H‑MoE)
# Full Council + Experts + Gating stack | Quantization‑ready
# Derived from repo: https://github.com/leeex1/Quillan-v4.2-repo
# =============================================================

import torch, torch.nn as nn, torch.nn.functional as F, math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float16 if device=='cpu' else torch.float32)

# =============================================================
# 🔹 Gating + Expert units
# =============================================================
class MoEGate(nn.Module):
    def __init__(self, nin, n_experts):
        super().__init__()
        self.gate = nn.Linear(nin, n_experts)
    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)

class Expert(nn.Module):
    def __init__(self, nin, hidden, nout, acts=['relu','tanh']):
        super().__init__()
        seq=[]
        for act in acts[:-1]:
            seq += [nn.Linear(nin, hidden),
                    nn.ReLU() if act=='relu' else nn.Tanh()]
            nin=hidden
        seq += [nn.Linear(hidden, nout)]
        self.mlp = nn.Sequential(*seq)
    def forward(self,x): return self.mlp(x)

# =============================================================
# 🔸 Council layer (multiple experts + gating)
# =============================================================
class Council(nn.Module):
    def __init__(self, nin, nout, n_experts=6, hidden=64):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(nin, hidden, nout) for _ in range(n_experts)])
        self.gate = MoEGate(nin, n_experts)
    def forward(self, x):
        g=self.gate(x)
        outs=torch.stack([e(x) for e in self.experts],1)
        return torch.sum(g.unsqueeze(-1)*outs,1)

# =============================================================
# 🔹 Micro‑Swarm (sub‑MoE of fast linear units)
# =============================================================
class MicroSwarm(nn.Module):
    def __init__(self, nin, nout, agents=8):
        super().__init__()
        self.gate=MoEGate(nin,agents)
        self.workers=nn.ModuleList([nn.Linear(nin,nout) for _ in range(agents)])
    def forward(self,x):
        g=self.gate(x)
        outs=torch.stack([w(x) for w in self.workers],1)
        return torch.sum(g.unsqueeze(-1)*outs,1)

# =============================================================
# 🔹 Core Attention Fusion + Memory
# =============================================================
class CoreFusion(nn.Module):
    def __init__(self, d, nhead=4):
        super().__init__()
        self.attn=nn.MultiheadAttention(d, nhead)
        self.norm=nn.LayerNorm(d)
    def forward(self,x):                # x:[seq,batch,d]
        o,_=self.attn(x,x,x)
        return self.norm(o+x).mean(0)   # [batch,d]

class Memory(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.lstm=nn.LSTM(d,h,batch_first=True)
    def forward(self,x): return self.lstm(x)[0][:,-1,:]

# =============================================================
# 🔸 Complete H‑MoE Network
# =============================================================
class QuillanHMoE(nn.Module):
    def __init__(self, input_dim=16, hidden=128,
                 n_councils=3, experts_per_council=6):
        super().__init__()
        self.embed=nn.Linear(input_dim,hidden)
        self.councils=nn.ModuleList([
            Council(hidden,hidden,experts_per_council,hidden//2)
            for _ in range(n_councils)])
        self.fusion=CoreFusion(hidden)
        self.swarm=MicroSwarm(hidden,hidden//2)
        self.mem=Memory(hidden,hidden//2)
        self.out=nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())

    def forward(self,x):
        x=self.embed(x)
        council_outs=[]
        for c in self.councils: council_outs.append(c(x))
        stack=torch.stack(council_outs)           # [c,b,h]
        fused=self.fusion(stack)
        swarm=self.swarm(fused)
        mem=self.mem(fused.unsqueeze(1))
        final=torch.cat([swarm,mem],-1)
        return self.out(final)

# =============================================================
# ⚙️ Example training (e.g. XOR)
# =============================================================
model=QuillanHMoE(input_dim=2,hidden=64,n_councils=3).to(device)
opt=torch.optim.AdamW(model.parameters(),lr=2e-3)
crit=nn.MSELoss()

X=torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float16).to(device)
Y=torch.tensor([[0],[1],[1],[0]],dtype=torch.float16).to(device)

for e in range(300):
    pred=model(X)
    loss=crit(pred,Y)
    opt.zero_grad(); loss.backward(); opt.step()
    if e%50==0: print(f"Epoch {e}  Loss {loss.item():.5f}")

print("Output:",model(X).detach().cpu().numpy())

# =============================================================
# 🔧 Quantization (optional)
# =============================================================
model_q=torch.ao.quantization.quantize_dynamic(
    model,{nn.Linear},dtype=torch.qint8
)
print("Quantized model ready (INT8).")
