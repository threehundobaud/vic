"""Reference with layer-input hooks for layers 0-5 so we can find first divergence."""
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

MODEL_DIR = "/data/huggingface/vib3-models/qwen3.6-27b-hf"
OUT_DIR = "/tmp/hfref"
os.makedirs(OUT_DIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
print("Prompt ids:", ids.tolist())

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
)
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
layers = lm.layers

# Dump the embedding output (layer 0 input)
cap = {}
def pre_hook(name):
    def h(mod, inp):
        t = inp[0] if isinstance(inp, tuple) else inp
        cap[name] = t.detach()
    return h

def post_hook(name):
    def h(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        cap[name] = t.detach()
    return h

hs = []
# embed output == input to layer 0
hs.append(lm.embed_tokens.register_forward_hook(post_hook("embed_out")))
for li in range(10):
    hs.append(layers[li].register_forward_pre_hook(pre_hook(f"layer{li}_input")))
    hs.append(layers[li].register_forward_hook(post_hook(f"layer{li}_output")))

with torch.no_grad():
    out = model(ids, use_cache=False)
print("top5:", out.logits[0, -1].topk(5))

for h in hs: h.remove()

def dump(name, t):
    """Dump tok0 (first prompt position) as FP32 and FP16."""
    t0 = t[0, 0] if t.dim()==3 else t[0]
    t1 = t[0, 1] if t.dim()==3 and t.shape[1]>1 else t0
    t0.detach().float().cpu().numpy().tofile(os.path.join(OUT_DIR, f"hf_{name}_tok0_f32.bin"))
    t1.detach().float().cpu().numpy().tofile(os.path.join(OUT_DIR, f"hf_{name}_tok1_f32.bin"))
    t0.detach().half().cpu().numpy().tofile(os.path.join(OUT_DIR, f"hf_{name}_tok0_f16.bin"))
    t1.detach().half().cpu().numpy().tofile(os.path.join(OUT_DIR, f"hf_{name}_tok1_f16.bin"))

for k, v in cap.items():
    dump(k, v)
    l2 = torch.norm(v[0, 0].float()).item() if v.dim()==3 else torch.norm(v[0].float()).item()
    print(f"  {k}: shape={tuple(v.shape)}, L2(tok0)={l2:.3f}")
