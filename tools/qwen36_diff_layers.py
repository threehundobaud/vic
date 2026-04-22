import os, numpy as np
VIB3 = "/home/brian/code/vib3/dump"
HF = "/tmp/hfref"

def load_f32(p): return np.frombuffer(open(p,"rb").read(), dtype=np.float32)

print(f"{'stage':<24} {'vib3_L2':>8} {'hf_L2':>8} {'diff_L2':>9} {'cos':>7}")
print("-" * 66)

for T in (0,):
    print(f"\n=== tok{T} ===")
    for L in range(10):
        v_path = f"{VIB3}/vib3_postlayer_f32_L{L}_tok{T}.bin"
        h_path = f"{HF}/hf_layer{L}_output_tok{T}_f32.bin"
        if not (os.path.exists(v_path) and os.path.exists(h_path)):
            print(f"L{L} output: missing")
            continue
        v = load_f32(v_path); h = load_f32(h_path)
        if v.size != h.size:
            print(f"L{L} output: SIZE MISMATCH vib3={v.size} hf={h.size}")
            continue
        la, lb = np.linalg.norm(v), np.linalg.norm(h)
        d = v - h
        ld = np.linalg.norm(d)
        cos = float(np.dot(v, h) / (la*lb + 1e-20))
        print(f"L{L} output             {la:>8.3f} {lb:>8.3f} {ld:>9.3f} {cos:>7.4f}")
