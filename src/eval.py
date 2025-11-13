#!/usr/bin/env python3
"""
eval.py
Generates A->B with a saved checkpoint and computes CleanFID, RMSE and Acc @ sigmas.
"""
import os, glob, re, json, argparse
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
import pandas as pd

# metrics libs
from cleanfid import fid as _fid
try:
    from cleanfid import kid as _kid
    CLEANFID_HAS_KID = True
except Exception:
    CLEANFID_HAS_KID = False
    from torch_fidelity import calculate_metrics as _tf_metrics

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="checkpoint path (.pt)")
parser.add_argument("--a-test", required=True, help="A test folder (inputs)")
parser.add_argument("--b-test", required=True, help="B test folder (reals)")
parser.add_argument("--out", required=True, help="output folder")
parser.add_argument("--use-ema", action="store_true")
parser.add_argument("--img-size", type=int, default=512)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

CKPT = args.ckpt
A_TEST = args.a_test
B_TEST = args.b_test
OUT = args.out
USE_EMA = args.use_ema
IMG_SIZE = args.img_size

os.makedirs(OUT, exist_ok=True)

# ---------- helpers ----------
def natural_sort_key(s):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def sorted_files(folder, exts=('png','jpg','jpeg','webp')):
    files=[]
    for e in exts: files += glob.glob(os.path.join(folder, f'*.{e}'))
    return sorted(files, key=natural_sort_key)

def extract_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.findall(r'(\d+)', stem)
    return m[-1] if m else stem

def to01(x): return (x.clamp(-1,1)+1)/2

# ---------- model (same generator as training) ----------
import torch.nn as nn
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0),
            nn.InstanceNorm2d(dim, affine=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0),
            nn.InstanceNorm2d(dim, affine=False),
        )
    def forward(self,x): return x + self.block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6):
        super().__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        in_ch = ngf
        for _ in range(2):
            out_ch = in_ch*2
            model += [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1), nn.InstanceNorm2d(out_ch), nn.ReLU(True)]
            in_ch = out_ch
        for _ in range(n_blocks):
            model += [ResnetBlock(in_ch)]
        self.encoder = nn.Sequential(*model)
        dec=[]
        for _ in range(2):
            out_ch = in_ch//2
            dec += [nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(out_ch), nn.ReLU(True)]
            in_ch = out_ch
        dec += [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, output_nc, 7), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)
    def forward(self,x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

# ---------- load checkpoint ----------
device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else "cpu")
G = ResNetGenerator().to(device)
state = torch.load(CKPT, map_location='cpu')
if USE_EMA and ('G_X2Y_ema' in state):
    try:
        G.load_state_dict(state['G_X2Y_ema']); tag='EMA'
    except Exception:
        G.load_state_dict(state.get('G_X2Y', state)); tag='RAW'
else:
    try:
        G.load_state_dict(state['G_X2Y']); tag='RAW'
    except Exception:
        G.load_state_dict(state); tag='RAW'
G.eval()
print("Loaded", tag)

# ---------- generate ----------
A_paths = sorted_files(A_TEST)
B_paths = sorted_files(B_TEST)
A_map = {extract_id(p): p for p in A_paths}
B_map = {extract_id(p): p for p in B_paths}
ids = sorted(set(A_map.keys()) & set(B_map.keys()), key=natural_sort_key)
print("Matched test IDs:", len(ids))

GEN_DIR  = os.path.join(OUT, f"gen_{tag.lower()}_{IMG_SIZE}"); os.makedirs(GEN_DIR, exist_ok=True)
REAL_DIR = os.path.join(OUT, f"real_{IMG_SIZE}"); os.makedirs(REAL_DIR, exist_ok=True)

tfA = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3,(0.5,)*3)
])
resize_ = transforms.Resize((IMG_SIZE,IMG_SIZE), Image.BICUBIC)

with torch.no_grad():
    for i, id_ in enumerate(tqdm(ids, desc=f"Generate A->B ({IMG_SIZE})")):
        a = tfA(Image.open(A_map[id_]).convert('RGB')).unsqueeze(0).to(device)
        y,_ = G(a)
        y01 = to01(y).squeeze(0).cpu().numpy().transpose(1,2,0)
        Image.fromarray((y01*255).clip(0,255).astype(np.uint8)).save(os.path.join(GEN_DIR, f"{id_}.png"))

for path in tqdm(sorted_files(B_TEST), desc=f"Prep real ({IMG_SIZE})"):
    img = Image.open(path).convert('RGB')
    img = resize_(img)
    img.save(os.path.join(REAL_DIR, os.path.basename(path)))

# ---------- compute CleanFID, KID fallback ----------
print("compute FID/KID ...")
fid_val = _fid.compute_fid(GEN_DIR, REAL_DIR, mode="clean")

kid_mean, kid_std = None, None
if CLEANFID_HAS_KID:
    try:
        km, ks = _kid.compute_kid(GEN_DIR, REAL_DIR, mode="clean")
        kid_mean, kid_std = float(km), float(ks)
    except Exception:
        kid_mean, kid_std = None, None
else:
    try:
        m = _tf_metrics(input1=GEN_DIR, input2=REAL_DIR, kid=True, fid=False, verbose=False)
        kid_mean = float(m["kernel_inception_distance_mean"])
        kid_std  = float(m["kernel_inception_distance_std"])
    except Exception:
        kid_mean, kid_std = None, None

print(f"FID({IMG_SIZE}): {fid_val:.6f} | KID: {kid_mean} ± {kid_std}")

# ---------- RMSE & accuracy ----------
SIGMA_2, SIGMA_5 = 2, 5
def rmse_and_acc(pred_path, real_path, sigma2=2, sigma5=5):
    pr = np.array(Image.open(pred_path).convert('RGB'), dtype=np.float32)
    gt = np.array(Image.open(real_path).convert('RGB'), dtype=np.float32)
    assert pr.shape == gt.shape, f"shape mismatch: {pr.shape} vs {gt.shape}"
    diff = np.abs(pr - gt)
    rmse_255 = np.sqrt((diff**2).mean())
    correct_sigma2 = (diff < sigma2).any(axis=2).mean()
    correct_sigma5 = (diff < sigma5).any(axis=2).mean()
    rmse_01 = rmse_255 / 255.0
    return rmse_01, rmse_255, correct_sigma2, correct_sigma5

rows = []
for id_ in tqdm(ids, desc="Compute per-image metrics"):
    pred = os.path.join(GEN_DIR,  f"{id_}.png")
    real = os.path.join(REAL_DIR, f"{id_}.png")
    rm01, rm255, acc2, acc5 = rmse_and_acc(pred, real, SIGMA_2, SIGMA_5)
    rows.append({"id": id_, "rmse_01": rm01, "rmse_255": rm255, "acc_sigma2": acc2, "acc_sigma5": acc5})

df = pd.DataFrame(rows).sort_values("id")
df_path = os.path.join(OUT, f"per_image_metrics_{tag.lower()}_{IMG_SIZE}.csv")
df.to_csv(df_path, index=False)

summary = {
    "pairs": len(ids),
    "fid_cleanfid_512": float(fid_val),
    "kid_mean": kid_mean,
    "kid_std": kid_std,
    "rmse_01_mean": float(df["rmse_01"].mean()),
    "rmse_255_mean": float(df["rmse_255"].mean()),
    "acc_sigma2_mean": float(df["acc_sigma2"].mean()),
    "acc_sigma5_mean": float(df["acc_sigma5"].mean()),
    "gen_dir": GEN_DIR,
    "real_dir": REAL_DIR,
    "used_weights": tag,
}
with open(os.path.join(OUT, f"summary_{tag.lower()}_{IMG_SIZE}.json"), "w") as f:
    json.dump(summary, f, indent=2)
pd.DataFrame([summary]).to_csv(os.path.join(OUT, f"summary_{tag.lower()}_{IMG_SIZE}.csv"), index=False)

print("Saved summary and per-image CSVs to", OUT)
