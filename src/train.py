#!/usr/bin/env python3
"""
train.py
Adapted from your Colab training cell. Run with --help for options.
"""
import os, glob, re, shutil, json, argparse, math, time
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils

# ----------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=str, required=True, help="Base run dir (checkpoints, samples will be created here)")
parser.add_argument("--src-folder", type=str, required=True, help="trainA folder (aerial)")
parser.add_argument("--tgt-folder", type=str, required=True, help="trainB folder (maps)")
parser.add_argument("--test-b-fid", type=str, required=True, help="Test B folder used for quick FID")
parser.add_argument("--resume-from-epoch", type=int, default=323)
parser.add_argument("--resume-epochs", type=int, default=10)
parser.add_argument("--img-size", type=int, default=512)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--num-workers", type=int, default=2)
parser.add_argument("--use-supervised", action="store_true")
parser.add_argument("--lambda-sup", type=float, default=30.0)
parser.add_argument("--reduce-cycle", action="store_true")
parser.add_argument("--lambda-cycle", type=float, default=3.0)
parser.add_argument("--high-res-finetune", action="store_true")
parser.add_argument("--freeze-enc-epochs", type=int, default=2)
parser.add_argument("--do-eval-at-end", action="store_true")
parser.add_argument("--palette-snap", action="store_true")
parser.add_argument("--palette-k", type=int, default=16)
parser.add_argument("--use-amp", action="store_true")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

# -- normalize boolean flags
USE_SUPERVISED = args.use_supervised
REDUCE_CYCLE = args.reduce_cycle
HIGH_RES_FINETUNE = args.high_res_finetune
DO_EVAL_AT_END = args.do_eval_at_end
PALETTE_SNAP = args.palette_snap

# ----------------- paths -----------------
RUN_DIR = args.run_dir
SRC_FOLDER = args.src_folder
TGT_FOLDER = args.tgt_folder
TEST_B_FOR_FID = args.test_b_fid
RESUME_FROM_EPOCH = args.resume_from_epoch
RESUME_EPOCHS = args.resume_epochs
IMG_SIZE = args.img_size
BATCH = args.batch_size
N_WORKERS = args.num_workers

CKPT_DIR = os.path.join(RUN_DIR, "checkpoints"); os.makedirs(CKPT_DIR, exist_ok=True)
SAMPLES_DIR = os.path.join(RUN_DIR, "samples"); os.makedirs(SAMPLES_DIR, exist_ok=True)
LOCAL_CKPT_BACKUP = os.path.join("/tmp", "ckpts_backup"); os.makedirs(LOCAL_CKPT_BACKUP, exist_ok=True)
LOCAL_SAMPLES_BACKUP = os.path.join("/tmp", "samples_backup"); os.makedirs(LOCAL_SAMPLES_BACKUP, exist_ok=True)

device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else "cpu")
device_type = "cuda" if device.type == "cuda" else "cpu"
print("device_type:", device_type)

# ----------------- helpers -----------------
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

def robust_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    shutil.copy(src, tmp)
    os.replace(tmp, dst)

def robust_save_checkpoint(state, epoch):
    local_tmp   = os.path.join('/tmp', f'ckpt_ep{epoch}.pt')
    final_local = os.path.join(LOCAL_CKPT_BACKUP, f'ckpt_ep{epoch}.pt')
    dst         = os.path.join(CKPT_DIR, f'ckpt_ep{epoch}.pt')
    try:
        torch.save(state, local_tmp)
    except Exception:
        torch.save(state, final_local)
        return False, final_local
    try:
        robust_copy(local_tmp, dst)
        os.remove(local_tmp)
        return True, dst
    except Exception:
        shutil.move(local_tmp, final_local)
        return False, final_local

def robust_save_sample(grid, epoch, tag=""):
    fname = f'sample_ep{epoch}{tag}.png'
    local = os.path.join(LOCAL_SAMPLES_BACKUP, fname)
    dst   = os.path.join(SAMPLES_DIR, fname)
    try:
        vutils.save_image(grid, local)
    except Exception:
        return False, None
    try:
        robust_copy(local, dst)
        return True, dst
    except Exception:
        return False, local

def to01(x): return (x.clamp(-1,1)+1)/2

# safer quick FID: writes small temp folder and calls CleanFID externally (user must have cleanfid installed)
def quick_fid_epoch_preview(G, epoch, src_files, real_dir, n=64, size=256):
    from cleanfid import fid as _fid
    G.eval(); imgs=[]
    tf = transforms.Compose([
        transforms.Resize((size,size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3,(0.5,)*3)
    ])
    small_dir = os.path.join(RUN_DIR, f"_fid_ep{epoch}", "_small")
    if os.path.exists(small_dir):
        shutil.rmtree(small_dir)
    os.makedirs(small_dir, exist_ok=True)

    with torch.no_grad():
        take = min(n, len(src_files))
        for i, a_path in enumerate(src_files[:take]):
            A = tf(Image.open(a_path).convert('RGB')).unsqueeze(0).to(device)
            y,_ = G(A)
            t = to01(y).squeeze(0).cpu().numpy().transpose(1,2,0)
            arr = (np.clip(t,0,1)*255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(small_dir, f"{i:04d}.png"))

    fid_val = _fid.compute_fid(small_dir, real_dir, mode="clean")
    return float(fid_val), small_dir

# ----------------- dataset (paired by ID) -----------------
src_all = sorted_files(SRC_FOLDER)
tgt_all = sorted_files(TGT_FOLDER)
print("Found dataset sizes:", len(src_all), "->", len(tgt_all))
src_dict = {extract_id(p): p for p in src_all}
tgt_dict = {extract_id(p): p for p in tgt_all}
common_ids = sorted(set(src_dict.keys()) & set(tgt_dict.keys()), key=natural_sort_key)
pairs = [(src_dict[i], tgt_dict[i], i) for i in common_ids]
if len(pairs) == 0:
    raise RuntimeError("No paired IDs between src and tgt")
print(f"Paired tiles (by ID): {len(pairs)}")

class PairDataset(Dataset):
    def __init__(self, pairs, size):
        self.pairs = pairs; self.size = size
        self.tfA = transforms.Compose([
            transforms.Resize((size,size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3,(0.5,)*3)
        ])
        self.tfB = transforms.Compose([
            transforms.Resize((size,size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3,(0.5,)*3)
        ])
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        a_path, b_path, id_str = self.pairs[idx]
        A = Image.open(a_path).convert('RGB')
        B = Image.open(b_path).convert('RGB')
        return {'A': self.tfA(A), 'B': self.tfB(B), 'id': id_str}

num_workers = max(0, min(N_WORKERS, (os.cpu_count() or 2)//2))
ds = PairDataset(pairs, IMG_SIZE)
dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=num_workers, drop_last=False)

# ----------------- model defs -----------------
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

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw=4; pad=1
        seq = [nn.Conv2d(input_nc, ndf, kw, stride=2, padding=pad), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult; nf_mult = min(2**n, 8)
            seq += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kw, stride=2, padding=pad), nn.InstanceNorm2d(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult; nf_mult = min(2**n_layers, 8)
        seq += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kw, stride=1, padding=pad), nn.InstanceNorm2d(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf*nf_mult, 1, kw, stride=1, padding=pad)]
        self.model = nn.Sequential(*seq)
    def forward(self,x): return self.model(x)

class MaskPredictor(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, max(in_ch//2,4), 3, padding=1), nn.ReLU(True),
            nn.Conv2d(max(in_ch//2,4), max(in_ch//4,4), 3, padding=1), nn.ReLU(True),
            nn.Conv2d(max(in_ch//4,4), in_ch, 1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x)

# instantiate & warm-up mask shape
_tmp = ResNetGenerator().to(device)
with torch.no_grad():
    _z = _tmp.encoder(torch.randn(1,3,IMG_SIZE,IMG_SIZE, device=device))
mask_predictor = MaskPredictor(_z.shape[1]).to(device)
del _tmp, _z

G_X2Y = ResNetGenerator().to(device)
G_Y2X = ResNetGenerator().to(device)
D_X = NLayerDiscriminator(3).to(device)
D_Y = NLayerDiscriminator(3).to(device)
G_X2Y_ema = ResNetGenerator().to(device)

def ema_update(model, ema, decay):
    with torch.no_grad():
        for (n, p), (_, ep) in zip(model.state_dict().items(), ema.state_dict().items()):
            ep.mul_(decay).add_(p, alpha=1.0-decay)

# ---------- hyperparams ----------
CFG = {
    'lr_D': 1.2e-5,
    'lr_G': 4.5e-6,
    'lambda_cyc': (args.lambda_cycle if (USE_SUPERVISED and REDUCE_CYCLE) else 10.0),
    'lambda_id': 0.05,
    'lambda_match': 0.65,
    'lambda_reg': 0.0001,
    'grad_clip': 5.0,
    'use_amp': args.use_amp,
    'ema_decay': 0.9995
}

MASK_MEAN_TARGET_BASE = 0.14
LAMBDA_MASK_MEAN_BASE = 0.35
MASK_MEAN_TARGET_WARM = 0.14
LAMBDA_MASK_MEAN_WARM = 0.40
MASK_WARMUP_EPOCHS    = 3
MASK_ENTROPY_W        = 0.02

# ---------- resume checkpoint ----------
def ckpt_path_for_epoch(ckpt_dir, ep):
    return os.path.join(ckpt_dir, f"ckpt_ep{ep}.pt")

ckpt_path = ckpt_path_for_epoch(CKPT_DIR, RESUME_FROM_EPOCH)
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

print(f"Resuming from checkpoint: {ckpt_path}")
state = torch.load(ckpt_path, map_location='cpu')

def safe_load(m, key):
    if key in state:
        try:
            m.load_state_dict(state[key]); print("loaded", key)
        except Exception as e:
            print("warn: failed to load", key, "->", e)

safe_load(G_X2Y, 'G_X2Y')
safe_load(G_Y2X, 'G_Y2X')
safe_load(D_X, 'D_X')
safe_load(D_Y, 'D_Y')
# try mask load (may mismatch if channel sizes changed)
try:
    safe_load(mask_predictor, 'mask')
except Exception as e:
    print("mask load failed:", e)

if 'G_X2Y_ema' in state:
    try:
        G_X2Y_ema.load_state_dict(state['G_X2Y_ema']); print("loaded G_X2Y_ema")
    except:
        G_X2Y_ema.load_state_dict(G_X2Y.state_dict()); print("fallback EMA=G")
else:
    G_X2Y_ema.load_state_dict(G_X2Y.state_dict()); print("init EMA=G")
for p in G_X2Y_ema.parameters(): p.requires_grad = False

# optimizers
g_params = list(G_X2Y.parameters()) + list(G_Y2X.parameters())
opt_G = torch.optim.Adam(
    [
        {'params': g_params,                   'lr': CFG['lr_G']},
        {'params': mask_predictor.parameters(),'lr': 2e-5},
    ],
    betas=(0.5, 0.999)
)
opt_Dx = torch.optim.Adam(D_X.parameters(), lr=CFG['lr_D'], betas=(0.5, 0.999))
opt_Dy = torch.optim.Adam(D_Y.parameters(), lr=CFG['lr_D'], betas=(0.5, 0.999))
scaler = torch.amp.GradScaler(enabled=CFG['use_amp'])

# try restore opt/scaler if present
for key, opt in (('opt_G', opt_G), ('opt_Dx', opt_Dx), ('opt_Dy', opt_Dy)):
    if key in state:
        try: opt.load_state_dict(state[key]); print(f"{key} restored")
        except Exception as e: print(f"{key} restore failed:", e)
if 'scaler' in state:
    try: scaler.load_state_dict(state['scaler']); print("scaler restored")
    except Exception as e: print("scaler restore failed:", e)

start_epoch = int(state.get('epoch', RESUME_FROM_EPOCH))
print(f"Resuming at epoch {start_epoch} -> running {RESUME_EPOCHS} more")

def set_encoders_requires_grad(flag: bool):
    for m in [G_X2Y.encoder, G_Y2X.encoder]:
        for p in m.parameters():
            p.requires_grad = flag

# ---------- loss helpers ----------
mse = nn.MSELoss()
def adv_D(D, real, fake):
    return 0.5*(mse(D(real), torch.ones_like(D(real))) + mse(D(fake.detach()), torch.zeros_like(D(fake.detach()))))
def adv_G(D, fake):
    return 0.5*mse(D(fake), torch.ones_like(D(fake)))

# ---------- training ----------
src_for_fid = [p[0] for p in pairs]
best_quick_fid = float('inf')
end_epoch = start_epoch + RESUME_EPOCHS

for epoch in range(start_epoch, end_epoch):
    if HIGH_RES_FINETUNE and (epoch - start_epoch) < args.freeze_enc_epochs:
        set_encoders_requires_grad(False)
    else:
        set_encoders_requires_grad(True)

    pbar = tqdm(enumerate(dl), total=len(dl), desc=f"Epoch {epoch+1}")
    running = {'G':0.0, 'D':0.0, 'mask_mean':0.0}; steps=0

    for i, batch in pbar:
        steps += 1
        A = batch['A'].to(device); B = batch['B'].to(device)

        with torch.amp.autocast(device_type=device_type, enabled=CFG['use_amp']):
            y_from_A, z_A = G_X2Y(A)
            x_from_B, z_B = G_Y2X(B)

            mask = mask_predictor(z_B)
            z_unmatch = mask * z_B
            z_match   = (1.0 - mask) * z_B

            xgen = G_Y2X.decoder(z_match)
            combined = z_A + z_unmatch
            yrec = G_X2Y.decoder(combined)

        # D update
        opt_Dx.zero_grad(set_to_none=True); opt_Dy.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_type, enabled=CFG['use_amp']):
            loss_Dx = adv_D(D_X, A, xgen)
            loss_Dy = adv_D(D_Y, B, y_from_A)
            loss_D  = loss_Dx + loss_Dy
        scaler.scale(loss_D).backward()
        try:
            scaler.unscale_(opt_Dx); scaler.unscale_(opt_Dy)
        except RuntimeError: pass
        torch.nn.utils.clip_grad_norm_(D_X.parameters(), CFG['grad_clip'])
        torch.nn.utils.clip_grad_norm_(D_Y.parameters(), CFG['grad_clip'])
        scaler.step(opt_Dx); scaler.step(opt_Dy)

        # G update
        opt_G.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_type, enabled=CFG['use_amp']):
            loss_G_adv  = adv_G(D_X, xgen) + adv_G(D_Y, y_from_A)
            X_rec, _    = G_Y2X(y_from_A)
            Y_rec, _    = G_X2Y(x_from_B)
            loss_cycle  = F.l1_loss(X_rec, A) + F.l1_loss(Y_rec, B)

            id_x, _     = G_Y2X(A); id_y, _ = G_X2Y(B)
            loss_id     = F.l1_loss(id_x, A) + F.l1_loss(id_y, B)

            lreg            = (mask.abs() + 1e-6).pow(0.5).mean()
            loss_match      = F.l1_loss(x_from_B, xgen)

            if (epoch - start_epoch) < MASK_WARMUP_EPOCHS:
                mask_target = MASK_MEAN_TARGET_WARM
                lambda_m    = LAMBDA_MASK_MEAN_WARM
            else:
                mask_target = MASK_MEAN_TARGET_BASE
                lambda_m    = LAMBDA_MASK_MEAN_BASE

            mask_mean_now   = mask.mean()
            loss_mask_mean  = (mask_mean_now - mask_target).pow(2) * lambda_m

            eps = 1e-6
            mask_entropy = -(mask.clamp(eps,1-eps)*torch.log(mask.clamp(eps,1-eps)) + (1-mask).clamp(eps,1-eps)*torch.log((1-mask).clamp(eps,1-eps))).mean()

            loss_sup = torch.tensor(0.0, device=device)
            if USE_SUPERVISED:
                loss_sup = F.l1_loss(y_from_A, B)
            lambda_sup_now = (10.0 if (epoch - start_epoch) < 2 else args.lambda_sup)

            loss_G = (loss_G_adv + CFG['lambda_cyc']*loss_cycle + 0.5*loss_id + CFG['lambda_reg']*lreg + CFG['lambda_match']*loss_match + loss_mask_mean + MASK_ENTROPY_W*mask_entropy + (lambda_sup_now * loss_sup if USE_SUPERVISED else 0.0))

        scaler.scale(loss_G).backward()
        try:
            scaler.unscale_(opt_G)
        except RuntimeError: pass
        torch.nn.utils.clip_grad_norm_(list(G_X2Y.parameters())+list(G_Y2X.parameters())+list(mask_predictor.parameters()), CFG['grad_clip'])
        scaler.step(opt_G); scaler.update()

        ema_update(G_X2Y, G_X2Y_ema, CFG['ema_decay'])

        mm = float(mask_mean_now.detach().cpu().item())
        running['G'] += float(loss_G.item()); running['D'] += float(loss_D.item()); running['mask_mean'] += mm
        pbar.set_postfix({'G_loss': running['G']/steps, 'D_loss': running['D']/steps, 'mask_mean': running['mask_mean']/steps})

    # epoch end: sample & save
    eid = epoch + 1
    with torch.no_grad():
        a = to01(A[:1].detach().cpu())
        b = to01(B[:1].detach().cpu())
        y_s, zA_s = G_X2Y_ema(A[:1])
        x_s, zB_s = G_Y2X(B[:1])
        mask_s = mask_predictor(zB_s).mean(dim=1, keepdim=True)
        mask_up = F.interpolate(mask_s, size=(IMG_SIZE,IMG_SIZE), mode='bilinear', align_corners=False).repeat(1,3,1,1).cpu().clamp(0,1)
        x01 = to01(x_s.detach().cpu()); y01 = to01(y_s.detach().cpu())
        grid = vutils.make_grid(torch.cat([a, x01, b, y01, mask_up], dim=0), nrow=5)
        ok_s, where_s = robust_save_sample(grid, eid, tag="_ema")
        print(f"[SAMPLE] → {where_s}")

    out_state = {
        'epoch': eid,
        'G_X2Y': G_X2Y.state_dict(), 'G_Y2X': G_Y2X.state_dict(),
        'D_X': D_X.state_dict(), 'D_Y': D_Y.state_dict(),
        'mask': mask_predictor.state_dict(),
        'G_X2Y_ema': G_X2Y_ema.state_dict(),
        'opt_G': opt_G.state_dict(), 'opt_Dx': opt_Dx.state_dict(), 'opt_Dy': opt_Dy.state_dict(),
        'scaler': scaler.state_dict()
    }
    ok, where = robust_save_checkpoint(out_state, eid)
    print(f"[CKPT] → {where}")

    # quick FID (uses current IMG_SIZE)
    try:
        qfid, small_dir = quick_fid_epoch_preview(G_X2Y_ema, eid, src_for_fid, TEST_B_FOR_FID, n=64, size=IMG_SIZE)
        print(f"[quick FID@{IMG_SIZE}] epoch {eid}: {qfid:.2f}  (folder: {small_dir})")
        if qfid < best_quick_fid:
            best_quick_fid = qfid
            best_path = os.path.join(CKPT_DIR, "ckpt_best_ema.pt")
            torch.save({'epoch': eid, 'G_X2Y_ema': G_X2Y_ema.state_dict()}, best_path)
            print(f"[BEST EMA] updated → {best_path}")
    except Exception as e:
        print("[quick FID] skipped:", e)

    print(f"Epoch {eid} done. AvgG={running['G']/len(dl):.4f}, AvgD={running['D']/len(dl):.4f}, mask_mean={running['mask_mean']/len(dl):.6f}")

print("Done.")
# Optional full eval is intentionally left out of train.py; use eval.py for full eval.
