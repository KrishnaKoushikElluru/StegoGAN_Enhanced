# StegoGAN-PlanIGN: Cross-Domain Context Enhanced StegoGAN

**Reimplementation & Extension on PlanIGN Aerial → Map Translation**

## Abstract

We re-implemented the StegoGAN framework for the PlanIGN Aerial↔Map dataset and introduced a novel cross-domain context module that predicts a mask from target-domain latent features and injects cross-domain information into the generator. This improves semantic preservation and reduces artifact hiding behaviors observed in the base model. We reproduced the baseline results at 256×256 and fine-tuned at 512×512 for higher fidelity. Evaluation uses CleanFID, KID, RMSE, and pixel-accuracy metrics. The repository contains training/evaluation scripts, model checkpoints, sample outputs, and the full IEEE research paper and presentation.


## Team Members

- **Krishna Koushik** – Register Number: 23MIA1107
- **Guru Raghav Raj** – Register Number: 23MIA1105
- **Amarender Reddy** – Register Number: 23MIA1012



## Base Paper Reference

- **StegoGAN**: Steganography Inspired Generative Adversarial Networks for Image Translation
- **PlanIGN Dataset Papers** (Aerial ↔ Map cartographic conversion)
- **link** : https://arxiv.org/abs/2403.20142


## Repository Structure
```
/StegoGAN-PlanIGN/
│
├── README.md
├── requirements.txt
├── dataset/                 # dataset or download_instruction.md
├── src/
│   ├── train.py             # training / fine-tuning script
│   ├── eval.py              # evaluation + metrics script
│   
├── models/
│   ├── ckpt_best_ema.pt     # (use LFS or Drive link)
│   └── baseline_ep256.pt
├── results/
│   ├── eval_baseline/
│   └── eval_ours/
├── presentation/
│   └── slides.pptx
└── report/
    └── IEEE_paper.pdf
```


## Short Description (≤350 characters)

Cross-domain StegoGAN implementation on the PlanIGN dataset with a novel cross-domain context/mask module. Reproduces baseline StegoGAN and fine-tunes at 512px. Includes training/evaluation scripts, metrics (FID/KID/RMSE/Acc), checkpoints, outputs, paper, and presentation.


## Tools & Libraries Used

- Python 3.9+
- PyTorch & TorchVision
- NumPy, Pandas
- CleanFID
- Torch-Fidelity
- Pillow (PIL)
- tqdm
- Matplotlib (optional)

### Example `requirements.txt`:
```txt
torch
torchvision
tqdm
Pillow
numpy
pandas
cleanfid
torch-fidelity
matplotlib
```

---

## Dataset Description

We use the **PlanIGN Aerial ↔ Map** paired dataset:

| Split      | Count | Description              |
|------------|-------|--------------------------|
| TrainA     | 1000  | Aerial tiles             |
| TrainB_TU  | 1000  | Map tiles (no toponyms)  |
| TestA      | 900   | Aerial test              |
| TestB_TU   | 900   | Ground truth maps        |

- All images originally **256×256**; fine-tuned at **512×512**
- RGB, PNG/JPG
- Contains roads, buildings, vegetation, urban/rural structures
- Paired by numeric tile IDs for pixel-alignment

A dataset download guide is provided in `/dataset/README.md`.

---

## How to Run (Colab or Local)
## COLAB

- import libraries from requirements
- Directly run the last cells, that was the updated one
## LOCAL(modified from Colab to .py)
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train / Resume
```bash
python src/train.py \
  --run-dir "./runs" \
  --src-folder "./dataset/trainA" \
  --tgt-folder "./dataset/trainB_TU" \
  --test-b-fid "./dataset/testB_TU" \
  --resume-from-epoch 323 \
  --resume-epochs 10 \
  --img-size 512 \
  --use-supervised \
  --reduce-cycle \
  --lambda-cycle 3 \
  --high-res-finetune \
  --freeze-enc-epochs 2 \
  --palette-snap \
  --palette-k 16 \
  --use-amp \
  --device cuda
```

### 3. Evaluate Model

**Baseline (256):**
```bash
python src/eval.py \
  --ckpt "./models/baseline_ep256.pt" \
  --a-test "./dataset/testA" \
  --b-test "./dataset/testB_TU" \
  --out "./results/baseline_ep256" \
  --use-ema \
  --img-size 256 \
  --device cuda
```

**Our method (512):**
```bash
python src/eval.py \
  --ckpt "./models/ckpt_best_ema.pt" \
  --a-test "./dataset/testA" \
  --b-test "./dataset/testB_TU" \
  --out "./results/ours_512" \
  --use-ema \
  --img-size 512 \
  --device cuda
```

**Outputs include:**
- `gen/` — generated maps
- `real/` — resized real maps
- `summary.csv` — summary metrics
- `per_image_metrics.csv` — RMSE & accuracy

---

## Output Screenshots / Results Summary

### Sample Output (Aerial → Map)

*(Insert your sample 5-image grid here)*

### Metrics Comparison

| Metric          | Baseline StegoGAN (256) | Our Cross-Attention GAN (512) |
|-----------------|-------------------------|-------------------------------|
| FID ↓           | 58.4                    | 58.5                          |
| KID ↓           | 0.024(2.4)              | 0.021(2.1)                    |
| RMSE-01 ↓       | 22.5                    | 10.48                         |
| Acc @ σ=2 ↑     | 66.1                    | 74.8                          |
| Acc @ σ=5 ↑     | 74.8                    | 84.54                          |



## YouTube Demo Link

👉 https://youtu.be/goLprWIh9Mo

