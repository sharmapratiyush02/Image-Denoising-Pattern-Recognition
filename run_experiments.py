"""
Pattern Recognition: Image De-noising System
Experimental Pipeline & Benchmarking Script
Authors: Pratiyush Sharma, Akshada Shinde, Ayushi Kadam, Pushpak Agarwal
Seed: 42 for complete reproducibility
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from sklearn.decomposition import PCA
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# -------------------------------------------------------------
# Configuration and Reproducibility
# -------------------------------------------------------------
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

FIGURES_DIR = os.path.join(os.getcwd(), 'figures')
RESULTS_DIR = os.path.join(os.getcwd(), 'results')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

print("=" * 70)
print("LAB EXERCISE 8: IMAGE DE-NOISING SYSTEM BENCHMARK")
print(f"Random Seed: {RANDOM_SEED} | Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print("=" * 70)

# -------------------------------------------------------------
# 1. Dataset Loading (Fashion-MNIST)
# -------------------------------------------------------------
print("\n[Step 1/7] Loading and Partitioning Fashion-MNIST Dataset...")
transform = transforms.Compose([transforms.ToTensor()])

train_set_full = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_set_full = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

# Extract images as normalized numpy float32 arrays in [0, 1]
# Training set: first 8,000 images
# Validation set: images 8,000-9,000 of training split
# Test benchmark set: first 500 images of test split
X_train_clean = train_set_full.data[:8000].numpy().astype(np.float32) / 255.0
X_val_clean = train_set_full.data[8000:9000].numpy().astype(np.float32) / 255.0
X_test_clean = test_set_full.data[:500].numpy().astype(np.float32) / 255.0

print(f"Data shapes: Train={X_train_clean.shape}, Val={X_val_clean.shape}, Test={X_test_clean.shape}")

# -------------------------------------------------------------
# 2. Noise Corruption Generator Functions
# -------------------------------------------------------------
print("\n[Step 2/7] Defining Noise Corruption Models...")

def add_gaussian_noise(images, sigma_255):
    """
    Adds zero-mean Additive White Gaussian Noise (AWGN).
    sigma_255 is standard deviation on 0-255 scale.
    Normalized sigma = sigma_255 / 255.0.
    Output is strictly clipped to [0, 1].
    """
    sigma = sigma_255 / 255.0
    noise = np.random.normal(loc=0.0, scale=sigma, size=images.shape).astype(np.float32)
    noisy = np.clip(images + noise, 0.0, 1.0)
    return noisy

def add_salt_and_pepper_noise(images, density):
    """
    Adds impulse Salt & Pepper noise with total affected pixel fraction = density.
    p(salt=1.0) = density / 2
    p(pepper=0.0) = density / 2
    """
    noisy = images.copy()
    num_salt = int(np.ceil(density * images.size * 0.5))
    num_pepper = int(np.ceil(density * images.size * 0.5))

    # Salt (1.0)
    salt_coords = [np.random.randint(0, i, num_salt) for i in images.shape]
    noisy[tuple(salt_coords)] = 1.0

    # Pepper (0.0)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in images.shape]
    noisy[tuple(pepper_coords)] = 0.0

    return noisy

# Define the 5 evaluation conditions (ASCII-safe names)
NOISE_CONDITIONS = [
    {"name": "Gaussian (sigma=15)", "display": r"Gaussian ($\sigma=15$)", "type": "gaussian", "param": 15},
    {"name": "Gaussian (sigma=25)", "display": r"Gaussian ($\sigma=25$)", "type": "gaussian", "param": 25},
    {"name": "Gaussian (sigma=50)", "display": r"Gaussian ($\sigma=50$)", "type": "gaussian", "param": 50},
    {"name": "S&P (d=0.05)",        "display": "S&P (d=0.05)",            "type": "sp",       "param": 0.05},
    {"name": "S&P (d=0.10)",        "display": "S&P (d=0.10)",            "type": "sp",       "param": 0.10},
]

# Generate noisy test sets FIRST right after seeding so random draw matches baseline exactly
test_noisy_dict = {}
for cond in NOISE_CONDITIONS:
    if cond["type"] == "gaussian":
        test_noisy_dict[cond["name"]] = add_gaussian_noise(X_test_clean, cond["param"])
    else:
        test_noisy_dict[cond["name"]] = add_salt_and_pepper_noise(X_test_clean, cond["param"])
print(f"Generated {len(test_noisy_dict)} corrupted test sets.")

# Generate noisy validation sets (200 images) for hyperparameter tuning
val_noisy_dict = {}
val_clean_200 = X_val_clean[:200]
for cond in NOISE_CONDITIONS:
    if cond["type"] == "gaussian":
        val_noisy_dict[cond["name"]] = add_gaussian_noise(val_clean_200, cond["param"])
    else:
        val_noisy_dict[cond["name"]] = add_salt_and_pepper_noise(val_clean_200, cond["param"])
print("Generated corrupted validation sets (200 images) for hyperparameter tuning.")

# -------------------------------------------------------------
# Metric Computation Helper
# Note: PSNR is computed as the mean of per-image PSNR values
# -------------------------------------------------------------
def compute_metrics(clean_imgs, denoised_imgs):
    """Computes mean per-image PSNR, SSIM, and MSE across an image batch."""
    psnrs = []
    ssims = []
    mses = []
    for c, d in zip(clean_imgs, denoised_imgs):
        mse_val = float(np.mean((c - d) ** 2))
        psnr_val = float(compute_psnr(c, d, data_range=1.0))
        ssim_val = float(compute_ssim(c, d, data_range=1.0))
        mses.append(mse_val)
        psnrs.append(psnr_val)
        ssims.append(ssim_val)
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(mses))

# -------------------------------------------------------------
# 3. Solution 1: Classical Filter Hyperparameter Tuning
# -------------------------------------------------------------
print("\n[Step 3/7] Tuning Classical Filters on Validation Set (200 images)...")

def tune_filters_on_validation(val_clean, val_noisy_dict):
    tuning_results = {}
    
    gaussian_ksizes = [3, 5]
    gaussian_sigmas = [0.5, 0.8, 1.0, 1.5]
    
    median_ksizes = [3, 5]
    
    bilateral_ds = [5, 7, 9]
    bilateral_sig_colors = [0.05, 0.1, 0.15, 0.25]
    bilateral_sig_spaces = [1.5, 3.0]
    
    for cond in NOISE_CONDITIONS:
        cname = cond["name"]
        v_noisy = val_noisy_dict[cname]
        tuning_results[cname] = {}
        
        # 1. Tune Gaussian Filter
        best_g_psnr = -1.0
        best_g_params = {}
        for ks in gaussian_ksizes:
            for sig in gaussian_sigmas:
                den = np.array([cv2.GaussianBlur(img, (ks, ks), sigmaX=sig, sigmaY=sig) for img in v_noisy])
                den = np.clip(den, 0.0, 1.0)
                p = float(np.mean([compute_psnr(c, d, data_range=1.0) for c, d in zip(val_clean, den)]))
                if p > best_g_psnr:
                    best_g_psnr = p
                    best_g_params = {"ksize": ks, "sigma": sig, "val_psnr": round(p, 2)}
        tuning_results[cname]["Gaussian Filter"] = best_g_params
        
        # 2. Tune Median Filter
        best_m_psnr = -1.0
        best_m_params = {}
        for ks in median_ksizes:
            den = np.array([cv2.medianBlur(img, ks) for img in v_noisy])
            den = np.clip(den, 0.0, 1.0)
            p = float(np.mean([compute_psnr(c, d, data_range=1.0) for c, d in zip(val_clean, den)]))
            if p > best_m_psnr:
                best_m_psnr = p
                best_m_params = {"ksize": ks, "val_psnr": round(p, 2)}
        tuning_results[cname]["Median Filter"] = best_m_params
        
        # 3. Tune Bilateral Filter
        best_b_psnr = -1.0
        best_b_params = {}
        for d in bilateral_ds:
            for sc in bilateral_sig_colors:
                for ss in bilateral_sig_spaces:
                    den = np.array([cv2.bilateralFilter(img, d=d, sigmaColor=sc, sigmaSpace=ss) for img in v_noisy])
                    den = np.clip(den, 0.0, 1.0)
                    p = float(np.mean([compute_psnr(c, d, data_range=1.0) for c, d in zip(val_clean, den)]))
                    if p > best_b_psnr:
                        best_b_psnr = p
                        best_b_params = {"d": d, "sigma_color": sc, "sigma_space": ss, "val_psnr": round(p, 2)}
        tuning_results[cname]["Bilateral Filter"] = best_b_params
        
        print(f"  [{cname}] Tuned Params:")
        print(f"    Gaussian:  {best_g_params}")
        print(f"    Median:    {best_m_params}")
        print(f"    Bilateral: {best_b_params}")
        
    return tuning_results

filter_tuning_results = tune_filters_on_validation(val_clean_200, val_noisy_dict)
with open(os.path.join(RESULTS_DIR, 'filter_tuning.json'), 'w') as f:
    json.dump(filter_tuning_results, f, indent=2)
print("Saved chosen filter hyperparameters to results/filter_tuning.json")

def apply_tuned_gaussian(images, cname):
    params = filter_tuning_results[cname]["Gaussian Filter"]
    ks = params["ksize"]
    sig = params["sigma"]
    denoised = np.array([cv2.GaussianBlur(img, (ks, ks), sigmaX=sig, sigmaY=sig) for img in images])
    return np.clip(denoised, 0.0, 1.0)

def apply_tuned_median(images, cname):
    params = filter_tuning_results[cname]["Median Filter"]
    ks = params["ksize"]
    denoised = np.array([cv2.medianBlur(img, ks) for img in images])
    return np.clip(denoised, 0.0, 1.0)

def apply_tuned_bilateral(images, cname):
    params = filter_tuning_results[cname]["Bilateral Filter"]
    d = params["d"]
    sc = params["sigma_color"]
    ss = params["sigma_space"]
    denoised = np.array([cv2.bilateralFilter(img, d=d, sigmaColor=sc, sigmaSpace=ss) for img in images])
    return np.clip(denoised, 0.0, 1.0)

# -------------------------------------------------------------
# 4. Solution 2: PCA-based Subspace De-noising
# -------------------------------------------------------------
print("\n[Step 4/7] Implementing & Tuning Solution 2: PCA Denoising (Extended Sweep up to k=600)...")

X_train_flat = X_train_clean.reshape(len(X_train_clean), -1)
pca_estimator = PCA(n_components=600, random_state=RANDOM_SEED)
pca_estimator.fit(X_train_flat)

cum_var = np.cumsum(pca_estimator.explained_variance_ratio_)
k_90 = int(np.where(cum_var >= 0.90)[0][0] + 1)
k_95 = int(np.where(cum_var >= 0.95)[0][0] + 1)
print(f"PCA Variance Thresholds: 90% at k={k_90}, 95% at k={k_95}")

# Extended k candidates sweep up to 600 components
k_candidates = sorted(list(set([10, 20, 35, 50, 75, k_90, 120, k_95, 220, 300, 400, 500, 600])))
k_psnr_sweep = []
sample_noisy_val = val_noisy_dict["Gaussian (sigma=25)"].reshape(200, -1)
sample_clean_val = val_clean_200

# Optimal k is selected by maximizing reconstruction PSNR on the validation set under Gaussian sigma=25 (rather than by an arbitrary variance threshold).
for k in k_candidates:
    pca_k = PCA(n_components=k, random_state=RANDOM_SEED)
    pca_k.fit(X_train_flat)
    val_recon = pca_k.inverse_transform(pca_k.transform(sample_noisy_val)).reshape(-1, 28, 28)
    val_recon = np.clip(val_recon, 0.0, 1.0)
    p = float(np.mean([compute_psnr(c, d, data_range=1.0) for c, d in zip(sample_clean_val, val_recon)]))
    k_psnr_sweep.append(p)
    print(f"  Candidate k={k:3d} (Explained Var: {cum_var[k-1]*100:.1f}%) -> Val PSNR: {p:.2f} dB")

best_k = k_candidates[int(np.argmax(k_psnr_sweep))]
pca_mname = f"PCA (k={best_k})"
print(f"Selected Optimal PCA Components: {pca_mname} (Val PSNR = {max(k_psnr_sweep):.2f} dB)")

# Fit final optimal PCA model
pca_model = PCA(n_components=best_k, random_state=RANDOM_SEED)
pca_train_start = time.perf_counter()
pca_model.fit(X_train_flat)
pca_fit_time = time.perf_counter() - pca_train_start
print(f"PCA fit time for {pca_mname}: {pca_fit_time:.2f} s")

def denoise_pca(images, model):
    n = len(images)
    flat = images.reshape(n, -1)
    recon = model.inverse_transform(model.transform(flat)).reshape(n, 28, 28)
    return np.clip(recon, 0.0, 1.0)

# -------------------------------------------------------------
# 5. Solution 3: Convolutional Denoising Autoencoder (CDAE)
# -------------------------------------------------------------
print("\n[Step 5/7] Loading Solution 3: Convolutional Denoising Autoencoder (No Retraining)...")

class ConvDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(ConvDenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 64, 7, 7)

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # (B, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), # (B, 16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),          # (B, 1, 28, 28)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cdae_model = ConvDenoisingAutoencoder().to(device)
param_count = sum(p.numel() for p in cdae_model.parameters() if p.requires_grad)
print(f"CDAE constructed with {param_count:,} trainable parameters.")

cdae_weights_path = os.path.join(RESULTS_DIR, 'cdae_weights.pth')
history_path = os.path.join(RESULTS_DIR, 'cdae_history.json')

if os.path.exists(cdae_weights_path) and os.path.exists(history_path):
    print("Found pre-trained CDAE weights. Loading checkpoint (NO RETRAINING)...")
    cdae_model.load_state_dict(torch.load(cdae_weights_path, map_location=device))
    with open(history_path, 'r') as hf:
        hist_data = json.load(hf)
        train_loss_history = hist_data['train_loss']
        val_loss_history = hist_data['val_loss']
        cdae_total_train_time = hist_data['train_time_s']
    print(f"Loaded CDAE trained in {cdae_total_train_time:.1f}s.")
else:
    raise FileNotFoundError("cdae_weights.pth or cdae_history.json not found in results/!")

def denoise_cdae(images, model):
    model.eval()
    n = len(images)
    tensor_in = torch.tensor(images[:, np.newaxis, :, :], dtype=torch.float32).to(device)
    out_list = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = tensor_in[i:i+batch_size]
            recon = model(batch)
            out_list.append(recon.cpu().numpy())
    result = np.concatenate(out_list, axis=0).squeeze(1)
    return np.clip(result, 0.0, 1.0)

# -------------------------------------------------------------
# 6. Comprehensive Benchmarking & Comparison Table
# -------------------------------------------------------------
print("\n[Step 6/7] Running Full Benchmark Across All 5 Conditions...")

benchmark_records = []
denoised_cache = {}

for cond in NOISE_CONDITIONS:
    cname = cond["name"]
    print(f"\n--> Evaluating condition: {cname}")
    X_noisy = test_noisy_dict[cname]
    denoised_cache[cname] = {}

    condition_methods = [
        {"name": "Noisy Input",      "func": lambda x: x},
        {"name": "Gaussian Filter",  "func": lambda x, cn=cname: apply_tuned_gaussian(x, cn)},
        {"name": "Median Filter",    "func": lambda x, cn=cname: apply_tuned_median(x, cn)},
        {"name": "Bilateral Filter", "func": lambda x, cn=cname: apply_tuned_bilateral(x, cn)},
        {"name": pca_mname,          "func": lambda x: denoise_pca(x, pca_model)},
        {"name": "CDAE",             "func": lambda x: denoise_cdae(x, cdae_model)},
    ]

    for m in condition_methods:
        mname = m["name"]
        t_start = time.perf_counter()
        denoised = m["func"](X_noisy)
        elapsed = time.perf_counter() - t_start
        per_img_ms = (elapsed / len(X_noisy)) * 1000.0

        psnr_val, ssim_val, mse_val = compute_metrics(X_test_clean, denoised)
        denoised_cache[cname][mname] = denoised

        record = {
            "Noise Condition": cname,
            "Method": mname,
            "PSNR (dB)": round(psnr_val, 2),
            "SSIM": round(ssim_val, 4),
            "MSE": round(mse_val, 5),
            "Runtime (ms/img)": round(per_img_ms, 3),
        }
        benchmark_records.append(record)
        print(f"    {mname:18s} | PSNR: {psnr_val:6.2f} dB | SSIM: {ssim_val:.4f} | MSE: {mse_val:.5f} | Latency: {per_img_ms:.3f} ms")

# Save records to CSV
df_results = pd.DataFrame(benchmark_records)
df_results.to_csv(os.path.join(RESULTS_DIR, 'metrics_summary.csv'), index=False)

# Compute per-condition best method by PSNR and best method by SSIM (among all methods, including or excluding baseline)
best_by_psnr = {}
best_by_ssim = {}
for cond in NOISE_CONDITIONS:
    cname = cond["name"]
    cond_df = df_results[df_results["Noise Condition"] == cname]
    # Filter out Noisy Input for best denoising method
    denoiser_df = cond_df[cond_df["Method"] != "Noisy Input"]
    
    top_p_row = denoiser_df.loc[denoiser_df["PSNR (dB)"].idxmax()]
    best_by_psnr[cname] = {
        "method": str(top_p_row["Method"]),
        "psnr": float(top_p_row["PSNR (dB)"])
    }
    
    top_s_row = denoiser_df.loc[denoiser_df["SSIM"].idxmax()]
    best_by_ssim[cname] = {
        "method": str(top_s_row["Method"]),
        "ssim": float(top_s_row["SSIM"])
    }

summary_json_data = {
    "records": benchmark_records,
    "pca_fit_time_s": round(pca_fit_time, 2),
    "cdae_train_time_s": round(cdae_total_train_time, 2),
    "pca_best_k": best_k,
    "pca_mname": pca_mname,
    "train_loss_history": [round(float(x), 6) for x in train_loss_history],
    "val_loss_history": [round(float(x), 6) for x in val_loss_history],
    "k_candidates": [int(x) for x in k_candidates],
    "k_psnr_sweep": [round(float(x), 2) for x in k_psnr_sweep],
    "cum_var_ratio": [round(float(cum_var[k-1]), 4) for k in k_candidates],
    "best_by_psnr": best_by_psnr,
    "best_by_ssim": best_by_ssim
}

with open(os.path.join(RESULTS_DIR, 'metrics_summary.json'), 'w') as f:
    json.dump(summary_json_data, f, indent=2)

print("\nBenchmark successfully saved to CSV and JSON!")
print("Best by PSNR:", best_by_psnr)
print("Best by SSIM:", best_by_ssim)

# -------------------------------------------------------------
# 7. High-Resolution Visualizations (Saved to figures/)
# -------------------------------------------------------------
print("\n[Step 7/7] Generating High-Resolution (300 DPI) Publication Figures...")

# Consistent palette and methods across all figures
all_eval_methods = ["Noisy Input", "Gaussian Filter", "Median Filter", "Bilateral Filter", pca_mname, "CDAE"]
method_colors = {
    "Noisy Input": "#9E9E9E",
    "Gaussian Filter": "#4A90E2",
    "Median Filter": "#50E3C2",
    "Bilateral Filter": "#F5A623",
    pca_mname: "#9013FE",
    "CDAE": "#D0021B"
}

# 7.1 Sample Clean, Noisy, and De-noised Images Side-by-Side
print("  Plot 1: sample_denoised_comparisons.png")
fig, axes = plt.subplots(4, 7, figsize=(15, 9), dpi=300)
test_sample_indices = [0, 1, 3, 4] # Ankle boot, T-shirt, Dress, Sneaker
eval_conditions_show = ["Gaussian (sigma=25)", "S&P (d=0.05)", "Gaussian (sigma=50)", "S&P (d=0.10)"]
eval_labels_show = [r"Gaussian ($\sigma=25$)", "S&P (d=0.05)", r"Gaussian ($\sigma=50$)", "S&P (d=0.10)"]

for row_idx, (img_idx, cond_name, cond_lbl) in enumerate(zip(test_sample_indices, eval_conditions_show, eval_labels_show)):
    # Col 0: Clean
    axes[row_idx, 0].imshow(X_test_clean[img_idx], cmap='gray', vmin=0, vmax=1)
    if row_idx == 0:
        axes[row_idx, 0].set_title("Ground Truth (Clean)", fontsize=10, fontweight='bold', pad=8)
    axes[row_idx, 0].set_ylabel(f"{cond_lbl}", fontsize=10, fontweight='bold')
    axes[row_idx, 0].set_xticks([])
    axes[row_idx, 0].set_yticks([])

    for col_idx, mname in enumerate(all_eval_methods):
        ax = axes[row_idx, col_idx + 1]
        img_show = denoised_cache[cond_name][mname][img_idx]
        ax.imshow(img_show, cmap='gray', vmin=0, vmax=1)
        if row_idx == 0:
            ax.set_title(mname, fontsize=10, fontweight='bold', pad=8)
        p_val = compute_psnr(X_test_clean[img_idx], img_show, data_range=1.0)
        ax.set_xlabel(f"{p_val:.1f} dB", fontsize=8, labelpad=2)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'sample_denoised_comparisons.png'), bbox_inches='tight')
plt.close(fig)

# 7.2 Zoomed Detail Crops Showing Detail & Texture Preservation (Includes Bilateral Filter!)
print("  Plot 2: zoomed_detail_crops.png")
fig, axes = plt.subplots(2, 7, figsize=(16, 5.5), dpi=300)
crop_img_idx = 0  # Ankle boot
crop_cond = "Gaussian (sigma=25)"
crop_y1, crop_y2, crop_x1, crop_x2 = 12, 26, 8, 22

full_items = [
    ("Clean Reference", X_test_clean[crop_img_idx]),
    (r"Noisy ($\sigma=25$)", test_noisy_dict[crop_cond][crop_img_idx]),
    ("Gaussian Filter", denoised_cache[crop_cond]["Gaussian Filter"][crop_img_idx]),
    ("Median Filter", denoised_cache[crop_cond]["Median Filter"][crop_img_idx]),
    ("Bilateral Filter", denoised_cache[crop_cond]["Bilateral Filter"][crop_img_idx]),
    (pca_mname, denoised_cache[crop_cond][pca_mname][crop_img_idx]),
    ("CDAE", denoised_cache[crop_cond]["CDAE"][crop_img_idx]),
]

for col, (title, img) in enumerate(full_items):
    axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=1)
    rect = Rectangle((crop_x1, crop_y1), crop_x2 - crop_x1, crop_y2 - crop_y1,
                     linewidth=1.5, edgecolor='red', facecolor='none')
    axes[0, col].add_patch(rect)
    axes[0, col].set_title(title, fontsize=10, fontweight='bold')
    axes[0, col].axis('off')

for col, (title, img) in enumerate(full_items):
    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
    axes[1, col].imshow(crop, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[1, col].set_title(f"Zoomed ROI", fontsize=9, color='#b30000', fontweight='bold')
    axes[1, col].axis('off')

plt.suptitle("Region of Interest (ROI) Edge Preservation & Artifact Analysis", fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'zoomed_detail_crops.png'), bbox_inches='tight')
plt.close(fig)

# 7.3 PSNR Bar Chart: Includes Noisy Input as Hatched Grey Baseline
print("  Plot 3: psnr_bar_chart.png")
cond_keys = [c["name"] for c in NOISE_CONDITIONS]
cond_displays = [c["display"] for c in NOISE_CONDITIONS]
x = np.arange(len(cond_keys))
n_m = len(all_eval_methods)
width = 0.13

max_psnr_in_data = df_results["PSNR (dB)"].max()
min_psnr_in_data = df_results["PSNR (dB)"].min()

fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
for i, mname in enumerate(all_eval_methods):
    vals = [df_results[(df_results["Noise Condition"] == c) & (df_results["Method"] == mname)]["PSNR (dB)"].values[0] for c in cond_keys]
    offset = (i - (n_m - 1) / 2) * width
    if mname == "Noisy Input":
        ax.bar(x + offset, vals, width, label=mname, color=method_colors[mname], hatch='//', edgecolor='black', linewidth=0.6, alpha=0.8)
    else:
        ax.bar(x + offset, vals, width, label=mname, color=method_colors[mname], edgecolor='black', linewidth=0.6, alpha=0.9)

ax.set_ylabel("PSNR (dB) - Higher is Better", fontsize=11, fontweight='bold')
ax.set_title("Quantitative PSNR Comparison Across Noise Conditions", fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(cond_displays, fontsize=10, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(10, np.ceil(max_psnr_in_data) + 2)
ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=9, loc='upper right')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'psnr_bar_chart.png'), bbox_inches='tight')
plt.close(fig)

# 7.4 SSIM Bar Chart: Includes Noisy Input as Hatched Grey Baseline
print("  Plot 4: ssim_bar_chart.png")
max_ssim_in_data = df_results["SSIM"].max()
fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
for i, mname in enumerate(all_eval_methods):
    vals = [df_results[(df_results["Noise Condition"] == c) & (df_results["Method"] == mname)]["SSIM"].values[0] for c in cond_keys]
    offset = (i - (n_m - 1) / 2) * width
    if mname == "Noisy Input":
        ax.bar(x + offset, vals, width, label=mname, color=method_colors[mname], hatch='//', edgecolor='black', linewidth=0.6, alpha=0.8)
    else:
        ax.bar(x + offset, vals, width, label=mname, color=method_colors[mname], edgecolor='black', linewidth=0.6, alpha=0.9)

ax.set_ylabel("SSIM (Structural Similarity Index) - Higher is Better", fontsize=11, fontweight='bold')
ax.set_title("Quantitative SSIM Structural Fidelity Comparison", fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(cond_displays, fontsize=10, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 1.05)
ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=9, loc='upper right')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'ssim_bar_chart.png'), bbox_inches='tight')
plt.close(fig)

# 7.5 PSNR vs Noise Level Line Plot
print("  Plot 5: psnr_vs_noise_level.png")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

g_sigmas = [15, 25, 50]
g_conds = ["Gaussian (sigma=15)", "Gaussian (sigma=25)", "Gaussian (sigma=50)"]
markers = {'Noisy Input': 'x', 'Gaussian Filter': 'o', 'Median Filter': 's', 'Bilateral Filter': '^', pca_mname: 'd', 'CDAE': '*'}

for mname in all_eval_methods:
    y_vals = [df_results[(df_results["Noise Condition"] == c) & (df_results["Method"] == mname)]["PSNR (dB)"].values[0] for c in g_conds]
    linestyle = '--' if mname == 'Noisy Input' else '-'
    ax1.plot(g_sigmas, y_vals, marker=markers[mname], linestyle=linestyle, linewidth=2, markersize=7, label=mname, color=method_colors[mname])

ax1.set_xlabel(r"Gaussian Standard Deviation ($\sigma$)", fontsize=10, fontweight='bold')
ax1.set_ylabel("PSNR (dB)", fontsize=10, fontweight='bold')
ax1.set_title("(a) De-noising Performance: Gaussian AWGN", fontsize=11, fontweight='bold')
ax1.set_xticks(g_sigmas)
ax1.set_ylim(bottom=10)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=8, loc='upper right')

sp_densities = [0.05, 0.10]
sp_conds = ["S&P (d=0.05)", "S&P (d=0.10)"]

for mname in all_eval_methods:
    y_vals = [df_results[(df_results["Noise Condition"] == c) & (df_results["Method"] == mname)]["PSNR (dB)"].values[0] for c in sp_conds]
    linestyle = '--' if mname == 'Noisy Input' else '-'
    ax2.plot([d * 100 for d in sp_densities], y_vals, marker=markers[mname], linestyle=linestyle, linewidth=2, markersize=7, label=mname, color=method_colors[mname])

ax2.set_xlabel("Impulse Noise Density (%)", fontsize=10, fontweight='bold')
ax2.set_ylabel("PSNR (dB)", fontsize=10, fontweight='bold')
ax2.set_title("(b) De-noising Performance: Salt & Pepper Noise", fontsize=11, fontweight='bold')
ax2.set_xticks([5, 10])
ax2.set_ylim(bottom=10)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=8, loc='upper right')

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'psnr_vs_noise_level.png'), bbox_inches='tight')
plt.close(fig)

# 7.6 PCA Explained Variance Curve and PSNR vs Number of Components (Extended Sweep up to 600)
print("  Plot 6: pca_variance_and_psnr.png")
fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
color1 = '#1f77b4'
color2 = '#d62728'

cum_var_sweep = [float(cum_var[k-1]) * 100 for k in k_candidates]
ax1.plot(k_candidates, cum_var_sweep, marker='o', color=color1, linewidth=2.2, label='Cumulative Variance (%)')
ax1.axhline(90, color=color1, linestyle=':', alpha=0.7, label=f'90% Variance (k={k_90})')
ax1.axhline(95, color=color1, linestyle='--', alpha=0.7, label=f'95% Variance (k={k_95})')
ax1.axvline(best_k, color='gray', linestyle='-.', alpha=0.9, label=f'Optimal k={best_k} (by Val PSNR)')
ax1.set_xlabel("Number of Principal Components (k)", fontsize=11, fontweight='bold')
ax1.set_ylabel("Cumulative Explained Variance (%)", color=color1, fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(k_candidates, k_psnr_sweep, marker='s', color=color2, linewidth=2.2, linestyle='-', label=r'Validation PSNR (dB) on $\sigma=25$')
ax2.set_ylabel(r"Validation PSNR (dB) on Gaussian ($\sigma=25$)", color=color2, fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True, fontsize=8.5)

plt.title("PCA Dimensionality Selection: Cumulative Variance vs Reconstruction PSNR", fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'pca_variance_and_psnr.png'), bbox_inches='tight')
plt.close(fig)

# 7.7 Autoencoder Training and Validation Loss Curves
print("  Plot 7: cdae_loss_curves.png")
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
epochs_arr = range(1, len(train_loss_history) + 1)
ax.plot(epochs_arr, train_loss_history, marker='o', linewidth=2.2, color='#2b5c8f', label='Training MSE Loss')
ax.plot(epochs_arr, val_loss_history, marker='s', linewidth=2.2, color='#e05a47', label='Validation MSE Loss')

ax.set_xlabel("Epoch Number", fontsize=11, fontweight='bold')
ax.set_ylabel("Mean Squared Error (MSE)", fontsize=11, fontweight='bold')
ax.set_title("CDAE Convergence Trajectory Over 10 Epochs", fontsize=12, fontweight='bold', pad=10)
ax.set_xticks(list(epochs_arr))
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(frameon=True, fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'cdae_loss_curves.png'), bbox_inches='tight')
plt.close(fig)

# 7.8 Runtime Comparison Chart
print("  Plot 8: runtime_comparison.png")
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
runtime_methods = ["Gaussian Filter", "Median Filter", "Bilateral Filter", pca_mname, "CDAE"]
avg_latencies = [df_results[df_results["Method"] == m]["Runtime (ms/img)"].mean() for m in runtime_methods]
colors_run = [method_colors[m] for m in runtime_methods]

bars = ax.barh(runtime_methods, avg_latencies, color=colors_run, edgecolor='black', linewidth=0.6, alpha=0.85)
ax.set_xlabel("Mean Inference Latency (ms / image)", fontsize=11, fontweight='bold')
ax.set_title("Computational Efficiency Benchmark (Inference Latency)", fontsize=12, fontweight='bold', pad=12)
ax.grid(axis='x', linestyle='--', alpha=0.6)

for bar in bars:
    w = bar.get_width()
    ax.annotate(f"{w:.3f} ms",
                xy=(w, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=9, fontweight='bold')

ax.set_xlim(0, max(avg_latencies) * 1.35)
textstr = f"Model Offline Training Times:\n• PCA ({pca_mname}): {pca_fit_time:.2f} s\n• CDAE (10 Epochs): {cdae_total_train_time:.1f} s ({cdae_total_train_time/60:.1f} min)"
props = dict(boxstyle='round,pad=0.5', facecolor='#f7f9fa', edgecolor='#cccccc', alpha=0.95)
ax.text(0.52, 0.22, textstr, transform=ax.transAxes, fontsize=9.5, verticalalignment='top', bbox=props)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'runtime_comparison.png'), bbox_inches='tight')
plt.close(fig)

# 7.9 Residual (Difference) Images for Each Method
print("  Plot 9: residual_difference_maps.png")
fig, axes = plt.subplots(2, 6, figsize=(15, 5.5), dpi=300)
res_idx = 1 # T-shirt / Pullover
res_cond = "Gaussian (sigma=25)"
clean_ref = X_test_clean[res_idx]

res_items = [
    ("Noisy Input", denoised_cache[res_cond]["Noisy Input"][res_idx]),
    ("Gaussian Filter", denoised_cache[res_cond]["Gaussian Filter"][res_idx]),
    ("Median Filter", denoised_cache[res_cond]["Median Filter"][res_idx]),
    ("Bilateral Filter", denoised_cache[res_cond]["Bilateral Filter"][res_idx]),
    (pca_mname, denoised_cache[res_cond][pca_mname][res_idx]),
    ("CDAE", denoised_cache[res_cond]["CDAE"][res_idx]),
]

for col, (title, img) in enumerate(res_items):
    axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, col].set_title(title, fontsize=10, fontweight='bold')
    axes[0, col].axis('off')

    residual = np.abs(clean_ref - img)
    im_res = axes[1, col].imshow(residual, cmap='inferno', vmin=0, vmax=0.5)
    axes[1, col].set_title("Residual Error", fontsize=9, color='#d62728', fontweight='bold')
    axes[1, col].axis('off')

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.3])
cb = fig.colorbar(im_res, cax=cbar_ax)
cb.set_label("Absolute Error |I_clean - I_est|", fontsize=8, fontweight='bold')

plt.suptitle("Restored Images (Top) and Absolute Residual Error Maps (Bottom)", fontsize=12, fontweight='bold', y=0.98)
fig.savefig(os.path.join(FIGURES_DIR, 'residual_difference_maps.png'), bbox_inches='tight')
plt.close(fig)

print("=" * 70)
print("ALL EXPERIMENTS AND 9 HIGH-RESOLUTION FIGURES SUCCESSFULLY COMPLETED!")
print("=" * 70)
