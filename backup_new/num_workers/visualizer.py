import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Academic styling
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'lines.linewidth': 2.5
})

# --- DATA FOR COMPARISON PLOTS (Unchanged) ---
batches = ['Batch 8', 'Batch 16', 'Batch 32']
x = np.arange(len(batches))
width = 0.35

time_w0 = [147, 237, 406]
time_w4 = [117, 194, 297]
energy_w0 = [0.022, 0.036, 0.062]
energy_w4 = [0.018, 0.031, 0.047]

# --- PLOT 1: TIME COMPARISON ---
plt.figure(figsize=(10, 8))
plt.bar(x - width/2, time_w0, width, label='num_workers=0', color='tab:red', alpha=0.8, edgecolor='black')
plt.bar(x + width/2, time_w4, width, label='num_workers=4', color='tab:green', alpha=0.8, edgecolor='black')
for i, val in enumerate(time_w0): plt.text(i - width/2, val + 5, f'{val}s', ha='center', fontweight='bold')
for i, val in enumerate(time_w4): plt.text(i + width/2, val + 5, f'{val}s', ha='center', fontweight='bold')
plt.title('Training Time: Workers Comparison', fontweight='bold')
plt.ylabel('Total Time (s)'); plt.xticks(x, batches); plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout(); plt.savefig('comparison_time.png', dpi=300); plt.close()

# --- PLOT 2: ENERGY COMPARISON ---
plt.figure(figsize=(10, 8))
plt.bar(x - width/2, energy_w0, width, label='num_workers=0', color='tab:red', alpha=0.8, edgecolor='black')
plt.bar(x + width/2, energy_w4, width, label='num_workers=4', color='tab:green', alpha=0.8, edgecolor='black')
for i, val in enumerate(energy_w0): plt.text(i - width/2, val + 0.001, f'{val:.3f}', ha='center', fontweight='bold')
for i, val in enumerate(energy_w4): plt.text(i + width/2, val + 0.001, f'{val:.3f}', ha='center', fontweight='bold')
plt.title('Total Energy: Workers Comparison', fontweight='bold')
plt.ylabel('Energy (kWh)'); plt.xticks(x, batches); plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout(); plt.savefig('comparison_energy.png', dpi=300); plt.close()

# --- AVERAGING LOGIC FOR GPU PLOTS ---
batch_dirs = {'Batch 8': 'batch_8', 'Batch 16': 'batch_16', 'Batch 32': 'batch_32'}
colors = {'Batch 8': 'tab:green', 'Batch 16': 'tab:blue', 'Batch 32': 'tab:red'}

def get_stats(b_dir, metric):
    files = glob.glob(os.path.join(b_dir, "*.csv"))
    if not files: return None, None
    dfs = [pd.read_csv(f) for f in files]
    min_len = min(len(d) for d in dfs)
    trimmed = [d[metric].iloc[:min_len].reset_index(drop=True) for d in dfs]
    combined = pd.concat(trimmed, axis=1)
    return combined.mean(axis=1), combined.std(axis=1)

def plot_csv_metric_with_err(metric_col, title, ylabel, filename):
    plt.figure(figsize=(12, 8))
    for label, b_dir in batch_dirs.items():
        if os.path.exists(b_dir):
            mu, sigma = get_stats(b_dir, metric_col)
            if mu is not None:
                # Smoothing
                smooth_mu = mu.rolling(window=5, min_periods=1).mean()
                smooth_sigma = sigma.rolling(window=5, min_periods=1).mean()
                
                plt.plot(mu.index, smooth_mu, label=label, color=colors[label])
                plt.fill_between(mu.index, smooth_mu - smooth_sigma, 
                                 smooth_mu + smooth_sigma, color=colors[label], alpha=0.2)
    
    plt.title(title, fontweight='bold')
    plt.ylabel(ylabel); plt.xlabel('Step'); plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6); plt.ylim(bottom=0)
    if metric_col == 'gpu_util_pct': plt.ylim(0, 105)
    plt.tight_layout(); plt.savefig(filename, dpi=300); plt.close()

# --- PLOT 3: GPU UTILIZATION (Averaged) ---
plot_csv_metric_with_err('gpu_util_pct', 'GPU Utilization (num_workers=0)', 'Utilization %', 'gpu_util_workers.png')

# --- PLOT 4: GPU POWER (Averaged) ---
plot_csv_metric_with_err('gpu_power', 'GPU Power Consumption (num_workers=0)', 'Watts', 'gpu_power_workers.png')

print("Four plots generated: comparison_time.png, comparison_energy.png, gpu_util_workers.png (with error fill), gpu_power_workers.png (with error fill)")