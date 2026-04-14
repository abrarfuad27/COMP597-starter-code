
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Set academic style defaults
plt.rcParams.update({
    'font.size': 23,
    'axes.titlesize': 23,
    'axes.labelsize': 23,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 23, 
    'lines.linewidth': 2.5,
    'figure.titlesize': 24
})

def get_batch_stats(batch_dir, batch_size):
    """Reads 1,2,3.csv. Returns Mean and Std Dev DataFrames."""
    files = glob.glob(os.path.join(batch_dir, "*.csv"))
    if not files:
        return None, None
        
    processed_dfs = []
    for f in files:
        df = pd.read_csv(f)
        df = df.select_dtypes(include=[np.number]).copy()
        
        if 'duration' in df.columns:
            df['steps_per_sec'] = 1.0 / df['duration']
            df['samples_per_sec'] = batch_size / df['duration']
            
            cols_to_clean = ['steps_per_sec', 'samples_per_sec']
            df[cols_to_clean] = df[cols_to_clean].replace([np.inf, -np.inf], np.nan)
            df[cols_to_clean] = df[cols_to_clean].fillna(df[cols_to_clean].mean())
            
        processed_dfs.append(df)

    if not processed_dfs:
        return None, None

    min_rows = min(len(d) for d in processed_dfs)
    trimmed_dfs = [d.iloc[:min_rows].reset_index(drop=True) for d in processed_dfs]
    
    combined = pd.concat(trimmed_dfs)
    stats_grouped = combined.groupby(combined.index)
    
    return stats_grouped.mean(), stats_grouped.std()

def plot_individual_metrics(root_dir):
    batch_configs = {'batch_8': 8, 'batch_16': 16, 'batch_32': 32}
    batch_means = {}
    batch_stds = {}

    for b_name, b_size in batch_configs.items():
        path = os.path.join(root_dir, b_name)
        if os.path.exists(path):
            mean_df, std_df = get_batch_stats(path, b_size)
            if mean_df is not None:
                batch_means[b_name] = mean_df
                batch_stds[b_name] = std_df
    
    if not batch_means:
        print("No data found.")
        return

    global_min_rows = min(len(df) for df in batch_means.values())
    for b in batch_means:
        batch_means[b] = batch_means[b].iloc[:global_min_rows]
        batch_stds[b] = batch_stds[b].iloc[:global_min_rows]

    colors = {'batch_8': 'tab:green', 'batch_16': 'tab:blue', 'batch_32': 'tab:red'}
    
    metrics = [
        ('cpu_util_pct', 'cpu_utilization', 'CPU Utilization', 'Utilization %', 'line'),
        ('gpu_util_pct', 'gpu_utilization', 'GPU Utilization', 'Utilization %', 'line'),
        ('gpu_power', 'gpu_power', 'GPU Power Consumption', 'Watts', 'line'),
        ('gpu_reserved_mem_mb', 'gpu_memory', 'GPU Memory Allocation', 'Memory (MB)', 'line'),
        ('energy_consumed', 'energy_consumption', 'Energy Consumption', 'Energy (kWh)', 'line'),
        ('steps_per_sec', 'steps_per_sec', 'Avg Steps per Second', 'Steps / sec', 'bar'),
        ('samples_per_sec', 'samples_per_sec', 'Avg Samples per Second', 'Samples / sec', 'bar')
    ]

    # --- PNG Plotting Logic (Untouched as requested) ---
    for col, file_suffix, title, ylabel, plot_type in metrics:
        if not any(col in df.columns for df in batch_means.values()):
            continue
        plt.figure(figsize=(12, 8))
        if plot_type == 'bar':
            names, values, errs, bar_colors = [], [], [], []
            for b_name in ['batch_8', 'batch_16', 'batch_32']:
                if b_name in batch_means and col in batch_means[b_name].columns:
                    names.append(b_name.replace('_', ' ').title())
                    values.append(batch_means[b_name][col].mean())
                    errs.append(batch_stds[b_name][col].mean())
                    bar_colors.append(colors[b_name])
            plt.bar(names, values, yerr=errs, color=bar_colors, edgecolor='black', alpha=0.8, capsize=10)
            for i, v in enumerate(values):
                plt.text(i, v + (max(values)*0.02), f"{v:.2f}", ha='center', fontweight='bold', size=19)
            
        else:
            lines_added = False
            for b_name in ['batch_8', 'batch_16', 'batch_32']:
                if b_name in batch_means and col in batch_means[b_name].columns:
                    m_df, s_df = batch_means[b_name], batch_stds[b_name]
                    mu, sigma = m_df[col].copy(), s_df[col].copy()
                    if col == 'cpu_util_pct' and mu.mean() > 1.0:
                        mu /= 100.0; sigma /= 100.0
                    smooth_mu = mu.rolling(window=5, min_periods=1).mean()
                    smooth_sigma = sigma.rolling(window=5, min_periods=1).mean()
                    plt.plot(m_df.index, smooth_mu, label=b_name.replace('_', ' ').title(), color=colors[b_name])
                    plt.fill_between(m_df.index, smooth_mu - smooth_sigma, smooth_mu + smooth_sigma, color=colors[b_name], alpha=0.2)
                    lines_added = True
            plt.xlabel('Training Step')
            if lines_added: plt.legend(loc='best', frameon=True)
        plt.title(title, fontweight='bold'); plt.ylabel(ylabel); plt.ylim(bottom=0)
        if col == 'cpu_util_pct': plt.ylim(0, 0.8)
        elif col == 'gpu_util_pct': plt.ylim(0, 105)
        elif col == 'gpu_active_mem_mb': plt.ylim(0, 4500)
        elif col == 'energy_consumed': plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.grid(True, axis='y', linestyle='--', alpha=0.6); plt.tight_layout()
        plt.savefig(f'scaling_{file_suffix}.png', dpi=350); plt.close()

    # --- Updated Summary Report ---
    with open("batch_metrics_summary.txt", "w") as f:
        f.write("Averaged Scaling Analysis (7 Metrics with Std Dev)\n")
        f.write("="*70 + "\n")
        
        for b_name in ['batch_8', 'batch_16', 'batch_32']:
            if b_name in batch_means:
                m_df = batch_means[b_name]
                s_df = batch_stds[b_name]
                f.write(f"\n{b_name.upper()}:\n")
                
                # Helper to write metric lines
                def write_metric(label, col_name, unit="", is_cpu=False):
                    if col_name in m_df.columns:
                        mu = m_df[col_name].mean()
                        sigma = s_df[col_name].mean()
                        if is_cpu and mu > 1.0: 
                            mu /= 100.0; sigma /= 100.0
                        f.write(f"   - {label:<18}: {mu:>10.4f} +/- {sigma:.4f} {unit}\n")

                write_metric("Steps/sec", "steps_per_sec")
                write_metric("Samples/sec", "samples_per_sec")
                write_metric("CPU Util", "cpu_util_pct", unit="(Factor)", is_cpu=True)
                write_metric("GPU Util", "gpu_util_pct", unit="%")
                write_metric("GPU Power", "gpu_power", unit="W")
                write_metric("GPU Memory", "gpu_active_mem_mb", unit="MB")
                
                if 'energy_consumed' in m_df.columns:
                    # Final energy value is the last row's mean and std
                    f_mu = m_df['energy_consumed'].iloc[-1]
                    f_sigma = s_df['energy_consumed'].iloc[-1]
                    f.write(f"   - Final Energy      : {f_mu:.6e} +/- {f_sigma:.6e} kWh\n")
                
                f.write("-" * 45 + "\n")

    print("\nProcessing complete. All 7 metrics logged in batch_metrics_summary.txt.")

if __name__ == "__main__":
    plot_individual_metrics(os.getcwd())