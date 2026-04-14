import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np

# Your established styling
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 19,
    'ytick.labelsize': 19,
    'legend.fontsize': 20,
    'lines.linewidth': 2.5,
    'figure.titlesize': 24
})

def process_all_phases(base_path="."):
    phases = ['forward', 'backward', 'optim']
    phase_keywords = {
        'forward': 'Forward pass',
        'backward': 'Backward pass',
        'optim': 'Optimisation step'
    }
    batch_folders = ["batch_8", "batch_16", "batch_32"]
    metrics = ['duration', 'emissions_rate', 'energy_consumed', 'gpu_power', 'gpu_util_pct', 'cpu_util_pct']
    
    titles = {
        'duration': 'Time (s)', 
        'emissions_rate': 'CO2 Rate (kg/s)', 
        'energy_consumed': 'Energy (kWh)',
        'gpu_power': 'GPU Power (W)', 
        'gpu_util_pct': 'GPU Util (%)', 
        'cpu_util_pct': 'CPU Util (%)'
    }

    all_data = []

    for phase in phases:
        phase_path = os.path.join(base_path, phase)
        if not os.path.exists(phase_path):
            continue

        for folder in batch_folders:
            folder_path = os.path.join(phase_path, folder)
            if not os.path.exists(folder_path):
                continue

            run_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            temp_dfs = []

            for rf in run_files:
                df = pd.read_csv(os.path.join(folder_path, rf))
                p_df = df[df['task_name'].str.contains(phase_keywords[phase], na=False)].copy()
                if p_df.empty: continue
                
                p_df['it'] = p_df['task_name'].str.extract(r'#(\d+)').astype(int)
                p_df = p_df.sort_values('it').reset_index(drop=True)
                temp_dfs.append(p_df[metrics])

            if not temp_dfs: continue

            min_rows = min(len(d) for d in temp_dfs)
            run_metrics = [d.head(min_rows).mean() for d in temp_dfs]

            run_metrics_df = pd.DataFrame(run_metrics)
            mean_vals = run_metrics_df.mean()
            std_vals = run_metrics_df.std()

            res = mean_vals.to_dict()
            res.update({f"{m}_std": std_vals[m] for m in metrics})
            
            # Clean X-axis naming: "Batch 8", etc.
            batch_num = re.search(r'\d+', folder).group()
            res['Batch'] = f"Batch {batch_num}"
            res['Phase'] = phase.capitalize()
            res['sort_val'] = int(batch_num)
            all_data.append(res)

    summary_df = pd.DataFrame(all_data).sort_values(['sort_val', 'Phase'])

    phase_colors = {'Forward': '#4e79a7', 'Backward': '#e15759', 'Optim': '#59a14f'}
    batch_labels = summary_df['Batch'].unique()
    x = np.arange(len(batch_labels))
    width = 0.25

    for m in metrics:
        plt.figure(figsize=(10, 8))
        
        for j, phase in enumerate(['Forward', 'Backward', 'Optim']):
            phase_df = summary_df[summary_df['Phase'] == phase]
            if phase_df.empty: continue
            
            pos = x + (j - 1) * width
            means = phase_df[m].values
            stds = phase_df[f"{m}_std"].values
            
            bar_container = plt.bar(pos, means, width, yerr=stds, 
                                    color=phase_colors[phase], edgecolor='black', 
                                    capsize=5, alpha=0.8, label=phase)
            
            for idx, bar in enumerate(bar_container):
                height = bar.get_height()
                
                # Metric-specific rounding logic
                if m in ['gpu_power', 'gpu_util_pct']:
                    label_text = f'{int(round(height))}'
                elif m == 'emissions_rate':
                    label_text = f'{height:.1e}'
                elif m == 'energy_consumed':
                    label_text = f'{height:.4f}'
                else:
                    label_text = f'{height:.2f}'
                
                plt.text(bar.get_x() + bar.get_width()/2, height + stds[idx] + (summary_df[m].max()*0.02),
                         label_text, ha='center', va='bottom', fontsize=14, fontweight='bold')

        plt.title(f"Phase Analysis: {titles[m]}", fontweight='bold', pad=20)
        plt.ylabel(titles[m])
        plt.xlabel('') # Removed "Batch Size" label
        plt.xticks(x, batch_labels)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
        
        plt.ylim(0, summary_df[m].max() * 1.3) # Increased headroom for integer labels
        
        plt.tight_layout()
        filename = f'phase_benchmark_{m}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {filename}")

if __name__ == "__main__":
    process_all_phases()