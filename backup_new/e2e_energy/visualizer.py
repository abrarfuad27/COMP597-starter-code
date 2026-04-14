# #=========================================
# ## END-TO-END MEASUREMENT VISUALIZER
# #=========================================

# import matplotlib.pyplot as plt
# import numpy as np

# # Set academic style
# plt.rcParams.update({'font.size': 18, 'axes.labelsize': 16, 'axes.titlesize': 18})

# batch_sizes = ['8', '16', '32']
# x = np.arange(len(batch_sizes))

# # Data: No Measurement (User provided)
# time_no_meas = [112, 184, 284]
# time_no_meas_std = [0.5, 1.7, 2.3] # Newly added std

# # Data: With Measurement (Calculated from logs)
# time_meas_avg = [117.77, 194.84, 297.44]
# time_meas_std = [10.22, 17.06, 16.33]

# energy_avg = [0.01837, 0.03053, 0.04704]
# energy_std = [0.00160, 0.00268, 0.00259]

# emissions_avg = [7.23e-5, 1.21e-4, 1.85e-4]
# emissions_std = [6.8e-6, 1.0e-5, 1.0e-5]

# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# axes = axes.flatten()

# # 1. E2E Time Comparison - Updated with Error Bars for "No Measurement"
# width = 0.35
# axes[0].bar(x - width/2, time_no_meas, width, yerr=time_no_meas_std, 
#             label='No Measurement', color='gray', alpha=0.6, capsize=5)
# axes[0].bar(x + width/2, time_meas_avg, width, yerr=time_meas_std, 
#             label='With Measurement', color='tab:blue', capsize=5)
# axes[0].set_title('E2E Training Time')
# axes[0].set_ylabel('Time (seconds)')
# axes[0].set_xticks(x)
# axes[0].set_xticklabels(batch_sizes)
# axes[0].legend()

# # 2. Total Energy Consumed
# axes[1].bar(batch_sizes, energy_avg, yerr=energy_std, color='tab:orange', capsize=5)
# axes[1].set_title('Total Energy Consumption')
# axes[1].set_ylabel('Energy (kWh)')

# # 3. Carbon Emissions
# axes[2].bar(batch_sizes, emissions_avg, yerr=emissions_std, color='tab:green', capsize=5)
# axes[2].set_title('Carbon Footprint')
# axes[2].set_ylabel('kg CO2eq')
# axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# # 4. Measurement Overhead % (Derived)
# overhead = [( (m - n) / n) * 100 for n, m in zip(time_no_meas, time_meas_avg)]
# axes[3].bar(batch_sizes, overhead, color='tab:red')
# axes[3].set_title('Measurement Overhead (%)')
# axes[3].set_ylabel('Percentage Increase')
# axes[3].set_ylim(0, 10)

# for ax in axes:
#     ax.set_xlabel('Batch Size')
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     ax.set_ylim(bottom=0)

# plt.tight_layout()
# plt.savefig('scaling_analysis.png', dpi=300)
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Set academic style
plt.rcParams.update({
    'font.size': 22, 
    'axes.labelsize': 22, 
    'axes.titlesize': 22,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 20,
})

batch_sizes = ['8', '16', '32']
x = np.arange(len(batch_sizes))

# Data: No Measurement
time_no_meas = [112, 184, 284]
time_no_meas_std = [0.5, 1.7, 2.3]

# Data: With Measurement
time_meas_avg = [117.77, 194.84, 297.44]
time_meas_std = [10.22, 17.06, 16.33]

energy_avg = [0.01837, 0.03053, 0.04704]
energy_std = [0.00160, 0.00268, 0.00259]

emissions_avg = [7.23e-5, 1.21e-4, 1.85e-4]
emissions_std = [6.8e-6, 1.0e-5, 1.0e-5]

# Calculate Overhead Percentage
overhead = [((m - n) / n) * 100 for n, m in zip(time_no_meas, time_meas_avg)]

# --- PLOT 1: E2E Training Time (No changes here) ---
plt.figure(figsize=(10, 8))
width = 0.35
bars1 = plt.bar(x - width/2, time_no_meas, width, yerr=time_no_meas_std, 
                label='No Measurement', color='gray', alpha=0.6, capsize=5, edgecolor='black')
bars2 = plt.bar(x + width/2, time_meas_avg, width, yerr=time_meas_std, 
                label='With Measurement', color='tab:blue', capsize=5, edgecolor='black')

for i, (bar, oh) in enumerate(zip(bars2, overhead)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + time_meas_std[i] + 2, 
             f'+{oh:.1f}%', ha='center', va='bottom', fontsize=18, fontweight='bold', color='tab:red')

plt.title('End-to-End Training Time Overhead')
plt.ylabel('Time (seconds)')
plt.xlabel('Batch Size')
plt.xticks(x, batch_sizes)
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, max(time_meas_avg) * 1.2)
plt.tight_layout()
plt.savefig('e2e_time_comparison.png', dpi=300)
plt.close()

# --- PLOT 2: Total Energy Consumption (With Bar Text) ---
plt.figure(figsize=(10, 8))
bars_energy = plt.bar(batch_sizes, energy_avg, yerr=energy_std, color='tab:orange', 
                      capsize=8, edgecolor='black', alpha=0.8)

# Add value labels
for i, bar in enumerate(bars_energy):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + energy_std[i] + (max(energy_avg)*0.01), 
             f'{height:.4f}', ha='center', va='bottom', fontsize=19, fontweight='bold')

plt.title('End-to-End Total Energy Consumption')
plt.ylabel('Energy (kWh)')
plt.xlabel('Batch Size')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, max(energy_avg) * 1.2)
plt.tight_layout()
plt.savefig('total_energy.png', dpi=300)
plt.close()

# --- PLOT 3: Carbon Footprint (With Scientific Notation Bar Text) ---
plt.figure(figsize=(10, 8))
bars_emissions = plt.bar(batch_sizes, emissions_avg, yerr=emissions_std, color='tab:green', 
                         capsize=8, edgecolor='black', alpha=0.8)

# Add value labels in scientific notation
for i, bar in enumerate(bars_emissions):
    height = bar.get_height()
    # Format to scientific notation (e.g., 1.21e-04)
    plt.text(bar.get_x() + bar.get_width()/2, height + emissions_std[i] + (max(emissions_avg)*0.01), 
             f'{height:.2e}', ha='center', va='bottom', fontsize=19, fontweight='bold')

plt.title('End-to-End Carbon Footprint')
plt.ylabel('kg CO2eq')
plt.xlabel('Batch Size')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, max(emissions_avg) * 1.2)
plt.tight_layout()
plt.savefig('carbon_emissions.png', dpi=300)
plt.close()

print("Three files generated with bar values: e2e_time_comparison.png, total_energy.png, carbon_emissions.png")