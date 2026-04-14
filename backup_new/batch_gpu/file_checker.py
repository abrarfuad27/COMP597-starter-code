import os
import pandas as pd
import glob

def check_file_lengths(root_dir, threshold=400):
    batches = ['batch_8', 'batch_16', 'batch_32']
    print(f"{'BATCH':<10} | {'FILE':<10} | {'STEPS':<10} | {'STATUS'}")
    print("-" * 50)
    
    for b in batches:
        path = os.path.join(root_dir, b)
        if not os.path.exists(path):
            continue
            
        # Look for all .csv files in the directory
        files = glob.glob(os.path.join(path, "*.csv"))
        
        for f in sorted(files):
            df = pd.read_csv(f)
            num_steps = len(df)
            file_name = os.path.basename(f)
            
            status = "UNDER 400" if num_steps < threshold else "OK"
            print(f"{b:<10} | {file_name:<10} | {num_steps:<10} | {status}")

if __name__ == "__main__":
    # Run this in your ~/COMP597-starter-code/backup_new/batch_gpu directory
    check_file_lengths(os.getcwd())
