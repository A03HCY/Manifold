import numpy as np
import pandas as pd
from tqdm import tqdm

from init import *
from manifold.data import mnist
from manifold.utils import test
from manifold.linear import LinearNetwork, ManifoldLinear

# Initialize 4 models
baseline_0 = LinearNetwork().to(device)
manifold_0 = ManifoldLinear().to(device)
baseline_5 = LinearNetwork().to(device)
manifold_5 = ManifoldLinear().to(device)

# Load weights
print("Loading weights...")
baseline_0.load_pretrained('data/exp_01_baseline_std0.safetensors')
manifold_0.load_pretrained('data/exp_01_manifold_std0.safetensors')
baseline_5.load_pretrained('data/exp_01_baseline_std5.safetensors')
manifold_5.load_pretrained('data/exp_01_manifold_std5.safetensors')

# Define noise levels
stds = np.round(np.arange(0.0, 1.25, 0.05), 2)

results = []

print("Evaluating models over different noise levels...")
for std in tqdm(stds):
    # We only need the test loader
    _, test_loader = mnist(batch_size, std=std)
    
    # Evaluate models
    _, acc_b0 = test(baseline_0, test_loader, device)
    _, acc_m0 = test(manifold_0, test_loader, device)
    _, acc_b5 = test(baseline_5, test_loader, device)
    _, acc_m5 = test(manifold_5, test_loader, device)
    
    results.append({
        'std': std,
        'baseline_std0_acc': acc_b0,
        'manifold_std0_acc': acc_m0,
        'baseline_std5_acc': acc_b5,
        'manifold_std5_acc': acc_m5
    })

# Save to CSV
df = pd.DataFrame(results)
csv_path = 'data/exp_02_noise_acc.csv'
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")
