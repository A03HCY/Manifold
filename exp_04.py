import numpy as np
import pandas as pd
from tqdm import tqdm

from init import *
from manifold.data import cifar
from manifold.utils import test
from manifold.conv import ConvNetwork, RiemannianConvNetwork

# Initialize 4 models
baseline_0 = ConvNetwork().to(device)
manifold_0 = RiemannianConvNetwork().to(device)
baseline_5 = ConvNetwork().to(device)
manifold_5 = RiemannianConvNetwork().to(device)

# Load weights
print("Loading weights...")
baseline_0.load_pretrained('data/exp_03_baseline_std0.safetensors')
manifold_0.load_pretrained('data/exp_03_manifold_std0.safetensors')
baseline_5.load_pretrained('data/exp_03_baseline_std5.safetensors')
manifold_5.load_pretrained('data/exp_03_manifold_std5.safetensors')

# Define noise levels
stds = np.round(np.arange(0.0, 1.25, 0.05), 2)

results = []

print("Evaluating Conv models over different noise levels...")
for std in tqdm(stds):
    # We only need the test loader
    _, test_loader = cifar(batch_size, std=std)
    
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
csv_path = 'data/exp_04_noise_acc.csv'
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")
