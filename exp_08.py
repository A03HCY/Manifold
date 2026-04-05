import numpy as np
import pandas as pd
from tqdm import tqdm

from init import *
from manifold.data import cifar_100
from manifold.utils import evaluate_topk
from manifold.vit import vit_cifar100

def main() -> None:
    # Initialize the 2 models from exp_06
    baseline_5 = vit_cifar100().to(device)
    manifold_5 = vit_cifar100('manifold').to(device)

    # Load weights from experiment 06
    print("Loading weights from exp_06...")
    try:
        baseline_5.load_pretrained('data/exp_06_baseline_std5.safetensors')
        manifold_5.load_pretrained('data/exp_06_manifold_std5.safetensors')
    except Exception as e:
        print(f"Error loading weights: {e}. Please ensure exp_06.py has been run and models are saved.")
        return

    # Define Gaussian noise levels
    stds = np.round(np.arange(0.0, 1.25, 0.05), 2)

    results = []

    print("Evaluating models over different Gaussian noise levels on CIFAR-100...")
    for std in tqdm(stds):
        # Get test loader for CIFAR-100 with the specified std
        _, test_loader = cifar_100(batch_size, std=std)
        
        # Evaluate models (Top-1 and Top-5)
        res_b5 = evaluate_topk(baseline_5, test_loader, device, topk=(1, 5))
        res_m5 = evaluate_topk(manifold_5, test_loader, device, topk=(1, 5))
        
        results.append({
            'std': std,
            'baseline_std5_top1_acc': res_b5['top1'],
            'baseline_std5_top5_acc': res_b5['top5'],
            'manifold_std5_top1_acc': res_m5['top1'],
            'manifold_std5_top5_acc': res_m5['top5']
        })

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = 'data/exp_08_noise_acc.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    main()
