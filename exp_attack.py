import torch
import torchattacks
import numpy as np
import pandas as pd
from tqdm import tqdm

from init import *
from manifold.data import cifar, CIFAR_MEAN, CIFAR_STD
from manifold.conv import ConvNetwork, RiemannianConvNetwork

def main():
    # Initialize 2 models
    baseline_5 = ConvNetwork().to(device)
    manifold_5 = RiemannianConvNetwork().to(device)

    # Load weights
    print("Loading weights...")
    baseline_5.load_pretrained('data/exp_03_baseline_std5.safetensors')
    manifold_5.load_pretrained('data/exp_03_manifold_std5.safetensors')

    baseline_5.eval()
    manifold_5.eval()

    # We only need the test loader, standard validation (std=None)
    _, test_loader = cifar(batch_size, std=None)

    # Define denormalize function to revert images to [0, 1] range
    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
        return tensor * std + mean

    # Define normalize function to feed unnormalized images to models
    def normalize(tensor, mean, std):
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
        return (tensor - mean) / std

    # Define attack perturbation levels
    epsilons = [0.0, 1/255, 2/255, 4/255, 8/255, 12/255, 16/255, 24/255, 32/255]
    
    results = []

    print("Evaluating Conv models under PGD attack...")
    for eps in tqdm(epsilons):
        if eps == 0.0:
            alpha = 0.0
        else:
            alpha = max(eps / 4, 1/255)
            
        atk_baseline = torchattacks.PGD(baseline_5, eps=eps, alpha=alpha, steps=10, random_start=True)
        atk_baseline.set_normalization_used(CIFAR_MEAN, CIFAR_STD)
        
        atk_manifold = torchattacks.PGD(manifold_5, eps=eps, alpha=alpha, steps=10, random_start=True)
        atk_manifold.set_normalization_used(CIFAR_MEAN, CIFAR_STD)
        
        correct_b5 = 0
        correct_m5 = 0
        total = 0
        
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Denormalize data since CIFAR dataloader yields normalized images
            data_denorm = denormalize(data, CIFAR_MEAN, CIFAR_STD)
            
            # Attack
            if eps > 0.0:
                adv_data_b5 = atk_baseline(data_denorm, target)
                adv_data_m5 = atk_manifold(data_denorm, target)
            else:
                adv_data_b5 = data_denorm
                adv_data_m5 = data_denorm
                
            # Normalize adversarial examples before feeding into the models
            norm_adv_b5 = normalize(adv_data_b5, CIFAR_MEAN, CIFAR_STD)
            norm_adv_m5 = normalize(adv_data_m5, CIFAR_MEAN, CIFAR_STD)

            with torch.no_grad():
                logits_b5 = baseline_5(norm_adv_b5)
                logits_m5 = manifold_5(norm_adv_m5)
                
                _, pred_b5 = torch.max(logits_b5.data, 1)
                _, pred_m5 = torch.max(logits_m5.data, 1)
                
                correct_b5 += (pred_b5 == target).sum().item()
                correct_m5 += (pred_m5 == target).sum().item()
                total += target.size(0)
                
        acc_b5 = 100 * correct_b5 / total
        acc_m5 = 100 * correct_m5 / total
        
        results.append({
            'eps': round(eps * 255), # Save eps as integer in [0, 255] scale for readability
            'eps_float': eps,
            'baseline_std5_acc': acc_b5,
            'manifold_std5_acc': acc_m5
        })
        
    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = 'data/exp_attack_acc.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == '__main__':
    main()
