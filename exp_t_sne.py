import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from init import *
from manifold.data import mnist
from manifold.linear import LinearNetwork, ManifoldLinear
from plot import set_scientific_style

def extract_features(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, num_samples: int = 2000) -> tuple:
    '''
    Extracts the hidden features (output of fc2) from the model.

    Args:
        model (nn.Module): The model to extract features from.
        dataloader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): The device to run on.
        num_samples (int): Number of samples to extract.

    Returns:
        tuple: Features array and labels array.
    '''
    model.eval()
    features = []
    labels = []
    count = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            # Manual forward pass up to fc2
            x = x.view(x.size(0), -1)
            x = F.silu(model.fc1(x))
            out = F.silu(model.fc2(x))
            
            features.append(out.cpu().numpy())
            labels.append(y.numpy())
            
            count += x.size(0)
            if count >= num_samples:
                break
                
    features = np.concatenate(features, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    return features, labels

def main() -> None:
    '''
    Main function to run feature extraction and t-SNE visualization.
    '''
    set_scientific_style()
    
    print("Initializing models...")
    baseline_model = LinearNetwork().to(device)
    manifold_model = ManifoldLinear().to(device)
    
    print("Loading weights...")
    baseline_model.load_pretrained('data/exp_01_baseline_std5.safetensors')
    manifold_model.load_pretrained('data/exp_01_manifold_std5.safetensors')
    
    print("Loading data...")
    _, test_loader = mnist(batch_size, std=0.5)
    
    print("Extracting features...")
    num_samples = 2000
    base_feats, base_labels = extract_features(baseline_model, test_loader, device, num_samples)
    man_feats, man_labels = extract_features(manifold_model, test_loader, device, num_samples)
    
    print("Running t-SNE (this might take a minute)...")
    tsne = TSNE(n_components=2, random_state=42)
    base_tsne = tsne.fit_transform(base_feats)
    man_tsne = tsne.fit_transform(man_feats)
    
    print("Plotting results...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    cmap = plt.get_cmap('tab10')
    
    for i in range(10):
        # Baseline
        idx = (base_labels == i)
        ax1.scatter(base_tsne[idx, 0], base_tsne[idx, 1], color=cmap(i), label=str(i), alpha=0.6, s=15)
        # Manifold
        idx = (man_labels == i)
        ax2.scatter(man_tsne[idx, 0], man_tsne[idx, 1], color=cmap(i), label=str(i), alpha=0.6, s=15)
        
    ax1.set_title('Baseline Linear (std=0.5) t-SNE')
    ax1.legend(title='Digit', loc='best', markerscale=2)
    ax1.axis('off')
    
    ax2.set_title('Manifold Linear (std=0.5) t-SNE')
    ax2.legend(title='Digit', loc='best', markerscale=2)
    ax2.axis('off')
    
    plt.tight_layout()
    save_path = 'data/exp_t_sne_std5.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    plt.show()

if __name__ == '__main__':
    main()
