import pandas as pd
import matplotlib.pyplot as plt
from plot import set_scientific_style

def main() -> None:
    '''
    Reads the accuracy results over different Gaussian noise levels from CSV and plots them.
    '''
    csv_path = 'data/exp_08_noise_acc.csv'
    
    try:
        df = pd.DataFrame(pd.read_csv(csv_path))
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please run exp_08.py first.")
        return

    set_scientific_style()
    
    # Create 2 subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    base_color = '#1f77b4' # Scientific blue
    man_color = '#d62728'  # Scientific red

    # Top-1 Accuracy Subplot
    ax1.plot(df['std'], df['baseline_std5_top1_acc'], label='Baseline ViT', color=base_color, marker='o', markersize=4)
    ax1.plot(df['std'], df['manifold_std5_top1_acc'], label='Manifold ViT', color=man_color, marker='s', markersize=4)
    ax1.set_title('ViT Model Robustness: Top-1 Accuracy (CIFAR-100)')
    ax1.set_ylabel('Top-1 Test Accuracy (%)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Top-5 Accuracy Subplot
    ax2.plot(df['std'], df['baseline_std5_top5_acc'], label='Baseline ViT', color=base_color, marker='o', markersize=4)
    ax2.plot(df['std'], df['manifold_std5_top5_acc'], label='Manifold ViT', color=man_color, marker='s', markersize=4)
    ax2.set_title('ViT Model Robustness: Top-5 Accuracy (CIFAR-100)')
    ax2.set_xlabel('Gaussian Noise Std')
    ax2.set_ylabel('Top-5 Test Accuracy (%)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    save_path = 'data/exp_08_noise_acc.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()
