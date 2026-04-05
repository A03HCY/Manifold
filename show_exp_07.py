import pandas as pd
import matplotlib.pyplot as plt
from plot import set_scientific_style

def main() -> None:
    '''
    Reads the accuracy results over different noise levels from CSV and plots them.
    '''
    csv_path = 'data/exp_07_noise_acc.csv'
    
    try:
        df = pd.DataFrame(pd.read_csv(csv_path))
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please run exp_07.py first.")
        return

    set_scientific_style()
    plt.figure(figsize=(10, 6))

    base_color = '#1f77b4' # Scientific blue
    man_color = '#d62728'  # Scientific red

    # Plot the 2 lines for std=0.5 trained models
    plt.plot(df['std'], df['baseline_std5_acc'], label='Baseline ViT (trained w/ std=0.5)', color=base_color, marker='o', markersize=4)
    plt.plot(df['std'], df['manifold_std5_acc'], label='Manifold ViT (trained w/ std=0.5)', color=man_color, marker='s', markersize=4)

    plt.title('ViT Model Robustness under Gaussian Noise (CIFAR-100)')
    plt.xlabel('Gaussian Noise Std')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    save_path = 'data/exp_07_noise_acc.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()
