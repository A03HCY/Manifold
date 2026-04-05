import pandas as pd
import matplotlib.pyplot as plt
from plot import set_scientific_style

def main() -> None:
    '''
    Reads the accuracy results over different noise levels from CSV and plots them for exp_04.
    '''
    set_scientific_style()
    csv_path = 'data/exp_04_noise_acc.csv'
    
    try:
        df = pd.DataFrame(pd.read_csv(csv_path))
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please run exp_04.py first.")
        return

    plt.figure(figsize=(10, 6))

    # Plot the 4 lines
    plt.plot(df['std'], df['baseline_std0_acc'], marker='o', label='Baseline Conv (trained w/ std=0.0)')
    plt.plot(df['std'], df['manifold_std0_acc'], marker='s', label='Manifold Conv (trained w/ std=0.0)')
    plt.plot(df['std'], df['baseline_std5_acc'], marker='^', label='Baseline Conv (trained w/ std=0.5)')
    plt.plot(df['std'], df['manifold_std5_acc'], marker='d', label='Manifold Conv (trained w/ std=0.5)')

    plt.title('Conv Model Robustness under Gaussian Noise')
    plt.xlabel('Gaussian Noise Std')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    save_path = 'data/exp_04_noise_acc.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()
