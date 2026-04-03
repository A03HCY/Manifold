import pandas as pd
import matplotlib.pyplot as plt
from config import set_scientific_style

def main() -> None:
    '''
    Reads the accuracy results over different noise levels from CSV and plots them.
    '''
    csv_path = 'data/exp_02_noise_acc.csv'
    
    try:
        df = pd.DataFrame(pd.read_csv(csv_path))
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please run exp_02.py first.")
        return

    set_scientific_style()
    plt.figure(figsize=(10, 6))

    # Plot the 4 lines
    plt.plot(df['std'], df['baseline_std0_acc'], label='Baseline (trained w/ std=0.0)')
    plt.plot(df['std'], df['manifold_std0_acc'], label='Manifold (trained w/ std=0.0)')
    plt.plot(df['std'], df['baseline_std5_acc'], label='Baseline (trained w/ std=0.5)')
    plt.plot(df['std'], df['manifold_std5_acc'], label='Manifold (trained w/ std=0.5)')

    plt.title('Model Robustness under Gaussian Noise')
    plt.xlabel('Gaussian Noise Std')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    save_path = 'data/exp_02_noise_acc.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()
