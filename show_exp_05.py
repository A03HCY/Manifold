import pandas as pd
import matplotlib.pyplot as plt
from plot import set_scientific_style

def main() -> None:
    '''
    Reads the experiment results and plots the robustness of models under different noises.
    '''
    set_scientific_style()
    # Load data
    csv_path = 'data/exp_05_robustness.csv'
    try:
        df = pd.DataFrame(pd.read_csv(csv_path))
    except FileNotFoundError:
        print(f"File {csv_path} not found. Please run exp_05.py first.")
        return

    # Set up the plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    noise_types = ['SaltAndPepper', 'GaussianBlur', 'Uniform']
    titles = ['Salt and Pepper Noise', 'Gaussian Blur', 'Uniform Noise']
    x_labels = ['Noise Density (d)', 'Blur Sigma ($\\sigma_b$)', 'Noise Range (u)']

    for i, noise_type in enumerate(noise_types):
        ax = axes[i]
        subset = df[df['noise_type'] == noise_type]

        if not subset.empty:
            params = subset['param'].values
            acc_b = subset['baseline_acc'].values
            acc_m = subset['manifold_acc'].values

            ax.plot(params, acc_b, marker='o', label='Conv (Baseline)')
            ax.plot(params, acc_m, marker='s', label='ManifoldConv')

            ax.set_title(titles[i], fontsize=14)
            ax.set_xlabel(x_labels[i], fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--')
            
            # Set integer/float ticks appropriately
            ax.set_xticks(params)

    plt.tight_layout()
    
    # Save the plot
    save_path = 'data/exp_05_robustness.png'
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()
