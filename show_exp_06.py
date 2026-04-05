import pandas as pd
import matplotlib.pyplot as plt
from plot import set_scientific_style, plot_smoothed

def main() -> None:
    '''
    Reads the CSV results from exp_06 and plots the test accuracy.
    '''
    set_scientific_style()
    baseline_std5_path = 'data/exp_06_baseline_std5.csv'
    manifold_std5_path = 'data/exp_06_manifold_std5.csv'

    df_base_5 = pd.read_csv(baseline_std5_path)
    df_man_5 = pd.read_csv(manifold_std5_path)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    base_color = '#1f77b4' # Scientific blue
    man_color = '#d62728'  # Scientific red

    # Plot 1: Train std=0.5, Test std=0
    plot_smoothed(axs[0], df_base_5['epoch'], df_base_5['test_acc_0'], label='Baseline ViT', color=base_color)
    plot_smoothed(axs[0], df_man_5['epoch'], df_man_5['test_acc_0'], label='Manifold ViT', color=man_color)
    axs[0].set_title('Train(std=0.5) -> Test(std=0.0)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Test Accuracy (%)')
    axs[0].grid(True, linestyle=':', color='#aaaaaa')
    axs[0].legend(loc='lower right')

    # Plot 2: Train std=0.5, Test std=0.5
    plot_smoothed(axs[1], df_base_5['epoch'], df_base_5['test_acc_5'], label='Baseline ViT', color=base_color)
    plot_smoothed(axs[1], df_man_5['epoch'], df_man_5['test_acc_5'], label='Manifold ViT', color=man_color)
    axs[1].set_title('Train(std=0.5) -> Test(std=0.5)')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Test Accuracy (%)')
    axs[1].grid(True, linestyle=':', color='#aaaaaa')
    axs[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('data/exp_06_test_acc.png')
    
    # Show figure
    plt.show()

if __name__ == '__main__':
    main()
