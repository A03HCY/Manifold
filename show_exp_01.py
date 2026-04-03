import pandas as pd
import matplotlib.pyplot as plt
from config import set_scientific_style, plot_smoothed

def main() -> None:
    '''
    Reads the CSV results from exp_01 and plots the test accuracy.
    '''
    set_scientific_style()
    baseline_std0_path = 'data/exp_01_baseline_std0.csv'
    manifold_std0_path = 'data/exp_01_manifold_std0.csv'
    baseline_std5_path = 'data/exp_01_baseline_std5.csv'
    manifold_std5_path = 'data/exp_01_manifold_std5.csv'

    df_base_0 = pd.read_csv(baseline_std0_path)
    df_man_0 = pd.read_csv(manifold_std0_path)
    df_base_5 = pd.read_csv(baseline_std5_path)
    df_man_5 = pd.read_csv(manifold_std5_path)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    base_color = '#1f77b4' # Scientific blue
    man_color = '#d62728'  # Scientific red

    # Plot 1: Train std=0, Test std=0
    plot_smoothed(axs[0, 0], df_base_0['epoch'], df_base_0['test_acc_0'], label='Baseline', color=base_color)
    plot_smoothed(axs[0, 0], df_man_0['epoch'], df_man_0['test_acc_0'], label='Manifold', color=man_color)
    axs[0, 0].set_title('Train(std=0.0) -> Test(std=0.0)')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Test Accuracy (%)')
    axs[0, 0].grid(True, linestyle=':', alpha=0.6, color='#aaaaaa')
    axs[0, 0].legend(loc='lower right')

    # Plot 2: Train std=0, Test std=0.5
    plot_smoothed(axs[0, 1], df_base_0['epoch'], df_base_0['test_acc_5'], label='Baseline', color=base_color)
    plot_smoothed(axs[0, 1], df_man_0['epoch'], df_man_0['test_acc_5'], label='Manifold', color=man_color)
    axs[0, 1].set_title('Train(std=0.0) -> Test(std=0.5)')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Test Accuracy (%)')
    axs[0, 1].grid(True, linestyle=':', alpha=0.6, color='#aaaaaa')
    axs[0, 1].legend(loc='lower right')

    # Plot 3: Train std=0.5, Test std=0
    plot_smoothed(axs[1, 0], df_base_5['epoch'], df_base_5['test_acc_0'], label='Baseline', color=base_color)
    plot_smoothed(axs[1, 0], df_man_5['epoch'], df_man_5['test_acc_0'], label='Manifold', color=man_color)
    axs[1, 0].set_title('Train(std=0.5) -> Test(std=0.0)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Test Accuracy (%)')
    axs[1, 0].grid(True, linestyle=':', alpha=0.6, color='#aaaaaa')
    axs[1, 0].legend(loc='lower right')

    # Plot 4: Train std=0.5, Test std=0.5
    plot_smoothed(axs[1, 1], df_base_5['epoch'], df_base_5['test_acc_5'], label='Baseline', color=base_color)
    plot_smoothed(axs[1, 1], df_man_5['epoch'], df_man_5['test_acc_5'], label='Manifold', color=man_color)
    axs[1, 1].set_title('Train(std=0.5) -> Test(std=0.5)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Test Accuracy (%)')
    axs[1, 1].grid(True, linestyle=':', alpha=0.6, color='#aaaaaa')
    axs[1, 1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('data/exp_01_test_acc.png')

    # Figure 2: Plot fc3_kappa and fc3_lambda_rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    color_std0 = '#1f77b4' # Scientific blue
    color_std5 = '#ff7f0e' # Scientific orange

    # Plot fc3_kappa
    ax1.plot(df_man_0['epoch'], df_man_0['fc3_kappa'], label='Manifold (std=0.0)', linestyle='-', color=color_std0)
    ax1.plot(df_man_5['epoch'], df_man_5['fc3_kappa'], label='Manifold (std=0.5)', linestyle='--', color=color_std5)
    ax1.set_title('Kappa Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Kappa')
    ax1.grid(True, linestyle=':', alpha=0.6, color='#aaaaaa')
    ax1.legend()

    # Plot fc3_lambda_rate
    ax2.plot(df_man_0['epoch'], df_man_0['fc3_lambda_rate'], label='Manifold (std=0.0)', linestyle='-', color=color_std0)
    ax2.plot(df_man_5['epoch'], df_man_5['fc3_lambda_rate'], label='Manifold (std=0.5)', linestyle='--', color=color_std5)
    ax2.set_title('Lambda Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Lambda')
    ax2.grid(True, linestyle=':', alpha=0.6, color='#aaaaaa')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('data/exp_01_fc3_params.png')
    
    # Show both figures
    plt.show()

if __name__ == '__main__':
    main()
