import pandas as pd
import matplotlib.pyplot as plt
from plot import plot_smoothed, set_scientific_style

def main() -> None:
    '''
    Reads the CSV results from exp_09 and plots the test accuracy and parameter changes.
    '''
    set_scientific_style()
    manifold_std5_path = 'data/exp_09_manifold_std5.csv'
    residual_std5_path = 'data/exp_09_residual_std5.csv'

    df_man = pd.read_csv(manifold_std5_path)
    df_res = pd.read_csv(residual_std5_path)

    # Figure 1: Test Accuracy
    fig1, axs1 = plt.subplots(1, 2, figsize=(14, 6))
    
    man_color = '#1f77b4' # Scientific blue
    res_color = '#d62728' # Scientific red

    # Plot 1: Train std=0.5, Test std=0
    plot_smoothed(axs1[0], df_man['epoch'], df_man['test_acc_0'], label='Manifold', color=man_color)
    plot_smoothed(axs1[0], df_res['epoch'], df_res['test_acc_0'], label='Residual Manifold', color=res_color)
    axs1[0].set_title('Train(std=0.5) → Test(std=0.0)')
    axs1[0].set_xlabel('Epoch')
    axs1[0].set_ylabel('Test Accuracy (%)')
    axs1[0].grid(True, linestyle=':', color='#aaaaaa')
    axs1[0].legend(loc='lower right')

    # Plot 2: Train std=0.5, Test std=0.5
    plot_smoothed(axs1[1], df_man['epoch'], df_man['test_acc_5'], label='Manifold', color=man_color)
    plot_smoothed(axs1[1], df_res['epoch'], df_res['test_acc_5'], label='Residual Manifold', color=res_color)
    axs1[1].set_title('Train(std=0.5) → Test(std=0.5)')
    axs1[1].set_xlabel('Epoch')
    axs1[1].set_ylabel('Test Accuracy (%)')
    axs1[1].grid(True, linestyle=':', color='#aaaaaa')
    axs1[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('data/exp_09_test_acc.png')

    # Figure 2: Plot fc3_kappa and fc3_lambda_rate
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 6))

    # Plot fc3_kappa
    axs2[0].plot(df_man['epoch'], df_man['fc3_kappa'], label='Manifold', linestyle='-', color=man_color)
    axs2[0].plot(df_res['epoch'], df_res['fc3_kappa'], label='Residual Manifold', linestyle='-', color=res_color)
    axs2[0].set_title('Kappa Comparison (fc3)')
    axs2[0].set_xlabel('Epoch')
    axs2[0].set_ylabel('Kappa')
    axs2[0].grid(True, linestyle=':', color='#aaaaaa')
    axs2[0].legend()

    # Plot fc3_lambda_rate
    axs2[1].plot(df_man['epoch'], df_man['fc3_lambda_rate'], label='Manifold', linestyle='-', color=man_color)
    axs2[1].plot(df_res['epoch'], df_res['fc3_lambda_rate'], label='Residual Manifold', linestyle='-', color=res_color)
    axs2[1].set_title('Lambda Rate Comparison (fc3)')
    axs2[1].set_xlabel('Epoch')
    axs2[1].set_ylabel('Lambda Rate')
    axs2[1].grid(True, linestyle=':', color='#aaaaaa')
    axs2[1].legend()

    plt.tight_layout()
    plt.savefig('data/exp_09_fc3_params.png')
    
    # Show both figures
    plt.show()

if __name__ == '__main__':
    main()
