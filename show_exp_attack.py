import pandas as pd
import matplotlib.pyplot as plt

from plot import set_scientific_style

# Read the CSV file
df = pd.read_csv('data/exp_attack_acc.csv')

# Configure the plot style
set_scientific_style()

plt.figure(figsize=(10, 6))
plt.grid(True, linestyle='--', alpha=0.7)

# Plot accuracy curves
plt.plot(df['eps'], df['baseline_std5_acc'], 
         marker='o', linestyle='-', linewidth=2, 
         label='Baseline (std=0.5)')

plt.plot(df['eps'], df['manifold_std5_acc'], 
         marker='^', linestyle='-', linewidth=2, 
         label='Manifold (std=0.5)')

# Set labels and title
plt.xlabel('PGD Attack Epsilon (x/255)')
plt.ylabel('Accuracy (%)')
plt.title('Robustness against PGD Attack')
plt.legend()

# Save the plot
plt.tight_layout()
plt.savefig('data/exp_attack_acc.png', dpi=300)
plt.show()

print("Plot saved to data/exp_attack_acc.png")
