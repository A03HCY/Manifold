from init import *
from manifold.data import mnist
from manifold.utils import train_and_eval

ep = 30

# with std=0
train_loader_0, test_loader_0 = mnist(batch_size, std=0.0)

# with std=0.5
train_loader_5, test_loader_5 = mnist(batch_size, std=0.5)


from manifold.linear import ManifoldLinear, ManifoldResidualLinear

baseline_linear_model_5 = ManifoldLinear().to(device)
manifold_linear_model_5 = ManifoldResidualLinear().to(device)

baseline_linear_opt_5 = Adam(baseline_linear_model_5.parameters(), lr=leaning_rate)
manifold_linear_opt_5 = Adam(manifold_linear_model_5.parameters(), lr=leaning_rate)

baseline_linear_cri_5 = nn.CrossEntropyLoss()
manifold_linear_cri_5 = nn.CrossEntropyLoss()

print("\n--- Starting Experiment 1: Baseline (std=0.5) ---")
train_and_eval(
    model=baseline_linear_model_5,
    opt=baseline_linear_opt_5,
    criterion=baseline_linear_cri_5,
    train_loader=train_loader_5,
    test_loader_0=test_loader_0,
    test_loader_5=test_loader_5,
    epochs=ep,
    save_csv_path='data/exp_09_manifold_std5.csv',
    device=device,
    is_manifold=True
)

print("\n--- Starting Experiment 1: Manifold (std=0.5) ---")
train_and_eval(
    model=manifold_linear_model_5,
    opt=manifold_linear_opt_5,
    criterion=manifold_linear_cri_5,
    train_loader=train_loader_5,
    test_loader_0=test_loader_0,
    test_loader_5=test_loader_5,
    epochs=ep,
    save_csv_path='data/exp_09_residual_std5.csv',
    device=device,
    is_manifold=True
)

# --- Save Models ---
print("\n--- Saving Models ---")
baseline_linear_model_5.save_pretrained('data/exp_09_manifold_std5.safetensors')
manifold_linear_model_5.save_pretrained('data/exp_09_residual_std5.safetensors')
