from init import *
from manifold.data import mnist
from manifold.utils import train_and_eval

info.save('info/exp_01_sys.json')

ep = 15

# with std=0
train_loader_0, test_loader_0 = mnist(batch_size, std=0.0)

# with std=0.5
train_loader_5, test_loader_5 = mnist(batch_size, std=0.5)


# --- Experiment 1: std = 0.0 ---
print("\n--- Starting Experiment 1: Baseline (std=0.0) ---")
train_and_eval(
    model=baseline_linear_model,
    opt=baseline_linear_opt,
    criterion=baseline_linear_cri,
    train_loader=train_loader_0,
    test_loader_0=test_loader_0,
    test_loader_5=test_loader_5,
    epochs=ep,
    save_csv_path='data/exp_01_baseline_std0.csv',
    device=device,
    is_manifold=False
)

print("\n--- Starting Experiment 1: Manifold (std=0.0) ---")
train_and_eval(
    model=manifold_linear_model,
    opt=manifold_linear_opt,
    criterion=manifold_linear_cri,
    train_loader=train_loader_0,
    test_loader_0=test_loader_0,
    test_loader_5=test_loader_5,
    epochs=ep,
    save_csv_path='data/exp_01_manifold_std0.csv',
    device=device,
    is_manifold=True
)

# --- Experiment 2: std = 0.5 ---
# Create fresh models and optimizers for the second experiment to ensure independence
from manifold.linear import LinearNetwork, ManifoldLinear

baseline_linear_model_5 = LinearNetwork().to(device)
manifold_linear_model_5 = ManifoldLinear().to(device)

baseline_linear_opt_5 = Adam(baseline_linear_model_5.parameters(), lr=leaning_rate)
manifold_linear_opt_5 = Adam(manifold_linear_model_5.parameters(), lr=leaning_rate)

baseline_linear_cri_5 = nn.CrossEntropyLoss()
manifold_linear_cri_5 = nn.CrossEntropyLoss()

print("\n--- Starting Experiment 2: Baseline (std=0.5) ---")
train_and_eval(
    model=baseline_linear_model_5,
    opt=baseline_linear_opt_5,
    criterion=baseline_linear_cri_5,
    train_loader=train_loader_5,
    test_loader_0=test_loader_0,
    test_loader_5=test_loader_5,
    epochs=ep,
    save_csv_path='data/exp_01_baseline_std5.csv',
    device=device,
    is_manifold=False
)

print("\n--- Starting Experiment 2: Manifold (std=0.5) ---")
train_and_eval(
    model=manifold_linear_model_5,
    opt=manifold_linear_opt_5,
    criterion=manifold_linear_cri_5,
    train_loader=train_loader_5,
    test_loader_0=test_loader_0,
    test_loader_5=test_loader_5,
    epochs=ep,
    save_csv_path='data/exp_01_manifold_std5.csv',
    device=device,
    is_manifold=True
)

# --- Save Models ---
print("\n--- Saving Models ---")
baseline_linear_model.save_pretrained('data/exp_01_baseline_std0.safetensors')
manifold_linear_model.save_pretrained('data/exp_01_manifold_std0.safetensors')
baseline_linear_model_5.save_pretrained('data/exp_01_baseline_std5.safetensors')
manifold_linear_model_5.save_pretrained('data/exp_01_manifold_std5.safetensors')
