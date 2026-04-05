from init import *
from manifold.data import cifar_100
from manifold.utils import train_and_eval
from manifold.vit import vit_cifar100

info.save('info/exp_06_sys.json')

leaning_rate = 1e-4

print(f'[exp_06.py] lr changed to {leaning_rate}')

ep = 30

# Test loader with std=0
_, test_loader_0 = cifar_100(batch_size, std=0.0)

# Train and test loader with std=0.5
train_loader_5, test_loader_5 = cifar_100(batch_size, std=0.5)

# Models
baseline = vit_cifar100().to(device)
manifold = vit_cifar100('manifold').to(device)

print(f"Baseline Params: {baseline.count_params(human_readable=True)}")
print(f"Manifold Params: {manifold.count_params(human_readable=True)}")

print(manifold.blocks[0].mlp.fc1)

# Optimizers
baseline_opt = Adam(baseline.parameters(), lr=leaning_rate)
manifold_opt = Adam(manifold.parameters(), lr=leaning_rate)

# Criterions
baseline_cri = nn.CrossEntropyLoss()
manifold_cri = nn.CrossEntropyLoss()

# --- Experiment 6: std = 0.5 ---
print("\n--- Starting Experiment 6: Baseline ViT (std=0.5) ---")
train_and_eval(
    model=baseline,
    opt=baseline_opt,
    criterion=baseline_cri,
    train_loader=train_loader_5,
    test_loader_0=test_loader_0,
    test_loader_5=test_loader_5,
    epochs=ep,
    save_csv_path='data/exp_06_baseline_std5.csv',
    device=device,
    is_manifold=False
)

print("\n--- Starting Experiment 6: Manifold ViT (std=0.5) ---")
train_and_eval(
    model=manifold,
    opt=manifold_opt,
    criterion=manifold_cri,
    train_loader=train_loader_5,
    test_loader_0=test_loader_0,
    test_loader_5=test_loader_5,
    epochs=ep,
    save_csv_path='data/exp_06_manifold_std5.csv',
    device=device,
    is_manifold=True
)

# --- Save Models ---
print("\n--- Saving Models ---")
baseline.save_pretrained('data/exp_06_baseline_std5.safetensors')
manifold.save_pretrained('data/exp_06_manifold_std5.safetensors')
