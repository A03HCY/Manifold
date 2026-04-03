from init import *
from manifold.data import cifar
from manifold.utils import train_and_eval

info.save('info/exp_03_sys.json')

ep = 15

# with std=0
train_loader_0, test_loader_0 = cifar(batch_size, std=0.0)

# with std=0.5
train_loader_5, test_loader_5 = cifar(batch_size, std=0.5)


# --- Experiment 3: std = 0.0 ---
print("\n--- Starting Experiment 3: Baseline Conv (std=0.0) ---")
train_and_eval(
    model=baseline_conv_model,
    opt=baseline_conv_opt,
    criterion=baseline_conv_cri,
    train_loader=train_loader_0,
    test_loader_0=test_loader_0,
    test_loader_5=test_loader_5,
    epochs=ep,
    save_csv_path='data/exp_03_baseline_std0.csv',
    device=device,
    is_manifold=False
)

print("\n--- Starting Experiment 3: Manifold Conv (std=0.0) ---")
train_and_eval(
    model=manifold_conv_model,
    opt=manifold_conv_opt,
    criterion=manifold_conv_cri,
    train_loader=train_loader_0,
    test_loader_0=test_loader_0,
    test_loader_5=test_loader_5,
    epochs=ep,
    save_csv_path='data/exp_03_manifold_std0.csv',
    device=device,
    is_manifold=True
)

# --- Experiment 3: std = 0.5 ---
# Create fresh models and optimizers for the second experiment to ensure independence
from manifold.conv import ConvNetwork, RiemannianConvNetwork

baseline_conv_model_5 = ConvNetwork().to(device)
manifold_conv_model_5 = RiemannianConvNetwork().to(device)

baseline_conv_opt_5 = Adam(baseline_conv_model_5.parameters(), lr=leaning_rate)
manifold_conv_opt_5 = Adam(manifold_conv_model_5.parameters(), lr=leaning_rate)

baseline_conv_cri_5 = nn.CrossEntropyLoss()
manifold_conv_cri_5 = nn.CrossEntropyLoss()

print("\n--- Starting Experiment 3: Baseline Conv (std=0.5) ---")
train_and_eval(
    model=baseline_conv_model_5,
    opt=baseline_conv_opt_5,
    criterion=baseline_conv_cri_5,
    train_loader=train_loader_5,
    test_loader_0=test_loader_0,
    test_loader_5=test_loader_5,
    epochs=ep,
    save_csv_path='data/exp_03_baseline_std5.csv',
    device=device,
    is_manifold=False
)

print("\n--- Starting Experiment 3: Manifold Conv (std=0.5) ---")
train_and_eval(
    model=manifold_conv_model_5,
    opt=manifold_conv_opt_5,
    criterion=manifold_conv_cri_5,
    train_loader=train_loader_5,
    test_loader_0=test_loader_0,
    test_loader_5=test_loader_5,
    epochs=ep,
    save_csv_path='data/exp_03_manifold_std5.csv',
    device=device,
    is_manifold=True
)

# --- Save Models ---
print("\n--- Saving Models ---")
baseline_conv_model.save_pretrained('data/exp_03_baseline_std0.safetensors')
manifold_conv_model.save_pretrained('data/exp_03_manifold_std0.safetensors')
baseline_conv_model_5.save_pretrained('data/exp_03_baseline_std5.safetensors')
manifold_conv_model_5.save_pretrained('data/exp_03_manifold_std5.safetensors')
