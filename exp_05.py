import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from init import *
from manifold.data import CIFAR_MEAN, CIFAR_STD
from manifold.utils import test
from manifold.conv import ConvNetwork, RiemannianConvNetwork

class SaltAndPepperNoise:
    '''
    Applies Salt and Pepper noise to a tensor.

    Attributes:
        prob (float): The probability of replacing a pixel with salt or pepper noise.
    '''

    def __init__(self, prob: float) -> None:
        self.prob = prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.rand(tensor.size())
        tensor = tensor.clone()
        tensor[noise < self.prob / 2] = 0.0
        tensor[(noise >= self.prob / 2) & (noise < self.prob)] = 1.0
        return tensor

class UniformNoise:
    '''
    Applies Uniform noise to a tensor.

    Attributes:
        u (float): The range of uniform noise [-u, u].
    '''

    def __init__(self, u: float) -> None:
        self.u = u

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = (torch.rand(tensor.size()) * 2 - 1) * self.u
        tensor = tensor + noise
        tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor

def get_blur_transform(sigma: float) -> transforms.GaussianBlur:
    '''
    Gets a GaussianBlur transform with a dynamic kernel size.

    Args:
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        transforms.GaussianBlur: The initialized transform.
    '''
    k = int(4 * sigma) // 2 * 2 + 1
    k = max(3, k)
    return transforms.GaussianBlur(kernel_size=k, sigma=sigma)

def get_test_loader(custom_transform) -> DataLoader:
    '''
    Creates a DataLoader for CIFAR-10 test set with a specific custom transform.

    Args:
        custom_transform: The transform to apply after ToTensor and before Normalize.

    Returns:
        DataLoader: The resulting test dataloader.
    '''
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        custom_transform,
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    test_data = torchvision.datasets.CIFAR10(
        root='../../data/cifar10',
        train=False,
        download=True,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )
    return test_loader

def main() -> None:
    # Initialize models
    baseline_model = ConvNetwork().to(device)
    manifold_model = RiemannianConvNetwork().to(device)

    # Load weights trained with std=0.5
    print("Loading weights...")
    baseline_model.load_pretrained('data/exp_03_baseline_std5.safetensors')
    manifold_model.load_pretrained('data/exp_03_manifold_std5.safetensors')

    results = []

    # 1. Salt and Pepper Noise
    d_values = [0.1, 0.2, 0.3, 0.4]
    print("\nEvaluating Salt and Pepper Noise...")
    for d in tqdm(d_values):
        loader = get_test_loader(SaltAndPepperNoise(d))
        _, acc_b = test(baseline_model, loader, device)
        _, acc_m = test(manifold_model, loader, device)
        results.append({
            'noise_type': 'SaltAndPepper',
            'param': d,
            'baseline_acc': acc_b,
            'manifold_acc': acc_m
        })
        print(f"S&P (d={d}): Baseline={acc_b:.4f}, Manifold={acc_m:.4f}")

    # 2. Gaussian Blur
    sigma_values = [1.0, 2.0, 3.0]
    print("\nEvaluating Gaussian Blur...")
    for sigma in tqdm(sigma_values):
        loader = get_test_loader(get_blur_transform(sigma))
        _, acc_b = test(baseline_model, loader, device)
        _, acc_m = test(manifold_model, loader, device)
        results.append({
            'noise_type': 'GaussianBlur',
            'param': sigma,
            'baseline_acc': acc_b,
            'manifold_acc': acc_m
        })
        print(f"Blur (sigma={sigma}): Baseline={acc_b:.4f}, Manifold={acc_m:.4f}")

    # 3. Uniform Noise
    u_values = [0.2, 0.4, 0.6]
    print("\nEvaluating Uniform Noise...")
    for u in tqdm(u_values):
        loader = get_test_loader(UniformNoise(u))
        _, acc_b = test(baseline_model, loader, device)
        _, acc_m = test(manifold_model, loader, device)
        results.append({
            'noise_type': 'Uniform',
            'param': u,
            'baseline_acc': acc_b,
            'manifold_acc': acc_m
        })
        print(f"Uniform (u={u}): Baseline={acc_b:.4f}, Manifold={acc_m:.4f}")

    # Save results
    df = pd.DataFrame(results)
    csv_path = 'data/exp_05_robustness.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

if __name__ == '__main__':
    main()
