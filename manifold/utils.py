import random
import torch

class RandomGaussianNoise(object):
    def __init__(self, mean=0., max_std=0.5):
        self.max_std = max_std
        self.mean = mean
        
    def __call__(self, tensor):
        current_std = random.uniform(0, self.max_std)
        
        noise = torch.randn(tensor.size()) * current_std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

def test(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device):
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        acc_loss = 1 - correct / total
        accuracy = 100 * correct / total

        return acc_loss, accuracy

def evaluate_topk(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device, topk=(1, 5)):
    '''
    Evaluates the model and computes the accuracy over the k top predictions
    for the specified values of k.
    '''
    maxk = max(topk)
    correct_k = {k: 0 for k in topk}
    total = 0
    
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(data)
                
            _, pred = logits.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            total += target.size(0)
            for k in topk:
                correct_k[k] += correct[:k].reshape(-1).float().sum(0, keepdim=True).item()

    res = {f'top{k}': 100 * correct_k[k] / total for k in topk}
    return res

import pandas as pd
from tqdm import tqdm

def train_and_eval(model, opt, criterion, train_loader, test_loader_0, test_loader_5, epochs, save_csv_path, device, is_manifold=False):
    '''
    Trains and evaluates a model, saving metrics to a CSV file.
    
    Args:
        model: The PyTorch model to train.
        opt: The optimizer.
        criterion: The loss function.
        train_loader: DataLoader for training data.
        test_loader_0: DataLoader for testing data (std=0).
        test_loader_5: DataLoader for testing data (std=0.5).
        epochs: Number of epochs to train.
        save_csv_path: Path to save the output CSV.
        device: The device to run training and evaluation on.
        is_manifold: Whether the model is a manifold model needing extra parameter tracking.
    '''
    results = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            opt.zero_grad()
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = model(data)
                loss = criterion(output, target)
                
                if is_manifold:
                    loss += model.manifold_loss
                
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
            
        avg_train_loss = total_loss / len(train_loader)
        test_acc_loss_0, test_acc_0 = test(model, test_loader_0, device)
        test_acc_loss_5, test_acc_5 = test(model, test_loader_5, device)
        
        row = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss_0': test_acc_loss_0,
            'test_acc_0': test_acc_0,
            'test_loss_5': test_acc_loss_5,
            'test_acc_5': test_acc_5
        }
        
        if is_manifold:
            row['total_manifold_loss'] = model.manifold_loss.item()
            # Record manifold parameters by scanning modules dynamically
            layer_idx = 1
            for name, layer in model.named_modules():
                if hasattr(layer, 'kappa') and hasattr(layer, 'lambda_rate') and hasattr(layer, 'compute_loss'):
                    # Use the module name if available, else generic layer_idx
                    prefix = f'layer{layer_idx}' if not name else name.replace('.', '_')
                    row[f'{prefix}_kappa'] = layer.kappa.item()
                    row[f'{prefix}_lambda_rate'] = layer.lambda_rate.item()
                    m_loss = layer.compute_loss()
                    row[f'{prefix}_loss_cos'] = m_loss.cosine.item()
                    row[f'{prefix}_loss_lap'] = m_loss.laplacian.item()
                    layer_idx += 1
                
        results.append(row)
        print(f'Test Acc(std=0): {test_acc_0:.2f}% | Test Acc(std=0.5): {test_acc_5:.2f}% | Avg Train Loss: {avg_train_loss:.4f}\n')
        
    df = pd.DataFrame(results)
    df.to_csv(save_csv_path, index=False)
    print(f'Saved results to {save_csv_path}')
