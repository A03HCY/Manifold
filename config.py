from codon.utils.seed import seed_everything
from codon.utils.info import get_system_info
import os

if not os.path.exists('info') or not os.path.isdir('info'): os.mkdir('info')
if not os.path.exists('data') or not os.path.isdir('data'): os.mkdir('data')

def input_seed(prompt, default):
    user_input = input(f'{prompt} [{default}]: ').strip()
    return default if user_input == '' else float(user_input) if '.' in str(default) else int(user_input)

seed = input_seed('[config.py] set experiment seed', 42)
seed_everything(seed=seed, strict=True, verbose=False)
info = get_system_info(manual_seed=seed)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
leaning_rate = 1e-3

print(f'[config.py] bs: {batch_size}, lr: {leaning_rate}')

import matplotlib as mpl

def set_scientific_style():
    """Sets a scientific publication style for matplotlib."""
    pass

def plot_smoothed(ax, x, y, label, color, window=3):
    '''
    Plots original data with low alpha and smoothed data on top.
    '''
    # Original data
    ax.plot(x, y, alpha=0.3, color=color, linestyle='--')
    # Smoothed data
    y_smooth = y.rolling(window=window, min_periods=1).mean()
    ax.plot(x, y_smooth, label=label, color=color, linestyle='-', linewidth=1)
