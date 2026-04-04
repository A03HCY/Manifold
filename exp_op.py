import torch
import torch.nn as nn
import numpy as np
from typing import Callable
from manifold.ops import riemannian_manifold_linear, riemannian_manifold_conv2d

def benchmark(name: str, forward_func: Callable, backward_func: Callable, num_warmup: int = 10, num_iters: int = 100, num_trials: int = 10) -> None:
    '''
    Benchmarks the forward and backward passes over multiple trials to compute mean and standard deviation.

    Args:
        name (str): The name of the method being benchmarked.
        forward_func (Callable): Function to execute the forward pass.
        backward_func (Callable): Function to execute the backward pass.
        num_warmup (int): Number of warmup iterations.
        num_iters (int): Number of benchmark iterations per trial.
        num_trials (int): Number of trials to compute statistics.
    '''
    # Warmup
    for _ in range(num_warmup):
        out = forward_func()
        backward_func(out)
    
    torch.cuda.synchronize()
    
    fwd_times = []
    bwd_times = []
    tot_times = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(num_trials):
        # Measure forward
        start_event.record()
        for _ in range(num_iters):
            out = forward_func()
        end_event.record()
        torch.cuda.synchronize()
        fwd_times.append(start_event.elapsed_time(end_event) / num_iters)
        
        # Measure backward
        out = forward_func()
        start_event.record()
        for _ in range(num_iters):
            backward_func(out)
        end_event.record()
        torch.cuda.synchronize()
        bwd_times.append(start_event.elapsed_time(end_event) / num_iters)
        
        # Measure total
        start_event.record()
        for _ in range(num_iters):
            out = forward_func()
            backward_func(out)
        end_event.record()
        torch.cuda.synchronize()
        tot_times.append(start_event.elapsed_time(end_event) / num_iters)
    
    fwd_mean, fwd_std = np.mean(fwd_times), np.std(fwd_times)
    bwd_mean, bwd_std = np.mean(bwd_times), np.std(bwd_times)
    tot_mean, tot_std = np.mean(tot_times), np.std(tot_times)
    
    # Measure memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = forward_func()
    fwd_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    torch.cuda.reset_peak_memory_stats()
    backward_func(out)
    bwd_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    fwd_str = f"{fwd_mean:.3f} ± {fwd_std:.3f}"
    bwd_str = f"{bwd_mean:.3f} ± {bwd_std:.3f}"
    tot_str = f"{tot_mean:.3f} ± {tot_std:.3f}"
    
    print(f'{name:20s} | {fwd_str:17s} | {bwd_str:17s} | {tot_str:17s} | {fwd_mem:10.1f} | {bwd_mem:10.1f}')

def main() -> None:
    '''
    Main execution script for benchmarking.
    '''
    if not torch.cuda.is_available():
        print('CUDA is not available. Please run on a GPU.')
        return

    device = torch.device('cuda')
    
    # Dimensions
    batch_size = 8192
    in_features = 1024
    out_features = 1024
    
    print(f'Benchmarking with batch_size={batch_size}, in_features={in_features}, out_features={out_features}\n')
    
    # Tensors
    x = torch.randn(batch_size, in_features, device=device)
    w = torch.randn(out_features, in_features, device=device)
    b = torch.randn(out_features, device=device)
    
    x.requires_grad_(True)
    w.requires_grad_(True)
    b.requires_grad_(True)
    
    # Manifold specific
    kappa = torch.tensor(1.0, device=device, requires_grad=True)
    lambda_rate = torch.tensor(0.5, device=device, requires_grad=True)
    scale = torch.randn(out_features, device=device, requires_grad=True)
    bias = torch.randn(out_features, device=device, requires_grad=True)
    
    grad_out = torch.randn(batch_size, out_features, device=device)
    
    # ------------------
    # Correctness Check
    # ------------------
    # PyTorch Manifold
    out_pt = riemannian_manifold_linear(x, w, kappa, lambda_rate, scale, bias, rule='near', op='pytorch')
    out_pt.backward(grad_out, retain_graph=True)
    
    grad_x_pt = x.grad.clone() if x.grad is not None else None
    grad_w_pt = w.grad.clone() if w.grad is not None else None
    grad_k_pt = kappa.grad.clone() if kappa.grad is not None else None
    grad_l_pt = lambda_rate.grad.clone() if lambda_rate.grad is not None else None
    grad_s_pt = scale.grad.clone() if scale.grad is not None else None
    grad_b_pt = bias.grad.clone() if bias.grad is not None else None
    
    x.grad.zero_()
    w.grad.zero_()
    kappa.grad.zero_()
    lambda_rate.grad.zero_()
    scale.grad.zero_()
    bias.grad.zero_()
    
    # Triton Manifold
    out_tr = riemannian_manifold_linear(x, w, kappa, lambda_rate, scale, bias, rule='near', op='triton')
    out_tr.backward(grad_out, retain_graph=True)
    
    grad_x_tr = x.grad.clone() if x.grad is not None else None
    grad_w_tr = w.grad.clone() if w.grad is not None else None
    grad_k_tr = kappa.grad.clone() if kappa.grad is not None else None
    grad_l_tr = lambda_rate.grad.clone() if lambda_rate.grad is not None else None
    grad_s_tr = scale.grad.clone() if scale.grad is not None else None
    grad_b_tr = bias.grad.clone() if bias.grad is not None else None
    
    print('Correctness Check (Max Abs Diff):')
    print(f"Output: {(out_pt - out_tr).abs().max().item():.6e}")
    if grad_x_pt is not None and grad_x_tr is not None:
        print(f"Grad X: {(grad_x_pt - grad_x_tr).abs().max().item():.6e}")
    if grad_w_pt is not None and grad_w_tr is not None:
        print(f"Grad W: {(grad_w_pt - grad_w_tr).abs().max().item():.6e}")
    if grad_k_pt is not None and grad_k_tr is not None:
        print(f"Grad Kappa: {(grad_k_pt - grad_k_tr).abs().max().item():.6e}")
    if grad_l_pt is not None and grad_l_tr is not None:
        print(f"Grad Lambda: {(grad_l_pt - grad_l_tr).abs().max().item():.6e}")
    if grad_s_pt is not None and grad_s_tr is not None:
        print(f"Grad Scale: {(grad_s_pt - grad_s_tr).abs().max().item():.6e}")
    if grad_b_pt is not None and grad_b_tr is not None:
        print(f"Grad Bias: {(grad_b_pt - grad_b_tr).abs().max().item():.6e}")
    print()

    if x.grad is not None: x.grad.zero_()
    if w.grad is not None: w.grad.zero_()
    if kappa.grad is not None: kappa.grad.zero_()
    if lambda_rate.grad is not None: lambda_rate.grad.zero_()
    if scale.grad is not None: scale.grad.zero_()
    if bias.grad is not None: bias.grad.zero_()

    # ------------------
    # Benchmark
    # ------------------
    print('-' * 115)
    print(f"{'Method':20s} | {'Forward (ms)':17s} | {'Backward (ms)':17s} | {'Total (ms)':17s} | {'Fwd Mem(MB)':10s} | {'Bwd Mem(MB)':10s}")
    print('-' * 115)

    # 1. nn.Linear (Baseline)
    linear = nn.Linear(in_features, out_features, device=device)
    linear.weight.data.copy_(w)
    linear.bias.data.copy_(b)
    
    def fwd_nn() -> torch.Tensor:
        return linear(x)
    def bwd_nn(out: torch.Tensor) -> None:
        out.backward(grad_out, retain_graph=True)
        
    benchmark('nn.Linear', fwd_nn, bwd_nn)
    
    # 2. PyTorch Manifold Linear
    def fwd_pt() -> torch.Tensor:
        return riemannian_manifold_linear(x, w, kappa, lambda_rate, scale, bias, rule='near', op='pytorch')
    def bwd_pt(out: torch.Tensor) -> None:
        out.backward(grad_out, retain_graph=True)
        
    benchmark('Manifold L (PyTorch)', fwd_pt, bwd_pt)
    
    # 3. Triton Manifold Linear
    def fwd_tr() -> torch.Tensor:
        return riemannian_manifold_linear(x, w, kappa, lambda_rate, scale, bias, rule='near', op='triton')
    def bwd_tr(out: torch.Tensor) -> None:
        out.backward(grad_out, retain_graph=True)
        
    benchmark('Manifold L (Triton)', fwd_tr, bwd_tr)
    print('-' * 115)

    # =========================================================================
    # CONV BENCHMARK
    # =========================================================================
    
    conv_batch = 128
    in_channels = 64
    out_channels = 128
    h, w_img = 32, 32
    kernel_size = 3
    padding = 1
    
    print(f'\nBenchmarking Conv with batch={conv_batch}, in_ch={in_channels}, out_ch={out_channels}, hw={h}x{w_img}, k={kernel_size}\n')
    
    cx = torch.randn(conv_batch, in_channels, h, w_img, device=device)
    cw = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device)
    cb = torch.randn(out_channels, device=device)
    cw_ones = torch.ones(1, in_channels, kernel_size, kernel_size, device=device)
    
    cx.requires_grad_(True)
    cw.requires_grad_(True)
    cb.requires_grad_(True)
    
    c_kappa = torch.tensor(1.0, device=device, requires_grad=True)
    c_lambda = torch.tensor(0.5, device=device, requires_grad=True)
    c_scale = torch.randn(out_channels, device=device, requires_grad=True)
    c_bias = torch.randn(out_channels, device=device, requires_grad=True)
    
    c_grad_out = torch.randn(conv_batch, out_channels, h, w_img, device=device)
    
    # Correctness Check Conv
    out_c_pt = riemannian_manifold_conv2d(cx, cw, cw_ones, c_kappa, c_lambda, c_scale, c_bias, padding=padding, op='pytorch')
    out_c_pt.backward(c_grad_out, retain_graph=True)
    
    c_grad_x_pt = cx.grad.clone() if cx.grad is not None else None
    c_grad_w_pt = cw.grad.clone() if cw.grad is not None else None
    
    cx.grad.zero_()
    cw.grad.zero_()
    c_kappa.grad.zero_()
    c_lambda.grad.zero_()
    c_scale.grad.zero_()
    c_bias.grad.zero_()
    
    out_c_tr = riemannian_manifold_conv2d(cx, cw, cw_ones, c_kappa, c_lambda, c_scale, c_bias, padding=padding, op='triton')
    out_c_tr.backward(c_grad_out, retain_graph=True)
    
    c_grad_x_tr = cx.grad.clone() if cx.grad is not None else None
    c_grad_w_tr = cw.grad.clone() if cw.grad is not None else None
    
    print('Correctness Check Conv (Max Abs Diff):')
    print(f"Output: {(out_c_pt - out_c_tr).abs().max().item():.6e}")
    if c_grad_x_pt is not None and c_grad_x_tr is not None:
        print(f"Grad X: {(c_grad_x_pt - c_grad_x_tr).abs().max().item():.6e}")
    if c_grad_w_pt is not None and c_grad_w_tr is not None:
        print(f"Grad W: {(c_grad_w_pt - c_grad_w_tr).abs().max().item():.6e}")
    print()

    # Benchmark Conv
    print('-' * 115)
    print(f"{'Method':20s} | {'Forward (ms)':17s} | {'Backward (ms)':17s} | {'Total (ms)':17s} | {'Fwd Mem(MB)':10s} | {'Bwd Mem(MB)':10s}")
    print('-' * 115)
    
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, device=device)
    conv_layer.weight.data.copy_(cw)
    conv_layer.bias.data.copy_(cb)
    
    def fwd_c_nn() -> torch.Tensor:
        return conv_layer(cx)
    def bwd_c_nn(out: torch.Tensor) -> None:
        out.backward(c_grad_out, retain_graph=True)
        
    benchmark('nn.Conv2d', fwd_c_nn, bwd_c_nn)
    
    def fwd_c_pt() -> torch.Tensor:
        return riemannian_manifold_conv2d(cx, cw, cw_ones, c_kappa, c_lambda, c_scale, c_bias, padding=padding, op='pytorch')
    def bwd_c_pt(out: torch.Tensor) -> None:
        out.backward(c_grad_out, retain_graph=True)
        
    benchmark('Manifold C (PyTorch)', fwd_c_pt, bwd_c_pt)
    
    def fwd_c_tr() -> torch.Tensor:
        return riemannian_manifold_conv2d(cx, cw, cw_ones, c_kappa, c_lambda, c_scale, c_bias, padding=padding, op='triton')
    def bwd_c_tr(out: torch.Tensor) -> None:
        out.backward(c_grad_out, retain_graph=True)
        
    benchmark('Manifold C (Triton)', fwd_c_tr, bwd_c_tr)
    print('-' * 115)


if __name__ == '__main__':
    main()
