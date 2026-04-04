import torch

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
    JIT = True
except ImportError:
    JIT = False

if JIT:
    @triton.jit
    def manifold_conv_fuse_kernel_forward(
        c_ptr, scale_ptr, bias_ptr, out_ptr,
        kappa_val, lambda_val,
        n_elements, C, SPATIAL_SIZE,
        BLOCK_SIZE: tl.constexpr, RULE_IS_NEAR: tl.constexpr
    ):
        '''
        Forward pass fusion kernel.
        Fuses clamp, acos, exp, attraction logic, and cosine projection for convolutions.
        '''
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Determine which feature (channel) this thread is computing
        feat_idx = (offsets // SPATIAL_SIZE) % C

        # Load data from HBM to SRAM
        c = tl.load(c_ptr + offsets, mask=mask)
        scale = tl.load(scale_ptr + feat_idx, mask=mask)
        bias = tl.load(bias_ptr + feat_idx, mask=mask)

        # 1. Clamp to prevent acos NaN
        c_clamp = tl.maximum(c, -1.0 + 1e-6)
        c_clamp = tl.minimum(c_clamp, 1.0 - 1e-6)

        # 2. Angle and Gravitational Field Calculation
        theta = libdevice.acos(c_clamp)
        exp_val = tl.exp(kappa_val * (c_clamp - 1.0))

        if RULE_IS_NEAR:
            attraction = exp_val
        else:
            attraction = 1.0 - exp_val

        # 3. Geodesic Pullback
        safe_lambda = tl.maximum(lambda_val, 1e-6)
        safe_lambda = tl.minimum(safe_lambda, 1.0 - 1e-4)

        effective_theta = theta * (1.0 - safe_lambda * attraction)
        
        # 4. Output generation
        out = scale * tl.cos(effective_theta) + bias

        # Store to HBM
        tl.store(out_ptr + offsets, out, mask=mask)


    class ManifoldConvFuseFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, cosine, kappa, lambda_rate, scale, bias, rule):
            cosine = cosine.contiguous()
            scale = scale.contiguous()
            bias = bias.contiguous()

            out = torch.empty_like(cosine)
            n_elements = cosine.numel()
            
            C = cosine.size(1)
            SPATIAL_SIZE = cosine[0, 0].numel()

            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            rule_is_near = (rule == 'near')

            manifold_conv_fuse_kernel_forward[grid](
                cosine, scale, bias, out,
                kappa.item(), lambda_rate.item(),
                n_elements, C, SPATIAL_SIZE,
                BLOCK_SIZE=1024, RULE_IS_NEAR=rule_is_near
            )

            ctx.save_for_backward(cosine, kappa, lambda_rate, scale, bias)
            ctx.rule = rule
            return out

        @staticmethod
        def backward(ctx, grad_out):
            cosine, kappa, lambda_rate, scale, bias = ctx.saved_tensors
            rule = ctx.rule
            grad_out = grad_out.contiguous()

            # Using PyTorch native autograd for backward pass
            with torch.enable_grad():
                cosine_req = cosine.detach().requires_grad_(True)
                kappa_req = kappa.detach().requires_grad_(True)
                lambda_req = lambda_rate.detach().requires_grad_(True)
                scale_req = scale.detach().requires_grad_(True)
                bias_req = bias.detach().requires_grad_(True)

                c_clamp = torch.clamp(cosine_req, -1.0 + 1e-6, 1.0 - 1e-6)
                theta = torch.acos(c_clamp)
                exp_val = torch.exp(kappa_req * (c_clamp - 1.0))

                if rule == 'near':
                    attraction = exp_val
                else:
                    attraction = 1.0 - exp_val

                safe_lambda = torch.clamp(lambda_req, 1e-6, 1.0 - 1e-4)
                effective_theta = theta * (1.0 - safe_lambda * attraction)

                view_shape = [1, scale_req.size(0)] + [1] * (cosine_req.dim() - 2)
                out = scale_req.view(*view_shape) * torch.cos(effective_theta) + bias_req.view(*view_shape)

                out.backward(grad_out)

                return cosine_req.grad, kappa_req.grad, lambda_req.grad, scale_req.grad, bias_req.grad, None
