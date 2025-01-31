import math
import os
import unittest

import natten
import torch

from natten.utils.testing import (
    skip_if_acpp_is_not_supported,
    skip_if_experimental_ops_are_not_supported,
    skip_if_fvcore_is_not_available,
    skip_if_torch_flop_count_is_not_supported,
)

def _reset_everything():
    natten.use_acpp_na()
    os.environ["NATTEN_LOG_LEVEL"] = "CRITICAL"
    torch.use_deterministic_algorithms(True)  # Required for SYCL backend

class ACPPFlopCounterTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _build_model(self, dim_per_head, heads, kernel_size, dilation, 
                    is_causal, qkv_bias, use_experimental_ops, device="gfx908"):
        mod = natten.NeighborhoodAttention1D
        if hasattr(kernel_size, "__len__") and len(kernel_size) == 2:
            mod = natten.NeighborhoodAttention2D
        elif hasattr(kernel_size, "__len__") and len(kernel_size) == 3:
            mod = natten.NeighborhoodAttention3D

        return mod(
            dim=dim_per_head * heads,
            num_heads=heads,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
            use_experimental_ops=use_experimental_ops,
        ).to(device=device)

    def _build_input(self, batch, spatial_extent, heads, dim_per_head, device="gfx908"):
        shape = [batch, *spatial_extent, heads * dim_per_head]
        return torch.randn(shape, device=device)

    def _compute_flops(self, batch, spatial_extent, heads, dim_per_head, 
                      kernel_size, dilation, is_causal, qkv_bias, return_macs):
        # Reference implementation from test_flops.py
        c = 1 if return_macs else 2
        qkv_M, qkv_N, qkv_K = (
            batch * math.prod(spatial_extent),
            dim_per_head * heads * 3,
            dim_per_head * heads,
        )
        qkv_flops = qkv_M * qkv_N * qkv_K * c

        proj_M, proj_N, proj_K = (
            batch * math.prod(spatial_extent),
            dim_per_head * heads,
            dim_per_head * heads,
        )
        proj_flops = proj_M * proj_N * proj_K * c

        attn_0_M, attn_0_N, attn_0_K = (
            batch * heads * math.prod(spatial_extent),
            math.prod(kernel_size),
            dim_per_head,
        )
        attn_1_M, attn_1_N, attn_1_K = (
            batch * heads * math.prod(spatial_extent),
            dim_per_head,
            math.prod(kernel_size),
        )

        attn_0_flops = attn_0_M * attn_0_N * attn_0_K * c
        attn_1_flops = attn_1_M * attn_1_N * attn_1_K * c
        attn_flops = attn_0_flops + attn_1_flops

        return qkv_flops + attn_flops + proj_flops

    @skip_if_fvcore_is_not_available()
    @skip_if_acpp_is_not_supported()
    def test_fvcore_flops_gfx908(self):
        self._test_natten_flops_with_fvcore(
            batch=1,
            spatial_extent=(16, 10),
            heads=2,
            dim_per_head=32,
            kernel_size=(5, 3),
            dilation=(2, 1),
            is_causal=(False, False),
            qkv_bias=True,
            device="gfx908"
        )

    @skip_if_fvcore_is_not_available()
    @skip_if_acpp_is_not_supported()
    def test_fvcore_flops_gfx90a(self):
        self._test_natten_flops_with_fvcore(
            batch=1,
            spatial_extent=(16, 10),
            heads=2,
            dim_per_head=32,
            kernel_size=(5, 3),
            dilation=(2, 1),
            is_causal=(False, False),
            qkv_bias=True,
            device="gfx90a"
        )

if __name__ == "__main__":
    unittest.main() 