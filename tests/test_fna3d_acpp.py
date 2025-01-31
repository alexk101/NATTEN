import os
import unittest

import torch

from natten import (
    has_bfloat,
    has_half,
    set_memory_usage_preference,
    use_autotuner,
    use_kv_parallelism_in_fused_na,
)
from natten.functional import na3d, na3d_av, na3d_qk
from natten.utils import check_all_args
from natten.utils.testing import (
    fna_supports_additional_kv,
    skip_if_acpp_is_not_supported,
)

def _reset_everything():
    import natten
    from natten.context import AutotunerContext, NattenContext

    NattenContext.reset()
    AutotunerContext.reset()
    natten.use_acpp_na()
    os.environ["NATTEN_LOG_LEVEL"] = "CRITICAL"
    torch.use_deterministic_algorithms(True)  # Required for SYCL backend

HAS_HALF = has_half()
HAS_BFLOAT = has_bfloat()

def check_args(kernel_size, dilation, is_causal):
    return check_all_args(3, kernel_size, dilation, is_causal)

def init_cpu_ref(B, H, X, Y, Z, D, kernel_size, dilation, has_bias, is_causal=None):
    kernel_size, dilation, is_causal = check_args(kernel_size, dilation, is_causal)
    assert not has_bias or not any(is_causal)
    with torch.no_grad():
        q, k, v = (
            torch.randn((B, H, X, Y, Z, D)) * (D**-0.5),
            torch.randn((B, H, X, Y, Z, D)),
            torch.randn((B, H, X, Y, Z, D)),
        )
        rpb = (
            None
            if not has_bias
            else torch.randn(H, 2 * kernel_size[0] - 1, 2 * kernel_size[1] - 1, 2 * kernel_size[2] - 1)
        )
        q_, k_, v_ = (
            q.clone().to("cpu"),
            k.clone().to("cpu"),
            v.clone().to("cpu"),
        )
        rpb_ = None if rpb is None else rpb.clone().to("cpu")

        attn_ref = na3d_qk(
            q,
            k,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            rpb=rpb,
        )
        attn_ref = attn_ref.softmax(dim=-1)
        out_ref = na3d_av(
            attn_ref, v, kernel_size=kernel_size, dilation=dilation, is_causal=is_causal
        )
    return (q_, k_, v_, rpb_, kernel_size, dilation), (attn_ref.to("cpu"), out_ref.to("cpu"))

class FNA3DACPPTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _test_against_cpu(self, inputs, reference, dtype, device, eps=1e-4, is_causal=None):
        q, k, v, rpb, kernel_size, dilation = inputs
        attn_ref, out_ref = reference

        q = q.to(device=device, dtype=dtype)
        k = k.to(device=device, dtype=dtype)
        v = v.to(device=device, dtype=dtype)
        if rpb is not None:
            rpb = rpb.to(device=device, dtype=dtype)

        out = na3d(q, k, v, kernel_size, dilation, is_causal, rpb)
        torch.testing.assert_close(
            out.cpu(), out_ref, rtol=eps, atol=eps, msg="Forward pass failed."
        )

    @skip_if_acpp_is_not_supported()
    def test_acpp_gfx908(self):
        self._test_all_dtypes_against_cpu(
            B=1,
            H=2,
            X=6,
            Y=5,
            Z=8,
            D=32,
            kernel_size=(5, 3, 7),
            dilation=(1, 1, 1),
            device="gfx908",
        )

    @skip_if_acpp_is_not_supported()
    def test_acpp_gfx90a(self):
        self._test_all_dtypes_against_cpu(
            B=1,
            H=2,
            X=6,
            Y=5,
            Z=8,
            D=32,
            kernel_size=(5, 3, 7),
            dilation=(1, 1, 1),
            device="gfx90a",
        )

    def _test_all_dtypes_against_cpu(
        self,
        B,
        H,
        X,
        Y,
        Z,
        D,
        kernel_size,
        dilation,
        device,
        has_bias=False,
        is_causal=None,
    ):
        kernel_size, dilation, is_causal = check_args(kernel_size, dilation, is_causal)
        assert not has_bias or not any(is_causal)
        inputs, reference = init_cpu_ref(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            has_bias=has_bias,
            is_causal=is_causal,
        )

        # Test float32 which is supported on both devices
        self._test_against_cpu(
            inputs=inputs,
            reference=reference,
            dtype=torch.float32,
            device=device,
            eps=1e-4,
            is_causal=is_causal,
        )

        # Test float16 which is supported on gfx908
        if device == "gfx908" and HAS_HALF:
            self._test_against_cpu(
                inputs=inputs,
                reference=reference,
                dtype=torch.float16,
                device=device,
                eps=1e-1,
                is_causal=is_causal,
            )

        # Test bfloat16 which is only supported on gfx90a
        if device == "gfx90a" and HAS_BFLOAT:
            self._test_against_cpu(
                inputs=inputs,
                reference=reference,
                dtype=torch.bfloat16,
                device=device,
                eps=1e-1,
                is_causal=is_causal,
            )

if __name__ == "__main__":
    unittest.main() 