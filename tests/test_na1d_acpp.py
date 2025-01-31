import os
import unittest

import torch

from natten import (
    has_bfloat,
    has_half,
)
from natten.functional import na1d, na1d_av, na1d_qk
from natten.utils import check_all_args
from natten.utils.testing import skip_if_acpp_is_not_supported

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
    return check_all_args(1, kernel_size, dilation, is_causal)

def init_cpu_ref(B, H, L, D, kernel_size, dilation, has_bias, is_causal=None):
    kernel_size, dilation, is_causal = check_args(kernel_size, dilation, is_causal)
    assert not has_bias or not any(is_causal)
    with torch.no_grad():
        q, k, v = (
            torch.randn((B, H, L, D)) * (D**-0.5),
            torch.randn((B, H, L, D)),
            torch.randn((B, H, L, D)),
        )
        rpb = None if not has_bias else torch.randn(H, 2 * kernel_size[0] - 1)
        q_, k_, v_ = (
            q.clone().to("cpu"),
            k.clone().to("cpu"),
            v.clone().to("cpu"),
        )
        rpb_ = None if rpb is None else rpb.clone().to("cpu")

        attn_ref = na1d_qk(
            q,
            k,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            rpb=rpb,
        )
        attn_ref = attn_ref.softmax(dim=-1)
        out_ref = na1d_av(
            attn_ref, v, kernel_size=kernel_size, dilation=dilation, is_causal=is_causal
        )
    return (q_, k_, v_, rpb_, kernel_size, dilation), (attn_ref.to("cpu"), out_ref.to("cpu"))

class NA1DACPPTests(unittest.TestCase):
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

        attn = na1d_qk(
            q,
            k,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            rpb=rpb,
        )
        attn = attn.softmax(dim=-1)
        out = na1d_av(
            attn,
            v,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
        )

        torch.testing.assert_close(attn.cpu(), attn_ref, rtol=eps, atol=eps)
        torch.testing.assert_close(out.cpu(), out_ref, rtol=eps, atol=eps)

    @skip_if_acpp_is_not_supported()
    def test_acpp_gfx908(self):
        self._test_all_dtypes_against_cpu(
            B=2,
            H=4,
            L=16,
            D=32,
            kernel_size=(5,),
            dilation=(1,),
            device="gfx908",
        )

    @skip_if_acpp_is_not_supported()
    def test_acpp_gfx90a(self):
        self._test_all_dtypes_against_cpu(
            B=2,
            H=4,
            L=16,
            D=32,
            kernel_size=(5,),
            dilation=(1,),
            device="gfx90a",
        )

    def _test_all_dtypes_against_cpu(
        self,
        B,
        H,
        L,
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
            L=L,
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