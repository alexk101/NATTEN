import os
import unittest

import torch

from natten import has_bfloat, has_half
from natten.libnatten import compute_delta  # type: ignore
from natten.utils.testing import skip_if_acpp_is_not_supported

def _reset_everything():
    import natten
    from natten.context import AutotunerContext, NattenContext

    NattenContext.reset()
    AutotunerContext.reset()
    natten.use_acpp_na()
    torch.use_deterministic_algorithms(False)
    os.environ["NATTEN_LOG_LEVEL"] = "CRITICAL"

    # NOTE: SYCL backend requires deterministic algorithms
    torch.use_deterministic_algorithms(True)

HAS_HALF = has_half()
HAS_BFLOAT = has_bfloat()

def compute_delta_pt(out: torch.Tensor, d_out: torch.Tensor, dtype) -> torch.Tensor:
    assert out.dim() == d_out.dim() and out.dim() >= 2
    for i in range(out.dim()):
        assert out.shape[i] == d_out.shape[i]
    with torch.no_grad():
        return (out.clone().to(dtype) * d_out.clone().to(dtype)).sum(-1)

class ComputeDeltaACPPTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _test_against_reference(self, input_shape, eps, dtype, dtype_out, device):
        assert len(input_shape) >= 2
        out = torch.randn(input_shape, device=device, dtype=dtype)
        d_out = torch.randn_like(out)

        with torch.no_grad():
            delta_ref = compute_delta_pt(out, d_out, dtype_out)

            delta = torch.empty(input_shape[:-1], device=device, dtype=dtype_out)
            compute_delta(out, d_out, delta)

            torch.testing.assert_close(delta, delta_ref, atol=eps, rtol=0)

    def _test_all_dtypes_against_reference(self, input_shape, device):
        self._test_against_reference(
            input_shape=input_shape,
            eps=1e-2,
            dtype=torch.float32,
            dtype_out=torch.float32,
            device=device,
        )
        if HAS_HALF:
            self._test_against_reference(
                input_shape=input_shape,
                eps=1e-1,
                dtype=torch.float16,
                dtype_out=torch.float32,
                device=device,
            )

        if HAS_BFLOAT and device == "gfx90a":  # Only test bfloat16 on gfx90a
            self._test_against_reference(
                input_shape=input_shape,
                eps=1e-1,
                dtype=torch.bfloat16,
                dtype_out=torch.float32,
                device=device,
            )

    @skip_if_acpp_is_not_supported()
    def test_against_bmm_style_gfx908(self):
        input_sizes = [
            (2, 4),
            (5, 4),
            (1, 4, 64, 32),
            (128, 49),
            (50, 60),
            (127, 61),
            (128, 4, 56, 56, 32),
            (128, 8, 56, 56, 128),
            (1, 1, 56, 56, 1024),
        ]
        for input_size in input_sizes:
            self._test_all_dtypes_against_reference(input_size, "gfx908")

    @skip_if_acpp_is_not_supported()
    def test_against_bmm_style_gfx90a(self):
        input_sizes = [
            (128, 4, 56, 56, 32),
            (128, 8, 56, 56, 128),
            (1, 1, 56, 56, 1024),
        ]
        for input_size in input_sizes:
            self._test_all_dtypes_against_reference(input_size, "gfx90a")

if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main() 