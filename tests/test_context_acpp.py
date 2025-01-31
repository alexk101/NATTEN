import os
import unittest

import natten
import torch

from natten.utils.testing import skip_if_acpp_is_not_supported


def _reset_everything():
    from natten.context import AutotunerContext, NattenContext

    NattenContext.reset()
    AutotunerContext.reset()
    natten.use_acpp_na()
    torch.use_deterministic_algorithms(True)  # Required for SYCL backend
    os.environ["NATTEN_LOG_LEVEL"] = "CRITICAL"


class ACPPContextTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    @skip_if_acpp_is_not_supported()
    def test_set_flags(self):
        assert natten.is_acpp_na_enabled()
        assert not natten.is_tiled_na_enabled()
        assert not natten.is_gemm_na_enabled()
        
        natten.use_tiled_na()
        assert not natten.is_acpp_na_enabled()
        assert natten.is_tiled_na_enabled()
        
        natten.use_acpp_na()
        assert natten.is_acpp_na_enabled()
        assert not natten.is_tiled_na_enabled()

    @skip_if_acpp_is_not_supported()
    def test_device_selection(self):
        # Test gfx908 selection
        natten.use_acpp_device("gfx908")
        assert natten.get_acpp_device() == "gfx908"
        
        # Test gfx90a selection
        natten.use_acpp_device("gfx90a")
        assert natten.get_acpp_device() == "gfx90a"
        
        # Test invalid device
        with self.assertRaises(ValueError):
            natten.use_acpp_device("invalid_device")

    @skip_if_acpp_is_not_supported()
    def test_deterministic_algorithms(self):
        # SYCL backend requires deterministic algorithms
        assert natten.are_deterministic_algorithms_enabled()
        
        with self.assertRaises(RuntimeError):
            torch.use_deterministic_algorithms(False)
            natten.use_acpp_na()


if __name__ == "__main__":
    unittest.main() 