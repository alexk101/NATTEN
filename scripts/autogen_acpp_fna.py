# Copyright (c) 2022-2024 Ali Hassani.
#
# This script is intended to emit fused kernel instantiations into
# a variable number of source files, generate appropriate headers
# and a single dispatcher interface, which will be used by the
# NATTEN API to call the kernels.

import os
from typing import List, Optional, Tuple
import click

DEFAULT_OUTPUT_DIR = "csrc/"

# AMD GPU architectures
_ARCH_TO_UPPER_BOUND = {
    "gfx908": "gfx90a",  # MI100
    "gfx90a": "gfx940",  # MI200
    "gfx940": "gfx1100", # MI300
}

class KernelConfig:
    def __init__(self, na_dim: int, arch: str, gemm_shape: Tuple[int, int, int]):
        assert 0 < na_dim <= 3
        assert arch in _ARCH_TO_UPPER_BOUND.keys()
        assert len(gemm_shape) == 3
        self.na_dim = na_dim
        self.arch = arch
        self.gemm_shape = gemm_shape

    def get_name(self, is_backward: bool) -> str:
        backward_str = "" if not is_backward else "_backward"
        return f"fna{self.na_dim}d{backward_str}_{self.gemm_shape[0]}x{self.gemm_shape[1]}x{self.gemm_shape[2]}_{self.arch}"

class DataType:
    def __init__(self, name, natten_name, short_name, bits, min_arch):
        self.name = name
        self.natten_name = natten_name
        self.bits = bits
        self.short_name = short_name
        self.min_arch = min_arch

NATTEN_Float = DataType("float", "natten::float32", "float32", 32, "gfx908")
NATTEN_Half = DataType("sycl::half", "natten::float16", "float16", 16, "gfx908")
NATTEN_BFloat = DataType("sycl::bfloat16", "natten::bfloat16", "bfloat16", 16, "gfx90a")

# SYCL kernel template
KERNEL_DECL_TEMPLATE = """SYCL_EXTERNAL void {NAME}(
    sycl::queue& queue,
    typename {CPP_CLASS}::Params p);
"""

KERNEL_IMPL_TEMPLATE = """SYCL_EXTERNAL void {NAME}(
    sycl::queue& queue,
    typename {CPP_CLASS}::Params p) {{
    queue.submit([&](sycl::handler& cgh) {{
        cgh.parallel_for(
            sycl::nd_range<3>(/* compute work size */),
            [=](sycl::nd_item<3> item) {{
                if (!p.advance_to_block()) {{
                    return;
                }}
                {CPP_CLASS}::attention_kernel(p, item);
            }});
    }}).wait();
}}
"""

class KernelConfigList:
    def __init__(self, na_dim: int, arch: str, gemm_shapes: List[Tuple[int, int, int]]):
        assert 0 < na_dim <= 3
        assert arch in _ARCH_TO_UPPER_BOUND.keys()
        self.na_dim = na_dim
        self.arch = arch
        self.gemm_shapes = gemm_shapes

    @property
    def configs(self) -> List[KernelConfig]:
        return [
            KernelConfig(na_dim=self.na_dim, arch=self.arch, gemm_shape=gemm_shape)
            for gemm_shape in self.gemm_shapes
        ]

    def get_name(self, is_backward: bool) -> str:
        backward_str = "" if not is_backward else "_backward"
        return f"fna{self.na_dim}d{backward_str}_{self.arch}"

class DeviceDispatcher:
    def __init__(self, is_backward: bool, na_dim: int):
        self.na_dim = na_dim
        self.name_cc = (
            f"DISPATCH_FNA_FORWARD_{self.na_dim}D"
            if not is_backward
            else f"DISPATCH_FNA_BACKWARD_{self.na_dim}D"
        )
        self.name_target = self.name_cc + "_ARCH"
        self.devices: List[str] = []
        self.is_backward = is_backward

    def append(self, arch: str):
        self.devices.append(arch)

    def get_dispatcher(self):
        dispatcher_str = ""
        if self.is_backward:
            dispatcher_str += f"#define {self.name_cc}(device, dtype, is_causal, cb) \\\n"
        else:
            dispatcher_str += f"#define {self.name_cc}(device, dtype, is_causal, has_rpb, computes_lse, cb) \\\n"
        
        dispatcher_str += "  [&] { \\\n"
        for i, arch in enumerate(self.devices):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            # Use string comparison for device architecture
            dispatcher_str += f'if (device == "{arch}")'
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            if self.is_backward:
                dispatcher_str += f'  {self.name_target}_{arch}(dtype, is_causal, cb); \\\n'
            else:
                dispatcher_str += f'  {self.name_target}_{arch}(dtype, is_causal, has_rpb, computes_lse, cb); \\\n'
            dispatcher_str += "    } \\\n"
        
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN FNA kernel launch failed!" \\\n'
        dispatcher_str += '                << "Fused neighborhood attention is not implemented for this device." \\\n'
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();\n\n"
        return dispatcher_str

FORWARD_GEMM_SHAPES = {
    "gfx908": {  # MI100
        NATTEN_Float: [
            (64, 64, 32),   # Small problem sizes
            (64, 64, 64),   # Medium
            (64, 64, 128),  # Large
            (64, 64, 2**16) # Very large
        ],
        NATTEN_Half: [
            (64, 128, 32),
            (64, 128, 64),
            (64, 128, 128),
            (64, 128, 2**16)
        ]
    },
    "gfx90a": {  # MI200 - can handle larger tile sizes
        NATTEN_Float: [
            (128, 128, 32),
            (128, 128, 64),
            (128, 128, 128),
            (128, 128, 2**16)
        ],
        NATTEN_Half: [
            (128, 256, 32),
            (128, 256, 64),
            (128, 256, 128),
            (128, 256, 2**16)
        ],
        NATTEN_BFloat: [
            (128, 256, 32),
            (128, 256, 64),
            (128, 256, 128),
            (128, 256, 2**16)
        ]
    }
}

BACKWARD_GEMM_SHAPES = {
    "gfx908": {  # MI100
        NATTEN_Float: [
            (64, 64, 32),
            (64, 64, 64),
            (64, 64, 128),
            (64, 64, 2**16)
        ],
        NATTEN_Half: [
            (64, 128, 32),
            (64, 128, 64),
            (64, 128, 128),
            (64, 128, 2**16)
        ]
    },
    "gfx90a": {  # MI200
        NATTEN_Float: [
            (128, 128, 32),
            (128, 128, 64),
            (128, 128, 128),
            (128, 128, 2**16)
        ],
        NATTEN_Half: [
            (128, 256, 32),
            (128, 256, 64),
            (128, 256, 128),
            (128, 256, 2**16)
        ],
        NATTEN_BFloat: [
            (128, 256, 32),
            (128, 256, 64),
            (128, 256, 128),
            (128, 256, 2**16)
        ]
    }
}

class RankDispatcher:
    def __init__(self, is_backward: bool):
        self.name_cc = "DISPATCH_FNA_RANK" + ("_BACKWARD" if is_backward else "_FORWARD")
        self.name_target = self.name_cc + "_DIM"
        self.ranks: List[int] = []
        self.is_backward = is_backward

    def append(self, rank: int):
        self.ranks.append(rank)

    def get_dispatcher(self):
        dispatcher_str = ""
        if self.is_backward:
            dispatcher_str += f"#define {self.name_cc}(rank, device, dtype, is_causal, cb) \\\n"
        else:
            dispatcher_str += f"#define {self.name_cc}(rank, device, dtype, is_causal, has_rpb, computes_lse, cb) \\\n"
        
        dispatcher_str += "  switch(rank) { \\\n"
        for rank in self.ranks:
            dispatcher_str += f"    case {rank}: \\\n"
            if self.is_backward:
                dispatcher_str += f"      {self.name_target}{rank}(device, dtype, is_causal, cb); \\\n"
            else:
                dispatcher_str += f"      {self.name_target}{rank}(device, dtype, is_causal, has_rpb, computes_lse, cb); \\\n"
            dispatcher_str += "      break; \\\n"
        dispatcher_str += "  }\n\n"
        return dispatcher_str

class DataTypeDispatcher:
    def __init__(self, is_backward: bool, na_dim: int, arch: str):
        self.na_dim = na_dim
        self.arch = arch
        self.name_cc = f"DISPATCH_FNA_{'BACKWARD' if is_backward else 'FORWARD'}_{na_dim}D_ARCH_{arch}"
        self.name_target = self.name_cc + "_DTYPE"
        self.dtypes: List[DataType] = []
        self.is_backward = is_backward

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        if self.is_backward:
            dispatcher_str += f"#define {self.name_cc}(dtype, is_causal, cb) \\\n"
        else:
            dispatcher_str += f"#define {self.name_cc}(dtype, is_causal, has_rpb, computes_lse, cb) \\\n"
        
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "  "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f'if (dtype == "{dtype.natten_name}") {{ \\\n'
            if self.is_backward:
                dispatcher_str += f"    {self.name_target}_{dtype.short_name}(is_causal, cb); \\\n"
            else:
                dispatcher_str += f"    {self.name_target}_{dtype.short_name}(is_causal, has_rpb, computes_lse, cb); \\\n"
            dispatcher_str += "  } \\\n"
        
        dispatcher_str += "  else { \\\n"
        dispatcher_str += '    std::cerr << "NATTEN FNA kernel launch failed!" << std::endl; \\\n'
        dispatcher_str += "    exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "  }\n\n"
        return dispatcher_str

def write_header_file(content: str, path: str, namespaces: List[str], includes: List[str]):
    with open(path, "w") as f:
        f.write("// Auto-generated file. Do not edit!\n\n")
        f.write("#pragma once\n\n")
        for include in includes:
            f.write(f'#include "{include}"\n')
        f.write("\n")
        for ns in namespaces:
            f.write(f"namespace {ns} {{\n")
        f.write("\n")
        f.write(content)
        f.write("\n")
        for _ in namespaces:
            f.write("} // namespace\n")

def write_combined_source_file(
    path_to_sources: str,
    filename: str,
    headers: List[str],
    kernels: List['FusedNAKernel']
):
    with open(os.path.join(path_to_sources, filename), "w") as f:
        f.write("// Auto-generated file. Do not edit!\n\n")
        for header in headers:
            f.write(f'#include "{header}"\n')
        f.write("\n")
        for kernel in kernels:
            f.write(kernel.get_implementation())
            f.write("\n")

class FusedNAKernel:
    def __init__(
        self,
        na_dim: int,
        arch: str,
        dtype: DataType,
        gemm_shape: Tuple[int, int, int],
        is_backward: bool = False,
        is_causal: bool = False,
        has_rpb: bool = False,
        computes_lse: bool = True,
    ):
        self.na_dim = na_dim
        self.arch = arch
        self.dtype = dtype
        self.gemm_shape = gemm_shape
        self.is_backward = is_backward
        self.is_causal = is_causal
        self.has_rpb = has_rpb
        self.computes_lse = computes_lse
        
        # Generate names and class names
        self.name_cc = self.get_name(is_backward)
        self.cpp_class = self.get_cpp_class()
        self.path_to_header = f"natten/acpp/fna/{'backward' if is_backward else 'forward'}.h"

    def get_cpp_class(self) -> str:
        kernel_type = "FusedNeighborhoodAttentionBackwardKernel" if self.is_backward else "FusedNeighborhoodAttentionKernel"
        causal_mask = f"CausalMask<{str(self.is_causal).lower()}>" if self.na_dim == 1 else \
                     f"CausalMask<{str(self.is_causal).lower()}, {str(self.is_causal).lower()}>" if self.na_dim == 2 else \
                     f"CausalMask<{str(self.is_causal).lower()}, {str(self.is_causal).lower()}, {str(self.is_causal).lower()}>"
        
        return f"{kernel_type}<{self.na_dim}, {causal_mask}, {self.dtype.name}, {self.arch}, {str(self.has_rpb).lower()}, {self.gemm_shape[0]}, {self.gemm_shape[1]}, {self.gemm_shape[2]}, false, {str(self.computes_lse).lower()}>"

    def header(self) -> str:
        return KERNEL_DECL_TEMPLATE.format(
            NAME=self.name_cc,
            CPP_CLASS=self.cpp_class
        )

    def source(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            NAME=self.name_cc,
            CPP_CLASS=self.cpp_class
        )

    def get_implementation(self) -> str:
        return self.source()

def generate_acpp_kernels(path: str, num_splits: int = 2):
    kernels = []
    dtype_dispatchers = []
    device_dispatchers = []
    rank_dispatchers = []

    # Generate kernels for each dimension (1D, 2D, 3D)
    for na_dim in [1, 2, 3]:
        rank_dispatcher = RankDispatcher(False)
        rank_dispatcher.append(na_dim)
        rank_dispatchers.append(rank_dispatcher)

        # For each architecture
        for arch in _ARCH_TO_UPPER_BOUND.keys():
            device_dispatcher = DeviceDispatcher(False, na_dim)
            device_dispatcher.append(arch)
            device_dispatchers.append(device_dispatcher)

            # For each data type supported by this architecture
            for dtype in [NATTEN_Float, NATTEN_Half, NATTEN_BFloat]:
                if arch >= dtype.min_arch:
                    dtype_dispatcher = DataTypeDispatcher(False, na_dim, arch)
                    dtype_dispatcher.append(dtype)
                    dtype_dispatchers.append(dtype_dispatcher)

                    # Get GEMM shapes for this configuration
                    gemm_shapes = FORWARD_GEMM_SHAPES[arch][dtype]
                    
                    # Generate kernels for different configurations
                    for gemm_shape in gemm_shapes:
                        for is_causal in [True, False]:
                            for has_rpb in [True, False]:
                                for computes_lse in [True, False]:
                                    kernel = FusedNAKernel(
                                        na_dim=na_dim,
                                        arch=arch,
                                        dtype=dtype,
                                        gemm_shape=gemm_shape,
                                        is_backward=False,
                                        is_causal=is_causal,
                                        has_rpb=has_rpb,
                                        computes_lse=computes_lse
                                    )
                                    kernels.append(kernel)

    # Create output directories
    path_to_sources = f"{path}/autogen/src/acpp/fna/"
    rel_header = "natten_autogen/acpp/fna/"
    path_to_header_dir = f"{path}/autogen/include/{rel_header}"

    os.makedirs(path_to_sources, exist_ok=True)
    os.makedirs(path_to_header_dir, exist_ok=True)

    # Write header files
    path_headers = f"{path_to_header_dir}/kernels.h"
    path_dtype = f"{path_to_header_dir}/interface.h"
    path_cm = f"{path_to_header_dir}/dispatch_cm.h"

    # Generate dispatcher strings
    dtype_disp = ""
    for dispatcher in dtype_dispatchers:
        dtype_disp += dispatcher.get_dispatcher()

    cm_disp = ""
    for dispatcher in device_dispatchers:
        cm_disp += dispatcher.get_dispatcher()

    # Write headers
    headers = ""
    for kernel in kernels:
        headers += kernel.header()

    # Split kernels into multiple source files
    split_size = (len(kernels) + num_splits - 1) // num_splits
    for split_idx in range(num_splits):
        kernel_start_idx = split_size * split_idx
        kernel_end_idx = min(kernel_start_idx + split_size, len(kernels))
        source_list = kernels[kernel_start_idx:kernel_end_idx]
        write_combined_source_file(
            path_to_sources, f"source_{split_idx}.cpp", ["natten/acpp/fna/kernels.h"], source_list
        )

    # Write header files
    namespaces = ["natten", "acpp", "fna"]
    acpp_headers = ["natten/dtypes.h"]
    write_header_file(headers, path_headers, namespaces, acpp_headers)
    write_header_file(dtype_disp, path_dtype, namespaces, acpp_headers + [rel_header + "kernels.h"])
    write_header_file(cm_disp, path_cm, namespaces, acpp_headers + [rel_header + "interface.h"])

@click.command()
@click.option(
    "-o",
    "--output-directory",
    default=DEFAULT_OUTPUT_DIR,
    help="Path to the directory where the auto-generated kernel instantiations are dumped."
)
@click.option(
    "--num-splits",
    default=16,
    help="Number of source files into which the kernels are split. Default: 16.",
)
def generate_acpp_fused(output_directory: str, num_splits: int):
    generate_acpp_kernels(output_directory, num_splits=num_splits)

if __name__ == "__main__":
    generate_acpp_fused()