import os
from enum import Enum
from typing import List

import click

DEFAULT_OUTPUT_DIR = "csrc/"

class Operation(Enum):
    PN = 0  # PointwiseNeighborhood
    NN = 2  # NeighborhoodNeighborhood
    IN = 3  # InverseNeighborhood

ALL_OPS = [
    Operation.PN,
    Operation.NN,
    Operation.IN,
]

class DataType:
    def __init__(self, name, natten_dtype, bits, is_int=False):
        self.name = name
        self.natten_dtype = natten_dtype
        self.bits = bits
        # SYCL doesn't have TF32, so we'll handle this differently
        self.check_tf32 = False
        self.multi_source = False

    def __str__(self):
        return self.name

    def source(self):
        out = ""
        out += f"  using DConfig = natten::gemm::detail::DTypeConfig<{self.natten_dtype}>;\n"
        return out

# Define supported data types
NATTEN_Double = DataType("double", "natten::float64", 64)
NATTEN_Float = DataType("float", "natten::float32", 32)
NATTEN_Half = DataType("sycl::half", "natten::float16", 16)
NATTEN_BFloat = DataType("sycl::bfloat16", "natten::bfloat16", 16)

class GemmConfig:
    def __init__(
        self, M, N, K, warp_M, warp_N, warp_K, math_M, math_N, math_K, stages, arch
    ):
        self.M = M
        self.N = N
        self.K = K
        self.warp_M = warp_M
        self.warp_N = warp_N
        self.warp_K = warp_K
        self.math_M = math_M
        self.math_N = math_N
        self.math_K = math_K
        self.stages = stages
        self.arch = arch

    def is_simt(self):
        return self.math_M == 1

    def __str__(self):
        return f"{self.M}x{self.N}x{self.K}_{self.warp_M}x{self.warp_N}x{self.warp_K}_{self.math_M}x{self.math_N}x{self.math_K}_{self.stages}_{self.arch}"

    def source(self, dtype):
        out = ""
        out += "  using GConfig = natten::gemm::detail::GemmConfig<"
        out += f"{self.M}, {self.N}, {self.K}, "
        out += f"{self.warp_M}, {self.warp_N}, {self.warp_K}, "
        out += f"{self.math_M}, {self.math_N}, {self.math_K}, "
        out += f"{self.stages}>;\n"
        out += f'  using ArchConfig = natten::gemm::detail::ArchArgs<"{self.arch}", {dtype}>;\n'
        return out

class AlignmentConfig:
    def __init__(self, alignment_a, alignment_b, alignment_c):
        self.alignment_a = alignment_a
        self.alignment_b = alignment_b
        self.alignment_c = alignment_c

    def source(self):
        out = ""
        out += "  using AConfig = natten::gemm::detail::AlignmentConfig<"
        out += f"{self.alignment_a}, {self.alignment_b}, {self.alignment_c}>;\n"
        return out

class NAGemmKernel:
    def __init__(
        self,
        op: Operation,
        dtype: DataType,
        gemm_config: GemmConfig,
        alignment_config: AlignmentConfig,
        header_files: List[str],
    ):
        self.op = op
        self.dtype = dtype
        self.gemm_config = gemm_config
        self.alignment_config = alignment_config
        self.header_files = header_files
        self.filename = f"gemm_{op.name.lower()}_{str(dtype)}_{str(gemm_config)}.cpp"

    def method_decl(self):
        return f"SYCL_EXTERNAL void {self.get_name()}(sycl::queue& queue, typename {self.get_class_name()}::Params p)"

    def method_def(self):
        out = ""
        out += self.dtype.source()
        out += self.gemm_config.source(self.dtype)
        out += self.alignment_config.source()
        out += f"  using Kernel = {self.get_class_name()};\n"
        out += "  queue.submit([&](sycl::handler& cgh) {\n"
        out += "    cgh.parallel_for(\n"
        out += "      sycl::nd_range<3>(/* compute work size */),\n"
        out += "      [=](sycl::nd_item<3> item) {\n"
        out += "        if (!p.advance_to_block()) {\n"
        out += "          return;\n"
        out += "        }\n"
        out += "        Kernel::gemm_kernel(p, item);\n"
        out += "      });\n"
        out += "  }).wait();\n"
        return out

    def get_name(self):
        return f"gemm_{self.op.name.lower()}_{str(self.dtype)}_{str(self.gemm_config)}"

    def get_class_name(self):
        return f"natten::gemm::detail::GemmKernel<{self.op.name}, DConfig, GConfig, AConfig, ArchConfig>"

def write_combined_source_file(path: str, filename: str, headers: List[str], kernels: List[NAGemmKernel]):
    source_head = []
    source_head += ["#include <iostream>\n"]
    source_head += [f"#include <{h}>\n" for h in headers]
    source_head += ["\nnamespace natten {\n"]
    source_head += ["namespace acpp {\n"]
    source_head += ["namespace gemm {\n"]

    source_head = "".join(source_head)
    source_body = "".join(kernel.source() for kernel in kernels)
    source_foot = "".join(["}\n", "}\n", "}\n", "\n"])

    with open(f"{path}/{filename}", "w") as f:
        f.write(source_head)
        f.write(source_body)
        f.write(source_foot)

class DeviceDispatcher:
    def __init__(self, operation: Operation, dtype: DataType):
        self.operation = operation
        self.dtype = dtype
        self.name_cc = f"DISPATCH_GEMM_{operation.name}"
        self.name_target = self.name_cc + "_ARCH"
        self.devices: List[str] = []

    def append(self, arch: str):
        self.devices.append(arch)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(device, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        
        for i, arch in enumerate(self.devices):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f'if (device == "{arch}") {{ \\\n'
            dispatcher_str += f"      {self.name_target}_{arch}(__VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN GEMM kernel launch failed!" \\\n'
        dispatcher_str += '                << "Device not supported." \\\n'
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();\n\n"
        return dispatcher_str

# AMD GPU architectures
_ARCH_TO_UPPER_BOUND = {
    "gfx908": "gfx90a",  # MI100
    "gfx90a": "gfx940",  # MI200
    "gfx940": "gfx1100", # MI300
}

# GEMM configurations for different architectures and data types
GEMM_CONFIGS = {
    "gfx908": {
        NATTEN_Float: [
            GemmConfig(128, 128, 8, 32, 32, 8, 8, 8, 8, 2, "gfx908"),
            GemmConfig(128, 64, 8, 32, 32, 8, 8, 8, 8, 2, "gfx908"),
        ],
        NATTEN_Half: [
            GemmConfig(128, 128, 16, 32, 32, 16, 16, 16, 16, 2, "gfx908"),
            GemmConfig(128, 64, 16, 32, 32, 16, 16, 16, 16, 2, "gfx908"),
        ],
    },
    "gfx90a": {
        NATTEN_Float: [
            GemmConfig(128, 128, 8, 32, 32, 8, 8, 8, 8, 2, "gfx90a"),
            GemmConfig(128, 64, 8, 32, 32, 8, 8, 8, 8, 2, "gfx90a"),
        ],
        NATTEN_Half: [
            GemmConfig(128, 128, 16, 32, 32, 16, 16, 16, 16, 2, "gfx90a"),
            GemmConfig(128, 64, 16, 32, 32, 16, 16, 16, 16, 2, "gfx90a"),
        ],
        NATTEN_BFloat: [
            GemmConfig(128, 128, 16, 32, 32, 16, 16, 16, 16, 2, "gfx90a"),
            GemmConfig(128, 64, 16, 32, 32, 16, 16, 16, 16, 2, "gfx90a"),
        ],
    },
}

@click.command()
@click.option(
    "-o",
    "--output-directory",
    default=DEFAULT_OUTPUT_DIR,
    help="Path to the directory where the auto-generated kernel instantiations are dumped.",
)
@click.option(
    "--num-splits",
    default=16,
    help="Number of source files into which the kernels are split. Default: 16.",
)
def generate_acpp_gemm(output_directory: str, num_splits: int):
    kernels = []
    device_dispatchers = []

    # For each operation type
    for op in ALL_OPS:
        # For each architecture
        for arch in _ARCH_TO_UPPER_BOUND.keys():
            device_dispatcher = DeviceDispatcher(op, arch)
            device_dispatcher.append(arch)
            device_dispatchers.append(device_dispatcher)

            # For each data type
            for dtype in [NATTEN_Float, NATTEN_Half, NATTEN_BFloat]:
                if arch >= dtype.min_arch:
                    # Generate kernels with different configurations
                    for gemm_config in GEMM_CONFIGS[arch][dtype]:
                        kernel = NAGemmKernel(
                            op=op,
                            dtype=dtype,
                            gemm_config=gemm_config,
                            alignment_config=AlignmentConfig(8, 8, 8),  # Default alignment for now
                            header_files=["natten/acpp/gemm/kernel.h"]
                        )
                        kernels.append(kernel)

    # Create output directories
    path_to_sources = f"{output_directory}/autogen/src/acpp/gemm/"
    rel_header = "natten_autogen/acpp/gemm/"
    path_to_header_dir = f"{output_directory}/autogen/include/{rel_header}"

    os.makedirs(path_to_sources, exist_ok=True)
    os.makedirs(path_to_header_dir, exist_ok=True)

    # Write headers and source files
    headers = "".join(kernel.header() for kernel in kernels)
    write_header_file(headers, f"{path_to_header_dir}/kernels.h", ["natten", "acpp", "gemm"], ["natten/acpp/gemm/kernels.h"])

    # Split kernels into multiple source files
    split_size = (len(kernels) + num_splits - 1) // num_splits
    for split_idx in range(num_splits):
        start_idx = split_idx * split_size
        end_idx = min(start_idx + split_size, len(kernels))
        write_combined_source_file(
            path_to_sources,
            f"source_{split_idx}.cpp",
            ["natten/acpp/gemm/kernels.h"],
            kernels[start_idx:end_idx]
        )

def write_header_file(content: str, path: str, namespaces: List[str], headers: List[str]):
    header = []
    header += ["#pragma once\n\n"]
    header += ["#include <iostream>\n"]
    header += [f"#include <{h}>\n" for h in headers]
    header += ["\n"]
    header += [f"namespace {ns} {{\n" for ns in namespaces]
    header = "".join(header)

    footer = "".join(["}\n" for _ in namespaces])

    with open(path, "w") as f:
        f.write(header)
        f.write(content)
        f.write(footer)

if __name__ == "__main__":
    generate_acpp_gemm()
