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
    def __init__(self, name, natten_name, short_name, bits, min_arch):
        self.name = name
        self.natten_name = natten_name
        self.short_name = short_name
        self.bits = bits
        self.min_arch = min_arch

    def __str__(self):
        return self.name

    def source(self):
        out = ""
        out += f"  using DConfig = natten::gemm::detail::DTypeConfig<{self.natten_name}>;\n"
        return out

# Define supported data types with their minimum architecture requirements
NATTEN_Float = DataType("float", "natten::float32", "float32", 32, "gfx908")
NATTEN_Half = DataType("sycl::half", "natten::float16", "float16", 16, "gfx908")
NATTEN_BFloat = DataType("sycl::bfloat16", "natten::bfloat16", "bfloat16", 16, "gfx90a")

class GemmShape:
    def __init__(self, M, N, K, warp_M, warp_N, warp_K):
        self.M = M
        self.N = N
        self.K = K
        self.warp_M = warp_M
        self.warp_N = warp_N
        self.warp_K = warp_K

    def __str__(self):
        return f"{self.M}x{self.N}x{self.K}_{self.warp_M}x{self.warp_N}x{self.warp_K}"

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

# Define GEMM configurations for different architectures and operations
op_and_dtype_2_config = {
    "gfx908": {
        Operation.PN: {
            NATTEN_Half: GemmConfig(
                M=128, N=128, K=32,
                warp_M=64, warp_N=64, warp_K=32,
                math_M=16, math_N=16, math_K=16,
                stages=2, arch="gfx908"
            ),
        },
        Operation.NN: {
            NATTEN_Half: GemmConfig(
                M=64, N=64, K=32,
                warp_M=32, warp_N=32, warp_K=32,
                math_M=16, math_N=16, math_K=16,
                stages=2, arch="gfx908"
            ),
        },
    },
    "gfx90a": {
        Operation.PN: {
            NATTEN_BFloat: GemmConfig(
                M=128, N=128, K=32,
                warp_M=64, warp_N=64, warp_K=32,
                math_M=16, math_N=16, math_K=16,
                stages=2, arch="gfx90a"
            ),
        },
    }
}

# Add AMD GPU architecture mapping
_ARCH_TO_UPPER_BOUND = {
    "gfx908": "gfx90a",  # MI100
    "gfx90a": "gfx940",  # MI200
    "gfx940": "gfx1100", # MI300
}

class KernelConfig:
    def __init__(self, kernel_size: int, operation: Operation):
        self.kernel_size = kernel_size
        self.operation = operation

    def __str__(self):
        return f"k{self.kernel_size}"

class DataTypeDispatcher:
    def __init__(self, operation: Operation, arch: str):
        self.operation = operation
        self.name_base = f"na2d_{operation.name.lower()}_acpp_gemm_{arch}"
        self.name_cc = f"DISPATCH_DTYPE_{self.name_base}"
        self.name_target = f"DISPATCH_KERNELSIZE_{self.name_base}"
        self.dtypes: List[DataType] = []

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(dtype, kernel_size, dim, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f'if (dtype == "{dtype.short_name}")'
            dispatcher_str += " { \\\n"
            dispatcher_str += f"      {self.name_target}_{dtype.name}(kernel_size, dim, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN kernel launch failed! " \\\n'
        dispatcher_str += f'                << "{self.name_base} does not support this data type." \\\n'
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();\n\n"
        return dispatcher_str

class DeviceDispatcher:
    def __init__(self, operation: Operation):
        self.operation = operation
        self.name_base = f"na2d_{operation.name.lower()}_acpp_gemm"
        self.name_cc = f"LAUNCH_{self.name_base}"
        self.targets = {
            arch: DataTypeDispatcher(operation=operation, arch=arch).name_cc
            for arch in sorted(_ARCH_TO_UPPER_BOUND.keys())
        }

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(device, dtype, kernel_size, dim, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, (arch, target_name) in enumerate(self.targets.items()):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f'if (device == "{arch}")'
            dispatcher_str += " { \\\n"
            dispatcher_str += f"      {target_name}(dtype, kernel_size, dim, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN kernel launch failed! " \\\n'
        dispatcher_str += f'                << "{self.name_base} device not supported." \\\n'
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();\n\n"
        return dispatcher_str

class KernelSizeDispatcher:
    def __init__(self, dtype: DataType, operation: Operation, arch: str):
        self.operation = operation
        name_base = f"na2d_{operation.name.lower()}_acpp_gemm"
        self.name_base = name_base + f"_{arch}_{dtype}"
        self.name_target_base = name_base + f"_{dtype}"
        self.name_cc = f"DISPATCH_KERNELSIZE_{self.name_base}"
        self.name_target = f"DISPATCH_ALIGNMENT_{self.name_target_base}"
        self.configs: List[KernelConfig] = []

    def append(self, gemm_config: KernelConfig):
        self.configs.append(gemm_config)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(kernel_size, dim, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, conf in enumerate(self.configs):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (kernel_size == {conf.kernel_size})"
            dispatcher_str += " { \\\n"
            dispatcher_str += f"      {self.name_target}_{conf}(dim, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN kernel launch failed! " \\\n'
        dispatcher_str += f'                << "{self.name_base} does not support kernel size " << kernel_size << ". " \\\n'
        dispatcher_str += '                << "You may try generating it manually and build from source. " \\\n'
        dispatcher_str += '                << "Refer to NATTEN\'s github repository for more information." \\\n'
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();\n\n"
        return dispatcher_str

class AlignmentConfig:
    def __init__(self, numel: int, operation: Operation):
        self.numel = numel
        self.operation = operation

    def __str__(self):
        return f"align{self.numel}"

class AlignmentDispatcher:
    def __init__(self, dtype: DataType, operation: Operation, arch: str, kernel_config: KernelConfig):
        self.dtype = dtype
        self.operation = operation
        self.kernel_config = kernel_config
        name_base = f"na2d_{operation.name.lower()}_acpp_gemm"
        self.name_base = f"{name_base}_{arch}_{dtype}_{kernel_config}"
        self.name_cc = f"DISPATCH_ALIGNMENT_{self.name_base}"
        self.name_target = f"{name_base}_kernel"
        self.alignments = [8] if dtype.bits >= 32 else [16]  # Simplified alignment rules for SYCL

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(dim, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        i = 0
        if len(self.alignments) > 1:
            for numel in self.alignments:
                alignment = AlignmentConfig(numel, self.operation)
                dispatcher_str += "    "
                if i > 0:
                    dispatcher_str += "else "
                i += 1
                dispatcher_str += f"if (dim % {numel} == 0)"
                dispatcher_str += " { \\\n"
                dispatcher_str += f"      {self.name_target}_{alignment}(__VA_ARGS__); \\\n"
                dispatcher_str += "    } \\\n"
            dispatcher_str += "    else { \\\n"
            dispatcher_str += '      std::cerr << "NATTEN kernel launch failed! " \\\n'
            dispatcher_str += f'                << "{self.name_base} requires proper alignment. " \\\n'
            dispatcher_str += f'                << "Got dim=" << dim << ", dtype={self.dtype.name}. " \\\n'
            dispatcher_str += "                << std::endl; \\\n"
            dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
            dispatcher_str += "    } \\\n"
        else:
            alignment = AlignmentConfig(self.alignments[0], self.operation)
            dispatcher_str += f"  {self.name_target}_{alignment}(__VA_ARGS__); \\\n"
        dispatcher_str += "}();\n\n"
        return dispatcher_str