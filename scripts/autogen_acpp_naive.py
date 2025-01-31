# Copyright (c) 2022-2024 Ali Hassani.
#
# This script is intended to emit naive kernel instantiations into
# a variable number of source files, generate appropriate headers
# and a single dispatcher interface, which will be used by the
# NATTEN API to call the kernels.

import os
from enum import Enum
from itertools import product
from typing import List

import click

DEFAULT_OUTPUT_DIR = "csrc/"

class Operation(Enum):
    PN = 0
    PN_BIAS = 1
    NN = 2
    IN = 3
    RPBGRAD = 4

ALL_OPS = [
    Operation.PN,
    Operation.PN_BIAS,
    Operation.NN,
    Operation.IN,
    Operation.RPBGRAD,
]

class Problem(Enum):
    NA1D = 0
    NA2D = 1
    NA3D = 2

def naive_op_to_filename(operation: Operation, dim: Problem):
    out = ""
    if operation in [Operation.PN, Operation.PN_BIAS]:
        out += "pointwise_neighborhood"
    elif operation in [Operation.NN]:
        out += "neighborhood_neighborhood"
    elif operation in [Operation.IN]:
        out += "inverse_neighborhood"
    elif operation in [Operation.RPBGRAD]:
        out += "rel_pos_bias"
    else:
        raise ValueError()

    if dim == Problem.NA1D:
        out += "_1d"
    elif dim == Problem.NA2D:
        out += "_2d"
    elif dim == Problem.NA3D:
        out += "_3d"
    else:
        raise ValueError()

    return out 

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
        out += f"  using DConfig = natten::detail::DTypeConfig<{self.natten_name}>;\n"
        return out

# Define supported data types with their minimum architecture requirements
NATTEN_Float = DataType("float", "natten::float32", "float32", 32, "gfx908")
NATTEN_Half = DataType("sycl::half", "natten::float16", "float16", 16, "gfx908")
NATTEN_BFloat = DataType("sycl::bfloat16", "natten::bfloat16", "bfloat16", 16, "gfx90a")

# SYCL kernel templates
KERNEL_DECL_TEMPLATE = """SYCL_EXTERNAL void {NAME}(
    sycl::queue& queue,
    typename {CPP_CLASS}::Params p);
"""

KERNEL_IMPL_TEMPLATE = """SYCL_EXTERNAL void {NAME}(
    sycl::queue& queue,
    typename {CPP_CLASS}::Params p) {{
    queue.submit([&](sycl::handler& cgh) {{
        cgh.parallel_for(
            sycl::nd_range<3>(p.grid_dim, p.block_dim),
            [=](sycl::nd_item<3> item) {{
                if (!p.advance_to_block()) {{
                    return;
                }}
                {CPP_CLASS}::attention_kernel(p, item);
            }});
    }}).wait();
}}
"""

class CArg:
    def __init__(self, dtype: str, name: str):
        self.dtype = dtype
        self.name = name

    def __str__(self):
        return f"{self.dtype} {self.name}"

# Common arguments for all kernels
QUEUE_ARG = [CArg("sycl::queue&", "queue")]

# Dimension-specific arguments
ARGS_1D = [
    CArg("const std::tuple<int32_t>&", "kernel_size"),
    CArg("const std::tuple<int32_t>&", "dilation"),
]

ARGS_2D = [
    CArg("const std::tuple<int32_t, int32_t>&", "kernel_size"),
    CArg("const std::tuple<int32_t, int32_t>&", "dilation"),
]

ARGS_3D = [
    CArg("const std::tuple<int32_t, int32_t, int32_t>&", "kernel_size"),
    CArg("const std::tuple<int32_t, int32_t, int32_t>&", "dilation"),
]

# Operation-specific arguments
PN_COMMON_ARGS = [
    CArg("bool", "is_grad"),
    CArg("void *", "query_ptr"),
    CArg("void *", "key_ptr"),
    CArg("void *", "attn_ptr"),
]

PN_BIAS_COMMON_ARGS = [
    CArg("void *", "query_ptr"),
    CArg("void *", "key_ptr"),
    CArg("void *", "bias_ptr"),
    CArg("void *", "attn_ptr"),
]

NN_COMMON_ARGS = [
    CArg("void *", "attn_ptr"),
    CArg("void *", "value_ptr"),
    CArg("void *", "output_ptr"),
]

IN_COMMON_ARGS = [
    CArg("void *", "grad_out_ptr"),
    CArg("void *", "value_ptr"),
    CArg("void *", "grad_attn_ptr"),
]

RPBGRAD_COMMON_ARGS = [
    CArg("void *", "grad_out_ptr"),
    CArg("void *", "query_ptr"),
    CArg("void *", "key_ptr"),
    CArg("void *", "grad_bias_ptr"),
]

class NaiveNAKernel:
    def __init__(self, operation, dim, dtype, arch):
        self.operation = operation
        self.dim = dim
        self.dtype = dtype
        self.arch = arch
        self.filename = f"{naive_op_to_filename(operation, dim)}_{dtype.name}_{arch}.cpp"
        self.path_to_header = f"natten/naive/{naive_op_to_filename(operation, dim)}.h"

    def method_name(self):
        return f"na{self.dim.value + 1}d_{self.operation.name.lower()}_{self.dtype.name}_{self.arch}"

    def get_args(self):
        args = QUEUE_ARG.copy()
        
        if self.dim == Problem.NA1D:
            args.extend(ARGS_1D)
        elif self.dim == Problem.NA2D:
            args.extend(ARGS_2D)
        elif self.dim == Problem.NA3D:
            args.extend(ARGS_3D)
        
        if self.operation == Operation.PN:
            args.extend(PN_COMMON_ARGS)
        elif self.operation == Operation.PN_BIAS:
            args.extend(PN_BIAS_COMMON_ARGS)
        elif self.operation == Operation.NN:
            args.extend(NN_COMMON_ARGS)
        elif self.operation == Operation.IN:
            args.extend(IN_COMMON_ARGS)
        elif self.operation == Operation.RPBGRAD:
            args.extend(RPBGRAD_COMMON_ARGS)
        
        return args

    def method_decl(self):
        return f"void {self.method_name()}({', '.join(str(arg) for arg in self.get_args())})"

    def header(self):
        header_str = ""
        header_str += self.method_decl()
        header_str += ";\n\n"
        return header_str

    def source(self):
        source_str = ""
        source_str += self.method_decl()
        source_str += " {\n"
        source_str += self.dtype.source()
        source_str += "  using Kernel = natten::naive::detail::NAKernel<DConfig>;\n"
        source_str += "  Kernel::template attention_kernel<"
        source_str += f"{self.dim.value + 1}, {self.operation.value}>"
        source_str += "(queue, kernel_size, dilation"
        
        if self.operation == Operation.PN:
            source_str += ", is_grad, query_ptr, key_ptr, attn_ptr"
        elif self.operation == Operation.PN_BIAS:
            source_str += ", query_ptr, key_ptr, bias_ptr, attn_ptr"
        elif self.operation == Operation.NN:
            source_str += ", attn_ptr, value_ptr, output_ptr"
        elif self.operation == Operation.IN:
            source_str += ", grad_out_ptr, value_ptr, grad_attn_ptr"
        elif self.operation == Operation.RPBGRAD:
            source_str += ", grad_out_ptr, query_ptr, key_ptr, grad_bias_ptr"
        
        source_str += ");\n"
        source_str += "}\n\n"
        return source_str 

    def write_source_file(self, path):
        source_head = []
        source_head += ["#include <iostream>\n"]
        source_head += ["#include <natten/dtypes.h>\n"]
        source_head += [f"#include <{self.path_to_header}>\n"]

        source_head += ["namespace natten { \n"]
        source_head += ["namespace acpp { \n"]
        source_head += ["namespace naive { \n"]

        source_head = "".join(source_head)
        source_body = self.source()

        source_foot = "".join(
            [
                "} \n",
                "} \n",
                "} \n",
                "\n",
            ]
        )
        
        # Add architecture check
        if self.dtype.name in ["sycl::half", "sycl::bfloat16"]:
            arch_check = f"""
        if (queue.get_device().get_info<sycl::info::device::name>().find("{self.arch}") != std::string::npos) {{
            {source_body}
        }} else {{
            std::cerr << "This data type requires {self.arch} or newer architecture." << std::endl;
            exit(EXIT_FAILURE);
        }}
"""
            source_body = arch_check

        filename = f"{path}/{self.filename}"
        with open(filename, "w") as f:
            f.write(source_head)
            f.write(source_body)
            f.write(source_foot) 

class KernelSizeDispatcher:
    def __init__(
        self,
        dtype: DataType,
        operation: Operation,
        dim: Problem,
        arch: str,
    ):
        self.dtype = dtype
        self.operation = operation
        self.dim = dim
        self.arch = arch
        self.name_base = f"na{self.dim.value + 1}d_{operation.name.lower()}_acpp_naive"
        self.name_base += f"_{dtype.name}_{arch}"
        self.name_cc = f"DISPATCH_KERNEL_{self.name_base}"
        self.name_target = f"DISPATCH_DILATION_{self.name_base}"
        self.kernels = []

    def append(self, kernel_size: int):
        self.kernels.append(kernel_size)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(queue, kernel_size, dilation, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        
        for i, ks in enumerate(self.kernels):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            if ks == -1:  # Skip default case
                continue
            dispatcher_str += f"if (kernel_size == {ks})"
            dispatcher_str += " { \\\n"
            dispatcher_str += f"      {self.name_target}_ks_{ks}(queue, dilation, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        
        dispatcher_str += "    else { \\\n"
        dispatcher_str += f"      {self.name_target}_ks_any(queue, dilation, __VA_ARGS__); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();\n\n"
        return dispatcher_str

class DilationDispatcher:
    def __init__(
        self,
        dtype: DataType,
        kernel_size: int,
        operation: Operation,
        dim: Problem,
        arch: str,
    ):
        self.dtype = dtype
        self.operation = operation
        self.dim = dim
        self.arch = arch
        self.kernel_size = kernel_size
        self.name_base = f"na{self.dim.value + 1}d_{operation.name.lower()}_acpp_naive"
        self.name_base += f"_{dtype.name}_{arch}_ks_{kernel_size}"
        self.name_cc = f"DISPATCH_DILATION_{self.name_base}"
        self.name_target = f"naive::{self.name_base}"
        self.dilations = []

    def append(self, dilation: int):
        self.dilations.append(dilation)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(queue, dilation, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        
        for i, di in enumerate(self.dilations):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            if di == -1:  # Skip default case
                continue
            dispatcher_str += f"if (dilation == {di})"
            dispatcher_str += " { \\\n"
            dispatcher_str += f"      {self.name_target}_di_{di}(queue, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        
        dispatcher_str += "    else { \\\n"
        dispatcher_str += f"      {self.name_target}_di_any(queue, __VA_ARGS__); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();\n\n"
        return dispatcher_str 

class DataTypeDispatcher:
    def __init__(
        self,
        operation: Operation,
        dim: Problem,
        arch: str,
    ):
        self.dtypes: List[DataType] = []
        self.operation = operation
        self.dim = dim
        self.arch = arch
        self.name_base = f"na{self.dim.value + 1}d_{operation.name.lower()}_acpp_naive_{arch}"
        self.name_cc = f"DISPATCH_DTYPE_{self.name_base}"
        self.name_target = f"DISPATCH_KERNEL_{self.name_base}"

    def append(self, dtype: DataType):
        if dtype.min_arch <= self.arch:  # Only add if architecture supports this dtype
            self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(dtype, kernel_size, dilation, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f'if (dtype == "{dtype.natten_name}")'
            dispatcher_str += " { \\\n"
            dispatcher_str += f"      {self.name_target}_{dtype.short_name}(kernel_size, dilation, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN kernel launch failed!" \\\n'
        dispatcher_str += f'                << "{self.name_base} does not support this data type." \\\n'
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();\n\n"
        return dispatcher_str 

class DeviceDispatcher:
    def __init__(self, operation: Operation, dim: Problem):
        self.operation = operation
        self.dim = dim
        self.name_base = f"na{self.dim.value + 1}d_{operation.name.lower()}_acpp_naive"
        self.name_cc = f"LAUNCH_{self.name_base}"
        self.targets = {}  # arch -> DataTypeDispatcher name_cc mapping
        self.devices: List[str] = []

    def append(self, arch: str):
        if arch not in self.devices:
            self.devices.append(arch)
            self.targets[arch] = f"DISPATCH_DTYPE_{self.name_base}_{arch}"

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(device, dtype, kernel_size, dilation, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        
        for i, arch in enumerate(self.devices):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f'if (device == "{arch}")'
            dispatcher_str += " { \\\n"
            dispatcher_str += f"      {self.targets[arch]}(dtype, kernel_size, dilation, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN kernel launch failed!" \\\n'
        dispatcher_str += '                << "Naive neighborhood attention is not implemented for this device." \\\n'
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();\n\n"
        return dispatcher_str 

def write_combined_header_file(path: str, filename: str, kernels: List[NaiveNAKernel]):
    header_head = []
    header_head += ["#pragma once\n"]
    header_head += ["#include <natten/cuda/naive/naive.h>\n"]
    header_head += ["\nnamespace natten {\n"]
    header_head += ["namespace acpp {\n"]
    header_head += ["namespace naive {\n"]

    header_head = "".join(header_head)
    header_body = "".join(kernel.header() for kernel in kernels)
    header_foot = "".join(["}\n", "}\n", "}\n", "\n"])

    with open(f"{path}/{filename}", "w") as f:
        f.write(header_head)
        f.write(header_body)
        f.write(header_foot)

@click.command()
@click.option("--output-dir", default=DEFAULT_OUTPUT_DIR)
def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate kernels for each operation, dimension, and data type
    kernels = []
    dispatchers = {}
    
    for operation, dim in product(ALL_OPS, [Problem.NA1D, Problem.NA2D, Problem.NA3D]):
        device_dispatcher = DeviceDispatcher(operation, dim)
        
        for arch in ["gfx908", "gfx90a"]:
            dtype_dispatcher = DataTypeDispatcher(operation, dim, arch)
            
            for dtype in [NATTEN_Float, NATTEN_Half, NATTEN_BFloat]:
                if dtype.min_arch <= arch:  # Only add if architecture supports this dtype
                    kernel = NaiveNAKernel(operation, dim, dtype, arch)
                    kernels.append(kernel)
                    dtype_dispatcher.append(dtype)
            
            if dtype_dispatcher.dtypes:  # Only add if there are supported dtypes
                device_dispatcher.append(arch)
        
        if device_dispatcher.devices:  # Only add if there are supported devices
            dispatchers[f"{operation.name}_{dim.value}"] = device_dispatcher
    
    # Write kernel source files
    for kernel in kernels:
        kernel.write_source_file(output_dir)
    
    # Write dispatcher header
    dispatcher_str = ""
    for dispatcher in dispatchers.values():
        dispatcher_str += dispatcher.get_dispatcher()
    
    with open(f"{output_dir}/naive_dispatch.h", "w") as f:
        f.write("#pragma once\n\n")
        f.write(dispatcher_str)

if __name__ == "__main__":
    main() 