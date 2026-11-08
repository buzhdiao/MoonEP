"""
moonep build script. Compiles the CUDA extension `moonep._C` (the pybind11
module living in csrc/bindings.cu) into the `moonep` package so it can be
imported as `from moonep import _C` after either:

  pip install -e .                              # editable install (preferred)

or, for a quick in-place build without installing:

  python setup.py build_ext --inplace

The .so lands at `moonep/_C.<abi-tag>.so`.
"""
# 中文说明：本脚本是 moonep 的构建入口，把 CUDA 扩展 `moonep._C`（pybind11 模块，
# 源码位于 csrc/bindings.cu）编译进 moonep 包。两种用法：
#   1) `pip install -e .`：可编辑安装（推荐），安装后即可 `from moonep import _C`；
#   2) `python setup.py build_ext --inplace`：不安装、就地快速构建；
# 产物 .so 落在 `moonep/_C.<abi-tag>.so`。

# 导入 os 模块，用于路径拼接与目录处理
import os

# find_packages：自动发现包目录；setup：setuptools 构建入口
from setuptools import find_packages, setup
# PyTorch 提供的扩展构建设施：CUDAExtension 声明一个 CUDA 扩展模块，
# BuildExtension 接管 build_ext 命令（负责调用 nvcc、处理 .cu 源文件）
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# 仓库根目录 = setup.py 所在目录的绝对路径，供扩展的 include 路径引用
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


# nvcc（设备侧）编译选项列表，逐项含义如下：
nvcc_flags = [
    # "-O3"：最高级别优化
    "-O3",
    # "--use_fast_math"：启用快速数学近似（用精度换吞吐）
    "--use_fast_math",
    # "-std=c++20"：使用 C++20 标准
    "-std=c++20",
    # "--expt-extended-lambda"：允许在 device 代码中使用扩展 lambda
    "--expt-extended-lambda",
    # "--expt-relaxed-constexpr"：放宽 constexpr 限制，host 常量可直接用于 device 代码
    "--expt-relaxed-constexpr",
    # "-forward-unknown-to-host-compiler"：未知选项透传给宿主 C++ 编译器
    "-forward-unknown-to-host-compiler",
    # "-Xcompiler=-Wno-psabi"：传给 g++，抑制 ABI 变更告警
    "-Xcompiler=-Wno-psabi",
    # "-Xcompiler=-fno-strict-aliasing"：传给 g++，关闭严格别名，避免指针重解释被误优化
    "-Xcompiler=-fno-strict-aliasing",
    # "-DNDEBUG"：定义 NDEBUG，关闭 assert（发布构建）
    "-DNDEBUG",
    # "-lineinfo"：生成行号信息，便于 NCU/nsys 性能分析
    "-lineinfo",
    # "-diag-suppress=3189"：屏蔽 NVCC 3189 号诊断（告警噪音）
    "-diag-suppress=3189",
    # "-ftemplate-backtrace-limit=0"：模板报错不截断回溯（CuTe 深层模板调试友好）
    "-ftemplate-backtrace-limit=0",
    # 禁用内置 half 运算符重载，避免与项目自定义实现冲突
    "-D__CUDA_NO_HALF_OPERATORS__",
    # 禁用内置 half 隐式转换
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    # 禁用内置 bf16 隐式转换
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    # 禁用内置 half2 运算符重载
    "-D__CUDA_NO_HALF2_OPERATORS__",
]

# 宿主（C++ 侧）编译选项列表，逐项含义如下：
cxx_flags = [
    # "-O3"：最高级别优化
    "-O3",
    # "-fno-strict-aliasing"：关闭严格别名规则
    "-fno-strict-aliasing",
    # "-Wno-psabi"：抑制 ABI 变更告警
    "-Wno-psabi",
]


# 待构建的扩展模块列表：把 csrc/bindings.cu 编译为可导入的 moonep._C
ext_modules = [
    # name：扩展的 Python 导入名 moonep._C
    # sources：唯一源文件 csrc/bindings.cu（pybind11 绑定 + 内核封装）
    # include_dirs：头文件搜索路径为仓库 csrc 目录
    # libraries：链接 CUDA 驱动库 libcuda（供驱动 API 调用）
    # extra_compile_args：分别给宿主编译器与 nvcc 传上面的选项
    CUDAExtension(
        name="moonep._C",
        sources=[
            os.path.join("csrc", "bindings.cu"),
        ],
        include_dirs=[
            os.path.join(REPO_ROOT, "csrc"),
        ],
        libraries=["cuda"],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    ),
]


# setuptools 构建入口：包名、版本、依赖与构建命令定制
setup(
    # name/version：发行包名 moonep，版本 0.0.1
    name="moonep",
    version="0.0.1",
    # 只收集 moonep 及其子包
    packages=find_packages(include=["moonep", "moonep.*"]),
    # 运行期依赖：CuTe DSL（规划/内核 JIT 编译使用），锁定 4.6.2 版本
    install_requires=[
        "nvidia-cutlass-dsl==4.6.2",
    ],
    # 上面定义的 CUDA 扩展
    ext_modules=ext_modules,
    # 用 PyTorch 的 BuildExtension 执行 build_ext（负责编译 .cu）
    cmdclass={"build_ext": BuildExtension},
)
