// pybind11 绑定：把 nvl_shared_buffer.cuh 中的 NVLink/VMM 共享缓冲区接口导出为 Python 模块 _C
#include <torch/extension.h>
// 引入 torch 扩展头文件（pybind11 与 ATen）
#include "nvl_shared_buffer.cuh"
// 引入本项目头文件，包含全部 NVLink/VMM 实现函数

/**
 * Return the VMM allocation granularity in bytes for the current device.
 */
// 返回当前设备的 VMM 分配粒度（字节）
int64_t get_vmm_granularity() {
    // 包装函数：取两种句柄类型粒度中的最大值
    int device_id;
    // 用于接收当前 CUDA 设备编号
    CUDACHECK(cudaGetDevice(&device_id));
    // 查询当前设备编号，出错则打印并退出
    return static_cast<int64_t>(nvl_granularity_max(device_id));
    // 返回该设备的最大分配粒度
}

PYBIND11_MODULE(_C, m) {
    // 定义 Python 模块 _C，m 为模块对象
    m.attr("FABRIC_HANDLE_BYTES") = kFabricHandleBytes;
    // 导出 fabric 句柄字节长度常量（64）
    m.def("nvl_dist_alloc", &nvl_dist_alloc,
          // 绑定 nvl_dist_alloc：分配本地 chunk 并导出共享句柄
          pybind11::arg("shape"), pybind11::arg("dtype"),
          // 绑定参数名 shape 与 dtype
          pybind11::arg("use_fabric") = false,
          // 绑定参数 use_fabric，默认 false（用 fd）
          "Allocate a local chunk and export it for sharing, as a POSIX fd "
          // 文档字符串：分配本地 chunk 并导出以共享，形式为 POSIX fd
          "(same node) or a 64-byte fabric handle (same NVLink domain).");
          // （同节点）或 64 字节 fabric 句柄（同 NVLink 域）
    m.def("nvl_dist_map", &nvl_dist_map,
          // 绑定 nvl_dist_map：把所有 rank 的 chunk 映射为连续 VA
          pybind11::arg("chunk_shape"), pybind11::arg("dtype"),
          // 绑定参数名 chunk_shape 与 dtype
          pybind11::arg("shareables"), pybind11::arg("local_rank"),
          // 绑定参数名 shareables 与 local_rank
          pybind11::arg("world_size"), pybind11::arg("use_fabric") = false,
          // 绑定参数名 world_size 与 use_fabric（默认 false）
          "Map all ranks' chunks into a contiguous VA region (all RW).");
          // 文档字符串：把所有 rank 的 chunk 映射进连续 VA 区（全部可读写）
    m.def("nvl_fabric_supported", &nvl_fabric_supported,
          // 绑定 nvl_fabric_supported：查询 fabric 句柄支持
          "Whether the current device can export/import fabric handles.");
          // 文档字符串：当前设备能否导出/导入 fabric 句柄
    m.def("get_vmm_granularity", &get_vmm_granularity,
          // 绑定 get_vmm_granularity：查询 VMM 粒度
          "Return VMM allocation granularity in bytes.");
          // 文档字符串：返回 VMM 分配粒度（字节）
    m.def("get_multicast_granularity", &get_multicast_granularity,
          // 绑定 get_multicast_granularity：查询组播对齐粒度
          pybind11::arg("num_devices"),
          // 绑定参数名 num_devices
          "Return multicast recommended allocation granularity in bytes.");
          // 文档字符串：返回组播建议分配粒度（字节）
    m.def("nvl_multicast_supported", &nvl_multicast_supported,
          // 绑定 nvl_multicast_supported：查询组播支持
          "Whether the current device supports multicast (NVSwitch SHARP).");
          // 文档字符串：当前设备是否支持组播（NVSwitch SHARP）
    m.def("nvl_multicast_create", &nvl_multicast_create,
          // 绑定 nvl_multicast_create：root 创建组播对象
          pybind11::arg("size_bytes"), pybind11::arg("num_devices"),
          // 绑定参数名 size_bytes 与 num_devices
          pybind11::arg("use_fabric") = false,
          // 绑定参数 use_fabric，默认 false
          "Root-only: create a multicast object and export it for sharing.");
          // 文档字符串：仅 root：创建组播对象并导出以共享
    m.def("nvl_multicast_import", &nvl_multicast_import,
          // 绑定 nvl_multicast_import：非 root 导入组播对象
          pybind11::arg("shareable"), pybind11::arg("use_fabric") = false,
          // 绑定参数名 shareable 与 use_fabric（默认 false）
          "Non-root: import a multicast object from the root's handle.");
          // 文档字符串：非 root：从 root 的句柄导入组播对象
    m.def("nvl_multicast_add_device", &nvl_multicast_add_device,
          // 绑定 nvl_multicast_add_device：把本设备加入组播对象
          pybind11::arg("mc_handle"),
          // 绑定参数名 mc_handle
          "Add the current device to a multicast object (before bind).");
          // 文档字符串：把当前设备加入组播对象（bind 之前调用）
    m.def("nvl_multicast_bind_map", &nvl_multicast_bind_map,
          // 绑定 nvl_multicast_bind_map：绑定本地内存并映射组播 VA
          pybind11::arg("mc_handle"), pybind11::arg("owned_mem_handle"),
          // 绑定参数名 mc_handle 与 owned_mem_handle
          pybind11::arg("size_bytes"), pybind11::arg("num_devices"),
          // 绑定参数名 size_bytes 与 num_devices
          "Bind the local chunk's owned physical memory (nvl_dist_alloc handle) "
          // 文档字符串：把本地 chunk 的持有物理内存（nvl_dist_alloc 句柄）
          "to the multicast object and map a multicast VA; returns a keepalive "
          // 绑定到组播对象并映射组播 VA；返回一个保活的
          "int32 tensor whose data_ptr is the multimem address.");
          // int32 张量，其 data_ptr 即 multimem 地址
    m.def("nvl_release_mem_handle", &nvl_release_mem_handle,
          // 绑定 nvl_release_mem_handle：释放持有句柄
          pybind11::arg("mem_handle"),
          // 绑定参数名 mem_handle
          "Release an owned mem handle returned by nvl_dist_alloc.");
          // 文档字符串：释放 nvl_dist_alloc 返回的持有 mem 句柄
}
