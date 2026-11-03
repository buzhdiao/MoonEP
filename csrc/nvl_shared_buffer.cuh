// NVLink 共享缓冲区核心实现：基于 CUDA VMM 的跨进程/跨 rank 显存分配、映射与组播（NVSwitch SHARP）注册
#pragma once
// 头文件保护：仅包含一次

#include <algorithm>
// 引入 std::max 等算法
#include <cstdint>
// 引入定宽整数类型（int64_t 等）
#include <cstdio>
// 引入 fprintf 等标准输入输出
#include <cstdlib>
// 引入 exit 与 EXIT_FAILURE
#include <cstring>
// 引入 memcpy 等内存操作
#include <tuple>
// 引入 std::tuple 用于多返回值
#include <vector>
// 引入 std::vector

#include <cuda.h>
// 引入 CUDA 驱动 API（cuMemCreate 等 VMM 接口）
#include <cuda_runtime.h>
// 引入 CUDA 运行时 API（cudaGetDevice 等）

#include <ATen/ATen.h>
// 引入 ATen 张量库
#include <c10/cuda/CUDAGuard.h>
// 引入 CUDAGuard（设置当前设备）与流查询

// CUresult 错误检查宏（驱动 API）：出错时打印文件、行号与错误串并直接退出。
// 注意：宏体使用 `\` 续行，行尾绝不能添加注释。
#ifndef CUCHECK
// 防止重复定义
#define CUCHECK(cmd) do {                                      \
    CUresult err = cmd;                                            \
    if (err != CUDA_SUCCESS) {                                     \
        const char *errStr;                                        \
        cuGetErrorString(err, &errStr);                            \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",        \
            __FILE__, __LINE__, errStr);                           \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)
// do-while 包裹保证宏在任何语句位置都安全
#endif
// 结束 CUCHECK 定义保护

// cudaError_t 错误检查宏（运行时 API）：语义同 CUCHECK。
// 注意：宏体使用 `\` 续行，行尾绝不能添加注释。
#ifndef CUDACHECK
// 防止重复定义
#define CUDACHECK(cmd) do {                                    \
    cudaError_t err = cmd;                                         \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",        \
            __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)
// do-while 包裹保证宏在任何语句位置都安全
#endif
// 结束 CUDACHECK 定义保护

// 查询当前设备是否支持 fabric 句柄（先查属性，再实际试建一块内存验证）
static inline bool nvl_fabric_supported() {
    // 静态内联函数，头文件内可重复定义
    int device_id;
    // 用于接收当前 CUDA 设备编号
    CUDACHECK(cudaGetDevice(&device_id));
    // 查询当前设备编号
    CUdevice cu_device;
    // 驱动 API 的设备句柄
    CUCHECK(cuDeviceGet(&cu_device, device_id));
    // 由编号取得设备句柄
    int supported = 0;
    // 属性查询结果：是否支持 fabric 句柄
    CUresult err = cuDeviceGetAttribute(&supported,
        // 查询设备属性 HANDLE_TYPE_FABRIC_SUPPORTED
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cu_device);
        // 指定属性名与设备
    // Older drivers do not know the attribute at all.
    // 旧驱动可能完全识别不了该属性。
    if (err != CUDA_SUCCESS || !supported) return false;
    // 查询失败或不支持都返回 false

    CUmemAllocationProp prop = {};
    // 构造分配属性描述（零初始化）
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // 分配类型：锁定（设备）内存
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // 物理位置：设备内存
    prop.location.id = device_id;
    // 指定设备编号
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    // 请求 fabric 句柄类型
    size_t size;
    // 用于接收建议分配粒度
    if (cuMemGetAllocationGranularity(
            // 查询该属性组合的建议粒度
            &size, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS)
            // 失败则不支持
        return false;
        // 返回 false

    CUmemGenericAllocationHandle handle;
    // 试分配用的句柄
    if (cuMemCreate(&handle, size, &prop, 0) != CUDA_SUCCESS) return false;
    // 实际试建一块内存，失败则不支持
    CUCHECK(cuMemRelease(handle));
    // 立即释放试建的内存
    return true;
    // 全部通过，支持 fabric 句柄
}

// 查询指定设备在给定句柄类型下的建议分配粒度（字节）
static inline size_t nvl_granularity_for(int device_id,
                                         CUmemAllocationHandleType ht) {
    // 设备编号与句柄类型（fd 或 fabric）
    CUmemAllocationProp prop = {};
    // 构造分配属性描述（零初始化）
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // 分配类型：锁定（设备）内存
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // 物理位置：设备内存
    prop.location.id = device_id;
    // 指定设备编号
    prop.requestedHandleTypes = ht;
    // 指定句柄类型

    size_t granularity;
    // 用于接收粒度查询结果
    CUCHECK(cuMemGetAllocationGranularity(
        // 查询建议分配粒度
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        // 出错则打印并退出
    return granularity;
    // 返回粒度（字节）
}

// 查询指定设备可能用到的最大分配粒度（fd 与 fabric 两种类型的最大值）
static inline size_t nvl_granularity_max(int device_id) {
    // 设备编号
    size_t gran = nvl_granularity_for(
        // 先取 fd 类型的粒度
        device_id, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
        // 指定 fd 句柄类型
    if (nvl_fabric_supported()) {
        // 若设备支持 fabric 句柄
        gran = std::max(
            // 取两者中的较大者
            gran, nvl_granularity_for(device_id, CU_MEM_HANDLE_TYPE_FABRIC));
            // 再取 fabric 类型的粒度
    }
    return gran;
    // 返回最大粒度
}

// 根据是否使用 fabric 返回对应的 CUDA 句柄类型枚举
static inline CUmemAllocationHandleType nvl_handle_type(bool use_fabric) {
    // 是否使用 fabric
    return use_fabric ? CU_MEM_HANDLE_TYPE_FABRIC
        // fabric 模式返回 FABRIC 类型
                      : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        // 否则返回 POSIX fd 类型
}

// 预处理函数：校验形状、计算未对齐字节数与对齐后字节数，并取当前设备编号。
// 返回 (实际字节数, 对齐后分配字节数, 设备编号)
static inline std::tuple<size_t, size_t, int> nvl_prepare(
    // 返回三元组
    const std::vector<int64_t> &shape,
    // 张量形状
    at::ScalarType dtype,
    // 元素类型
    bool use_fabric
    // 是否使用 fabric 句柄
) {
    TORCH_CHECK(!shape.empty(), "Shape must be non-empty");
    // 校验形状非空，否则抛错

    int device_id;
    // 用于接收当前 CUDA 设备编号
    CUDACHECK(cudaGetDevice(&device_id));
    // 查询当前设备编号

    size_t size = c10::elementSize(dtype);
    // 起始为单个元素的字节大小
    for (auto dim : shape) {
        // 遍历每个维度
        TORCH_CHECK(dim > 0, "Size dimensions must be positive");
        // 校验维度均为正数
        size *= static_cast<size_t>(dim);
        // 累乘得到总字节数
    }

    size_t granularity = nvl_granularity_for(
        // 查询当前句柄类型的分配粒度
        device_id, nvl_handle_type(use_fabric));
        // 根据共享方式选句柄类型
    size_t allocated_size = (size + granularity - 1) / granularity * granularity;
    // 向上取整到粒度整数倍

    return {size, allocated_size, device_id};
    // 返回三元组
}

// 把一段已映射的 VMM 虚拟地址包装成 ATen 张量（deleter 负责解除映射并释放 VA）
static inline at::Tensor make_vmm_tensor(
    // 静态内联函数
    void *ptr,
    // 已映射的 VMM 虚拟地址
    size_t nbytes,
    // 实际有效字节数（张量 numel * 元素大小）
    size_t allocated_size,
    // 对齐后分配的总字节数（unmap/free 用）
    int device_id,
    // 所属设备编号
    const std::vector<int64_t> &shape,
    // 张量形状
    at::ScalarType dtype
    // 元素类型
) {
    auto deleter = [device_id, ptr, allocated_size](void *p) {
        // Storage 释放时调用的删除器（捕获设备、指针与大小）
        if (!p) return;
        // 空指针直接返回
        c10::cuda::CUDAGuard guard(device_id);
        // 切换到分配时的设备上下文
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream().stream());
        // 同步当前流，确保没有 kernels 还在用这块内存
        cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), allocated_size);
        // 解除虚拟地址的物理映射
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), allocated_size);
        // 释放保留的虚拟地址区间
    };

    auto storage = c10::Storage(
        // 构造 ATen Storage，持有 VMM 指针与删除器
        c10::Storage::use_byte_size_t(),
        // 标记按字节指定大小
        static_cast<int64_t>(nbytes),
        // 有效字节数
        at::InefficientStdFunctionContext::makeDataPtr(
            // 用 std::function 删除器构造 DataPtr
            ptr, std::move(deleter), c10::Device(c10::kCUDA, device_id)),
            // 指针、删除器与设备
        /*allocator=*/nullptr,
        // 无分配器（VMM 内存不走常规分配）
        /*resizable=*/false);
        // Storage 不可改变大小

    auto impl = c10::make_intrusive<at::TensorImpl>(
        // 构造 TensorImpl（张量实现层）
        std::move(storage),
        // 传入上面构造的 Storage
        c10::DispatchKeySet(c10::DispatchKey::CUDA),
        // 派发键：CUDA
        c10::scalarTypeToTypeMeta(dtype));
        // 元素类型元信息
    impl->set_sizes_contiguous(shape);
    // 设置为给定形状的连续布局
    return at::Tensor(std::move(impl));
    // 包装为张量返回
}

// fabric 句柄的字节长度常量（CUmemFabricHandle::data 的大小，64 字节）
static constexpr int64_t kFabricHandleBytes =
    // 编译期常量
    static_cast<int64_t>(sizeof(CUmemFabricHandle::data));
    // 取内部数据数组的大小

// Export an allocation (or multicast object) as a CPU tensor matching
// `use_fabric`: uint8[64] holding a CUmemFabricHandle, or int64[] (a scalar
// tensor) holding a POSIX fd.
// 把一个分配（或组播对象）导出为 CPU 张量：fabric 模式返回保存 CUmemFabricHandle 的
// uint8[64]，fd 模式返回保存 POSIX fd 的 int64 标量张量。
static inline at::Tensor nvl_export_shareable(
    // 静态内联函数
    CUmemGenericAllocationHandle handle, bool use_fabric
    // 待导出的通用分配句柄与共享方式
) {
    if (use_fabric) {
        // fabric 模式
        CUmemFabricHandle fabric = {};
        // 零初始化的 fabric 句柄结构
        CUCHECK(cuMemExportToShareableHandle(
            // 导出为 fabric 共享句柄
            &fabric, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
            // 写入 fabric 结构
        auto out = at::empty({kFabricHandleBytes},
                             // 分配长度 64 的输出张量
                             at::TensorOptions().dtype(at::kByte));
                             // 类型为 uint8
        std::memcpy(out.data_ptr(), fabric.data, sizeof(fabric.data));
        // 把句柄字节拷入张量
        return out;
        // 返回 uint8[64] 张量
    }
    int fd = -1;
    // fd 模式：接收导出的文件描述符
    CUCHECK(cuMemExportToShareableHandle(
        // 导出为 POSIX fd
        &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
        // 指定 fd 句柄类型
    TORCH_CHECK(fd >= 0, "cuMemExportToShareableHandle returned invalid fd");
    // 校验 fd 有效
    return at::scalar_tensor(fd, at::TensorOptions().dtype(at::kLong));
    // 返回 int64 标量张量
}

// 校验共享句柄张量的设备、连续性与形状/类型是否符合指定共享方式
static inline void nvl_check_handles(
    // 静态内联函数
    const at::Tensor &handles, int64_t world_size, bool use_fabric
    // 句柄张量、rank 数与共享方式
) {
    TORCH_CHECK(handles.device().is_cpu() && handles.is_contiguous(),
        // 校验为连续的 CPU 张量
        "shareable handles must be a contiguous CPU tensor, got device=",
        // 错误信息第一行
        handles.device(), " contiguous=", handles.is_contiguous());
        // 错误信息第二行（附设备与连续性）
    if (use_fabric) {
        // fabric 模式的校验
        TORCH_CHECK(handles.scalar_type() == at::kByte && handles.dim() == 2
                    // 要求 uint8 二维张量
                    && handles.size(0) == world_size
                    // 第一维等于 rank 数
                    && handles.size(1) == kFabricHandleBytes,
                    // 第二维等于句柄字节数（64）
            "fabric handles must be uint8[", world_size, ", ",
            // 错误信息第一行
            kFabricHandleBytes, "], got ", handles.scalar_type(), handles.sizes());
            // 错误信息第二行（附实际类型与形状）
    } else {
        // fd 模式的校验
        TORCH_CHECK(handles.scalar_type() == at::kLong
                    // 要求 int64 类型
                    && handles.numel() == world_size,
                    // 元素数等于 rank 数
            "fd handles must be int64 with ", world_size, " elements, got ",
            // 错误信息（附期望与实际）
            handles.scalar_type(), handles.sizes());
            // 附实际类型与形状
    }
}

// Import row `index` of a handle tensor already validated by nvl_check_handles.
// 从（已经 nvl_check_handles 校验过的）句柄张量中导入第 index 行，返回通用分配句柄。
static inline CUmemGenericAllocationHandle nvl_import_shareable(
    // 静态内联函数
    const at::Tensor &handles, int64_t index, bool use_fabric
    // 句柄张量、行号与共享方式
) {
    CUmemGenericAllocationHandle handle;
    // 接收导入结果的句柄
    if (use_fabric) {
        // fabric 模式
        CUmemFabricHandle fabric = {};
        // 零初始化的 fabric 句柄结构
        std::memcpy(fabric.data,
                    // 从张量拷贝 64 字节句柄数据
                    handles.const_data_ptr<uint8_t>() + index * kFabricHandleBytes,
                    // 第 index 行的起始地址
                    sizeof(fabric.data));
                    // 拷贝整个句柄长度
        CUCHECK(cuMemImportFromShareableHandle(
            // 从 fabric 句柄导入
            &handle, &fabric, CU_MEM_HANDLE_TYPE_FABRIC));
            // 指定 fabric 类型
    } else {
        // fd 模式
        int64_t fd = handles.const_data_ptr<int64_t>()[index];
        // 取第 index 个 fd
        TORCH_CHECK(fd >= 0, "invalid fd ", fd, " at index ", index);
        // 校验 fd 有效
        CUCHECK(cuMemImportFromShareableHandle(
            // 从 fd 导入
            &handle, reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
            // 把 fd 转成指针形式传入
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
            // 指定 fd 句柄类型
    }
    return handle;
    // 返回导入的通用分配句柄
}

// Returns (keepalive VA tensor, shareable handle, owned mem handle as int64).
// The owned mem handle is not released here (kept for multicast cuMulticastBindMem);
// the caller must call nvl_release_mem_handle when done (buffers that do not need
// multicast release it immediately, matching the old behavior; buffers that need
// multicast release it after bind).
// The shareable handle is what peers import: an int64 scalar tensor holding an
// fd the caller must close once all peers have imported it, or a uint8[64]
// fabric handle that needs no cleanup.
// 返回 (保活 VA 张量, 可共享句柄, 以 int64 表示的持有 mem 句柄)。持有句柄在此不释放
// （留给组播 cuMulticastBindMem 用）；用完后调用者必须调用 nvl_release_mem_handle。
// 可共享句柄供各 peer 导入：fd 模式是 int64 标量张量（所有 peer 导入后调用者需关闭
// fd），fabric 模式是 uint8[64]（无需清理）。
inline std::tuple<at::Tensor, at::Tensor, int64_t> nvl_dist_alloc(
    // 分配本地 chunk 并导出共享句柄
    const std::vector<int64_t> &chunk_shape,
    // 单个 chunk 的形状
    at::ScalarType dtype,
    // 元素类型
    bool use_fabric
    // 是否使用 fabric 句柄
) {
    auto [nbytes, allocated_size, device_id] =
        // 结构化绑定：实际字节数、对齐后字节数、设备编号
        nvl_prepare(chunk_shape, dtype, use_fabric);
        // 做校验与尺寸计算

    CUmemAllocationProp prop = {};
    // 构造分配属性描述（零初始化）
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // 分配类型：锁定（设备）内存
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // 物理位置：设备内存
    prop.location.id = device_id;
    // 指定设备编号
    prop.requestedHandleTypes = nvl_handle_type(use_fabric);
    // 根据共享方式设置句柄类型

    CUmemGenericAllocationHandle mem_handle;
    // 新分配物理内存的持有句柄
    CUCHECK(cuMemCreate(&mem_handle, allocated_size, &prop, 0));
    // 按对齐后大小创建物理分配

    CUdeviceptr dptr;
    // 保留的虚拟地址
    CUCHECK(cuMemAddressReserve(&dptr, allocated_size, 0, 0, 0));
    // 保留一段 VA 区间
    CUCHECK(cuMemMap(dptr, allocated_size, 0, mem_handle, 0));
    // 把物理内存映射到该 VA

    CUmemAccessDesc access_desc = {};
    // 访问权限描述（零初始化）
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // 权限作用位置：设备
    access_desc.location.id = device_id;
    // 指定设备编号
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // 授予读写权限
    CUCHECK(cuMemSetAccess(dptr, allocated_size, &access_desc, 1));
    // 使本设备获得读写访问

    // Export for other processes to import. Do not release mem_handle: keep the
    // owned generic handle for multicast BindMem. The caller is responsible for
    // nvl_release_mem_handle. The physical memory is referenced by the unicast
    // map (keepalive) and (optionally) multicast; it is only truly freed after
    // all unmaps once the handle is released.
    // 导出给其他进程导入。不要释放 mem_handle：保留持有句柄给组播 BindMem 用。
    // 由调用者负责 nvl_release_mem_handle。物理内存被单播映射（keepalive）和（可选的）
    // 组播引用；释放句柄后也要等全部 unmap 才会真正释放。
    auto shareable = nvl_export_shareable(mem_handle, use_fabric);
    // 导出可共享句柄

    auto keepalive = make_vmm_tensor(
        // 把 VA 包装成张量，删除器负责释放映射
        reinterpret_cast<void *>(dptr), nbytes, allocated_size,
        // 指针、有效字节数与分配大小
        device_id, chunk_shape, dtype);
        // 设备、形状与类型

    return {std::move(keepalive), std::move(shareable),
            // 返回保活张量与共享句柄
            static_cast<int64_t>(mem_handle)};
            // 持有句柄以 int64 形式返回
}

// Release the owned mem handle returned by nvl_dist_alloc (the physical memory
// stays alive via the existing maps).
// 释放 nvl_dist_alloc 返回的持有 mem 句柄（物理内存靠已有映射继续存活）。
static inline void nvl_release_mem_handle(int64_t mem_handle_u64) {
    // 以 int64 传入的持有句柄
    CUCHECK(cuMemRelease(static_cast<CUmemGenericAllocationHandle>(mem_handle_u64)));
    // 还原句柄类型并释放
}

/**
 * Map all chunk handles into a contiguous VA region.
 * Key difference from reference: ALL chunks get RW access (not just local_rank),
 * because dispatch needs to write to remote ranks' regions.
 *
 * `shareables` are the handles exported by each rank via nvl_dist_alloc and
 * routed to this process by the caller: POSIX fds (already valid here, the
 * caller may close them once imported) or 64-byte fabric blobs.
 */
// 把所有 rank 的 chunk 句柄映射进一段连续 VA 区。与参考实现的关键区别：
// 所有 chunk 都授予读写权限（不只本地 rank），因为 dispatch 需要写远端 rank 的区域。
// shareables 是各 rank 经 nvl_dist_alloc 导出、由调用者传来的句柄：
// POSIX fd（本进程内有效，导入后调用者可关闭）或 64 字节 fabric 句柄。
inline at::Tensor nvl_dist_map(
    // 把全组 chunk 映射为连续 VA 的函数
    const std::vector<int64_t> &chunk_shape,
    // 单个 chunk 的形状
    at::ScalarType dtype,
    // 元素类型
    const at::Tensor &shareables,
    // 全部 rank 的共享句柄张量
    int64_t local_rank,
    // 本 rank 编号（权限逻辑对所有 chunk 一致）
    int64_t world_size,
    // rank 总数
    bool use_fabric
    // 是否使用 fabric 句柄
) {
    nvl_check_handles(shareables, world_size, use_fabric);
    // 先校验句柄张量的合法性

    auto [chunk_nbytes, chunk_allocated_size, device_id] =
        // 结构化绑定：单 chunk 字节数、对齐后大小、设备编号
        nvl_prepare(chunk_shape, dtype, use_fabric);
        // 做校验与尺寸计算

    TORCH_CHECK(chunk_allocated_size == chunk_nbytes,
        // 校验 chunk 字节数已对齐到粒度
        "Chunk byte size (", chunk_nbytes, ") must be aligned to VMM granularity (",
        // 错误信息第一行
        chunk_allocated_size, "). Adjust chunk dimensions so that the total bytes "
        // 错误信息第二行
        "are a multiple of the allocation granularity.");
        // 错误信息第三行（提示调整维度使总字节数为粒度整数倍）

    size_t total_allocated_size = chunk_allocated_size * world_size;
    // 全部 chunk 的对齐总字节数
    size_t total_nbytes = chunk_nbytes * world_size;
    // 全部 chunk 的有效总字节数

    CUdeviceptr dptr;
    // 保留的连续 VA 起始地址
    CUCHECK(cuMemAddressReserve(&dptr, total_allocated_size, 0, 0, 0));
    // 一次性保留整段 VA 区间

    for (int64_t i = 0; i < world_size; i++) {
        // 逐个导入并映射每个 rank 的 chunk
        CUmemGenericAllocationHandle mem_handle =
            // 第 i 个 rank 的分配句柄
            nvl_import_shareable(shareables, i, use_fabric);
            // 从共享句柄张量导入

        CUdeviceptr chunk_va = dptr + i * chunk_allocated_size;
        // 该 chunk 在连续 VA 中的起始地址
        CUCHECK(cuMemMap(chunk_va, chunk_allocated_size, 0, mem_handle, 0));
        // 把该 chunk 映射到对应偏移

        // All chunks get RW access for dispatch writes
        // 所有 chunk 都授予读写权限，供 dispatch 写远端区域
        CUmemAccessDesc access = {};
        // 访问权限描述（零初始化）
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        // 权限作用位置：设备
        access.location.id = device_id;
        // 指定设备编号
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        // 授予读写权限
        CUCHECK(cuMemSetAccess(chunk_va, chunk_allocated_size, &access, 1));
        // 使本设备获得该 chunk 的读写访问

        CUCHECK(cuMemRelease(mem_handle));
        // 导入句柄用完即释放（物理内存由映射引用）
    }

    std::vector<int64_t> full_shape = chunk_shape;
    // 拷贝单 chunk 形状以构造整块形状
    full_shape[0] *= world_size;
    // dim0 乘以 rank 数得到完整形状

    int ws = world_size;
    // 拷贝为 int，供 lambda 捕获
    size_t cas = chunk_allocated_size;
    // 拷贝单 chunk 分配大小，供 lambda 捕获
    size_t tas = total_allocated_size;
    // 拷贝总分配大小，供 lambda 捕获
    auto deleter = [device_id, dptr, cas, ws, tas](void *p) {
        // Storage 释放时调用的删除器
        if (!p) return;
        // 空指针直接返回
        c10::cuda::CUDAGuard guard(device_id);
        // 切换到映射时的设备上下文
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream().stream());
        // 同步当前流，确保没有 kernels 还在用这块内存
        for (int i = 0; i < ws; i++) {
            // 逐个解除每个 chunk 的映射
            cuMemUnmap(dptr + i * cas, cas);
            // 解除第 i 个 chunk 的 VA 映射
        }
        cuMemAddressFree(dptr, tas);
        // 释放整段保留的 VA 区间
    };

    auto storage = c10::Storage(
        // 构造 ATen Storage，持有整段 VA 与删除器
        c10::Storage::use_byte_size_t(),
        // 标记按字节指定大小
        static_cast<int64_t>(total_nbytes),
        // 有效总字节数
        at::InefficientStdFunctionContext::makeDataPtr(
            // 用 std::function 删除器构造 DataPtr
            reinterpret_cast<void *>(dptr), std::move(deleter),
            // VA 指针与删除器
            c10::Device(c10::kCUDA, device_id)),
            // 所属设备
        /*allocator=*/nullptr,
        // 无分配器（VMM 内存不走常规分配）
        /*resizable=*/false);
        // Storage 不可改变大小

    auto impl = c10::make_intrusive<at::TensorImpl>(
        // 构造 TensorImpl（张量实现层）
        std::move(storage),
        // 传入上面构造的 Storage
        c10::DispatchKeySet(c10::DispatchKey::CUDA),
        // 派发键：CUDA
        c10::scalarTypeToTypeMeta(dtype));
        // 元素类型元信息
    impl->set_sizes_contiguous(full_shape);
    // 设置为完整形状的连续布局
    return at::Tensor(std::move(impl));
    // 包装为张量返回
}

// ============================================================
// Multicast (NVSwitch SHARP) — register once, hold persistently
// ============================================================
// ============================================================
// 组播（NVSwitch SHARP）—— 一次性注册，长期持有
// ============================================================
//
// Adds a multicast mapping on top of an already-allocated NVLink chunk (each
// rank's own cuMemCreate physical memory): bind every rank's chunk physical
// memory to the same multicast object, then map a multicast VA. A single
// multimem.st write to the multicast VA is fanned out by NVSwitch hardware to
// the corresponding offset of every rank's local physical memory.
// 在已分配的 NVLink chunk（每个 rank 自己 cuMemCreate 的物理内存）之上叠加组播映射：
// 把每个 rank 的 chunk 物理内存绑定到同一个组播对象，再映射一个组播 VA。
// 对该 VA 的一次 multimem.st 写会被 NVSwitch 硬件扇出到每个 rank 本地物理内存的相同偏移。
//
// Uses no extra device memory — it only adds one more VA mapping over the
// existing chunk physical memory. Lifetime: the caller (MoonEP Buffer) holds
// the returned mc VA tensor, same lifetime as meta_buf; the mc VA tensor's
// deleter releases the multicast handle after unmap/free of the VA.
// 不占用额外显存——只是在既有 chunk 物理内存上多加一个 VA 映射。生命周期：
// 调用者（MoonEP Buffer）持有返回的 mc VA 张量，与 meta_buf 同生命周期；
// mc VA 张量的删除器在 unmap/free VA 之后释放组播句柄。
//
// Registration must follow the Python-side coordination order (separated by
// dist.barrier, otherwise bind fails with CUDA_ERROR_ILLEGAL_STATE or hangs):
//   root: nvl_multicast_create -> send the exported POSIX fd to other ranks
//   non-root: nvl_multicast_import
//   all ranks: nvl_multicast_add_device
//   --- dist.barrier ---
//   all ranks: nvl_multicast_bind_map
//   --- dist.barrier ---
// 注册必须遵循 Python 侧的协调顺序（用 dist.barrier 分隔，否则 bind 会报
// CUDA_ERROR_ILLEGAL_STATE 或挂死）：
//   root: nvl_multicast_create -> 把导出的 POSIX fd 发给其他 rank
//   非 root: nvl_multicast_import
//   所有 rank: nvl_multicast_add_device
//   --- dist.barrier ---
//   所有 rank: nvl_multicast_bind_map
//   --- dist.barrier ---

// Whether the device supports multicast.
// 查询当前设备是否支持组播。
static inline bool nvl_multicast_supported() {
    // 静态内联函数
    int device_id;
    // 用于接收当前 CUDA 设备编号
    CUDACHECK(cudaGetDevice(&device_id));
    // 查询当前设备编号
    CUdevice cu_device;
    // 驱动 API 的设备句柄
    CUCHECK(cuDeviceGet(&cu_device, device_id));
    // 由编号取得设备句柄
    int supported = 0;
    // 属性查询结果：是否支持组播
    CUCHECK(cuDeviceGetAttribute(&supported,
        // 查询设备属性 MULTICAST_SUPPORTED
        CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cu_device));
        // 指定属性名与设备
    return supported != 0;
    // 返回是否支持
}

// 查询给定设备数与句柄类型下组播对象的建议对齐粒度（字节）
static inline size_t nvl_multicast_granularity_for(
    // 静态内联函数
    int num_devices, CUmemAllocationHandleType ht
    // 参与设备数与句柄类型
) {
    CUmulticastObjectProp prop = {};
    // 构造组播对象属性（零初始化）
    prop.numDevices = static_cast<unsigned int>(num_devices);
    // 参与组播的设备数
    prop.size = 0;
    // 大小填 0（仅查询粒度）
    prop.handleTypes = ht;
    // 指定句柄类型
    prop.flags = 0;
    // 无附加标志
    size_t gran = 0;
    // 用于接收粒度查询结果
    CUCHECK(cuMulticastGetGranularity(
        // 查询组播的建议对齐粒度
        &gran, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
        // 出错则打印并退出
    return gran;
    // 返回粒度（字节）
}

// Recommended multicast alignment granularity (bytes). Both the addr and size
// of a bind must be aligned to it. Like nvl_granularity_max, this is the max
// over every handle type that may be used, so a buffer padded to it stays
// valid whichever type the process group picks.
// 组播的建议对齐粒度（字节）。bind 的地址与大小都必须按它对齐。与 nvl_granularity_max
// 类似，这里取所有可能用到的句柄类型中的最大值，这样按它填充的缓冲区无论进程组
// 选择哪种句柄类型都仍然有效。
static inline size_t nvl_multicast_granularity(int num_devices) {
    // 参与设备数
    size_t gran = nvl_multicast_granularity_for(
        // 先取 fd 类型的组播粒度
        num_devices, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
        // 指定 fd 句柄类型
    if (nvl_fabric_supported()) {
        // 若设备支持 fabric 句柄
        gran = std::max(gran, nvl_multicast_granularity_for(
            // 取两者中的较大者
            num_devices, CU_MEM_HANDLE_TYPE_FABRIC));
            // 再取 fabric 类型的组播粒度
    }
    return gran;
    // 返回最大粒度
}

// get_multicast_granularity 的 int64 包装（供 pybind 导出）
static inline int64_t get_multicast_granularity(int64_t num_devices) {
    // 以 int64 接收设备数
    return static_cast<int64_t>(
        // 把 size_t 结果转回 int64
        nvl_multicast_granularity(static_cast<int>(num_devices)));
        // 调用核心实现
}

// root only: create the multicast object and export it for the other ranks.
// Returns (mc_handle as uint64, shareable handle).
// mc_handle is a handle value valid within this process; Python holds it and
// passes it back unchanged to bind_map. The caller passes the shareable handle
// to the other ranks (closing the fd, if it is one, after all have imported).
// 仅 root：创建组播对象并导出给其他 rank。返回 (以 uint64 表示的 mc_handle, 可共享句柄)。
// mc_handle 是仅在本进程内有效的句柄值；Python 持有它并原样传回 bind_map。
// 调用者把可共享句柄发给其他 rank（若是 fd，等所有 rank 导入后关闭）。
inline std::tuple<int64_t, at::Tensor> nvl_multicast_create(
    // 创建组播对象的函数
    int64_t size_bytes, int64_t num_devices, bool use_fabric
    // 单 chunk 字节数、设备数与共享方式
) {
    TORCH_CHECK(nvl_multicast_supported(),
        // 校验设备支持组播
        "Multicast not supported on this device");
        // 不支持时抛错
    const size_t gran = nvl_multicast_granularity(static_cast<int>(num_devices));
    // 查询组播对齐粒度
    const size_t aligned =
        // 把大小向上取整到粒度整数倍
        (static_cast<size_t>(size_bytes) + gran - 1) / gran * gran;
        // 标准向上取整公式

    CUmulticastObjectProp prop = {};
    // 构造组播对象属性（零初始化）
    prop.numDevices = static_cast<unsigned int>(num_devices);
    // 参与组播的设备数
    prop.size = aligned;
    // 对齐后的对象大小
    prop.handleTypes = nvl_handle_type(use_fabric);
    // 根据共享方式设置句柄类型
    prop.flags = 0;
    // 无附加标志

    CUmemGenericAllocationHandle mc_handle;
    // 新建组播对象的持有句柄
    CUCHECK(cuMulticastCreate(&mc_handle, &prop));
    // 创建组播对象

    auto shareable = nvl_export_shareable(mc_handle, use_fabric);
    // 导出可共享句柄供其他 rank 导入
    return {static_cast<int64_t>(mc_handle), std::move(shareable)};
    // 返回句柄值与共享句柄
}

// non-root: import the multicast object from the handle sent by root. If it is
// an fd, the caller may close it once imported.
// 非 root：从 root 发来的句柄导入组播对象。若是 fd，导入后调用者即可关闭。
inline int64_t nvl_multicast_import(
    // 导入组播对象的函数
    const at::Tensor &shareable, bool use_fabric
    // root 的共享句柄张量与共享方式
) {
    auto one = use_fabric ? shareable.view({1, kFabricHandleBytes})
        // fabric：视图成 1x64 矩阵以复用校验逻辑
                          : shareable.view({1});
        // fd：视图成单元素向量
    nvl_check_handles(one, 1, use_fabric);
    // 按单 rank 校验句柄张量
    return static_cast<int64_t>(nvl_import_shareable(one, 0, use_fabric));
    // 导入第 0 行并转成 int64 返回
}

// all ranks: add this rank's device to the multicast group. Must happen before
// bind, and bind may only happen after all ranks have added (dist.barrier).
// 所有 rank：把本 rank 的设备加入组播组。必须在 bind 之前执行，且 bind 只有在
// 所有 rank 都完成加入之后才能进行（用 dist.barrier 分隔）。
inline void nvl_multicast_add_device(int64_t mc_handle_u64) {
    // 以 int64 传入的组播对象句柄
    int device_id;
    // 用于接收当前 CUDA 设备编号
    CUDACHECK(cudaGetDevice(&device_id));
    // 查询当前设备编号
    CUdevice cu_device;
    // 驱动 API 的设备句柄
    CUCHECK(cuDeviceGet(&cu_device, device_id));
    // 由编号取得设备句柄
    CUCHECK(cuMulticastAddDevice(
        // 把本设备加入组播对象
        static_cast<CUmemGenericAllocationHandle>(mc_handle_u64), cu_device));
        // 还原句柄类型并传入设备
}

// all ranks: bind this rank's local chunk owned physical memory (the mem
// handle kept by nvl_dist_alloc) to the multicast object, then reserve a VA
// for the multicast handle + map + setAccess. Returns an int32 keepalive
// tensor holding the multicast VA (.data_ptr() is the multimem address). Uses
// BindMem (owned handle) instead of BindAddr: BindAddr returns invalid
// argument for IPC-imported VAs on this platform.
// 所有 rank：把本 rank 本地 chunk 的持有物理内存（nvl_dist_alloc 保留的 mem 句柄）
// 绑定到组播对象，然后为组播句柄保留 VA + 映射 + 设置访问权限。返回持有组播 VA 的
// int32 保活张量（.data_ptr() 即 multimem 地址）。使用 BindMem（持有句柄）而非
// BindAddr：在本平台上 BindAddr 对 IPC 导入的 VA 会返回无效参数错误。
inline at::Tensor nvl_multicast_bind_map(
    // 绑定内存并映射组播 VA 的函数
    int64_t mc_handle_u64, int64_t owned_mem_handle_u64,
    // 组播对象句柄与本地持有 mem 句柄
    int64_t size_bytes, int64_t num_devices
    // 单 chunk 字节数与参与设备数
) {
    int device_id;
    // 用于接收当前 CUDA 设备编号
    CUDACHECK(cudaGetDevice(&device_id));
    // 查询当前设备编号
    CUmemGenericAllocationHandle mc_handle =
        // 还原组播对象句柄类型
        static_cast<CUmemGenericAllocationHandle>(mc_handle_u64);
        // 从 int64 还原
    CUmemGenericAllocationHandle mem_handle =
        // 还原本地持有 mem 句柄类型
        static_cast<CUmemGenericAllocationHandle>(owned_mem_handle_u64);
        // 从 int64 还原

    const size_t gran = nvl_multicast_granularity(static_cast<int>(num_devices));
    // 查询组播对齐粒度
    TORCH_CHECK(static_cast<size_t>(size_bytes) % gran == 0,
        // 校验绑定大小是粒度的整数倍
        "multicast bind size (", size_bytes, ") must be a multiple of multicast "
        // 错误信息第一行
        "granularity (", gran, "); pad meta_buf chunk accordingly");
        // 错误信息第二行（提示相应填充 meta_buf chunk）
    const size_t size = static_cast<size_t>(size_bytes);
    // 转成 size_t 供驱动 API 使用

    // Bind this rank's owned physical allocation to offset 0 of the multicast object.
    // 把本 rank 的持有物理分配绑定到组播对象的 0 号偏移。
    CUCHECK(cuMulticastBindMem(
        // 用 BindMem 绑定物理内存
        mc_handle, /*mcOffset=*/0, mem_handle, /*memOffset=*/0, size, 0));
        // 组播偏移 0、内存偏移 0、长度 size、无标志

    // Reserve a separate VA for the multicast handle and map it. Pass 0 for
    // alignment (the driver aligns to granularity by default, same as
    // nvl_dist_alloc/nvl_dist_map; a non-zero alignment is rejected by
    // cuMemAddressReserve as invalid argument).
    // 为组播句柄保留一段独立 VA 并映射。对齐参数传 0（驱动默认按粒度对齐，与
    // nvl_dist_alloc/nvl_dist_map 一致；非零对齐会被 cuMemAddressReserve
    // 以无效参数拒绝）。
    CUdeviceptr mc_va;
    // 组播 VA
    CUCHECK(cuMemAddressReserve(&mc_va, size, 0, 0, 0));
    // 保留 VA 区间
    CUCHECK(cuMemMap(mc_va, size, 0, mc_handle, 0));
    // 把组播对象映射到该 VA

    CUmemAccessDesc access = {};
    // 访问权限描述（零初始化）
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // 权限作用位置：设备
    access.location.id = device_id;
    // 指定设备编号
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    // 授予读写权限
    CUCHECK(cuMemSetAccess(mc_va, size, &access, 1));
    // 使本设备获得组播 VA 的读写访问

    // keepalive: the deleter unmaps + frees the VA, then releases mc_handle.
    // data_ptr() = multimem address.
    // 保活：删除器先 unmap + 释放 VA，然后释放 mc_handle。data_ptr() 即 multimem 地址。
    auto deleter = [device_id, mc_va, size, mc_handle](void* p) {
        // Storage 释放时调用的删除器
        if (!p) return;
        // 空指针直接返回
        c10::cuda::CUDAGuard guard(device_id);
        // 切换到映射时的设备上下文
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream().stream());
        // 同步当前流，确保没有 kernels 还在用这块内存
        cuMemUnmap(mc_va, size);
        // 解除组播 VA 的映射
        cuMemAddressFree(mc_va, size);
        // 释放保留的 VA 区间
        cuMemRelease(mc_handle);
        // 释放组播对象句柄
    };

    std::vector<int64_t> shape = {static_cast<int64_t>(size / sizeof(int32_t))};
    // 按 int32 元素个数构造一维形状
    auto storage = c10::Storage(
        // 构造 ATen Storage，持有组播 VA 与删除器
        c10::Storage::use_byte_size_t(),
        // 标记按字节指定大小
        static_cast<int64_t>(size),
        // 总字节数
        at::InefficientStdFunctionContext::makeDataPtr(
            // 用 std::function 删除器构造 DataPtr
            reinterpret_cast<void*>(mc_va), std::move(deleter),
            // 组播 VA 指针与删除器
            c10::Device(c10::kCUDA, device_id)),
            // 所属设备
        /*allocator=*/nullptr,
        // 无分配器（VMM 内存不走常规分配）
        /*resizable=*/false);
        // Storage 不可改变大小

    auto impl = c10::make_intrusive<at::TensorImpl>(
        // 构造 TensorImpl（张量实现层）
        std::move(storage),
        // 传入上面构造的 Storage
        c10::DispatchKeySet(c10::DispatchKey::CUDA),
        // 派发键：CUDA
        c10::scalarTypeToTypeMeta(at::kInt));
        // 元素类型：int32
    impl->set_sizes_contiguous(shape);
    // 设置为一维连续布局
    return at::Tensor(std::move(impl));
    // 包装为张量返回
}
