import os  # 导入 os 模块，用于环境变量读取和文件描述符关闭等系统操作
import shutil  # 导入 shutil 模块，用于删除临时目录
import socket  # 导入 socket 模块，用于通过 UNIX 域套接字传递 IPC 文件描述符
import struct  # 导入 struct 模块，用于打包/解包整数（fd、rank 编号）
import tempfile  # 导入 tempfile 模块，用于创建 IPC 套接字文件所需的临时目录

import torch  # 导入 PyTorch 主库，用于构造张量
import torch.distributed as dist  # 导入 torch 分布式模块，用于集合通信（广播、栅栏等）

# 从编译好的 CUDA 扩展 _C 中导入 NVLink/VMM 共享缓冲区相关的底层函数
from moonep._C import (  # 从 _C 扩展模块导入以下符号
    FABRIC_HANDLE_BYTES as _FABRIC_HANDLE_BYTES,  # fabric 句柄的字节长度（64），重命名为私有别名
    nvl_dist_alloc,  # 在本进程分配一块 VMM 显存并导出可共享句柄
    nvl_dist_map,  # 将所有 rank 的显存块映射为一段连续虚拟地址
    nvl_fabric_supported,  # 查询当前设备是否支持 fabric 句柄
    nvl_release_mem_handle,  # 释放 nvl_dist_alloc 返回的持有句柄
    get_vmm_granularity,  # 获取 VMM 分配粒度（字节）
    get_multicast_granularity,  # 获取组播对象的对齐粒度（字节）
    nvl_multicast_supported,  # 查询当前设备是否支持组播（NVSwitch SHARP）
    nvl_multicast_create,  # root rank 创建组播对象并导出共享句柄
    nvl_multicast_import,  # 非 root rank 从 root 导入组播对象
    nvl_multicast_add_device,  # 将本设备的 GPU 加入组播对象
    nvl_multicast_bind_map,  # 将本地显存绑定为组播内存并映射组播虚拟地址
)

# How VMM allocations are shared between the ranks of an EP group. "auto" (the
# default) picks fabric handles when the group spans more than one node and the
# device supports them, and POSIX fds otherwise; "fabric" / "fd" force one.
# 环境变量：控制 EP 组内各 rank 之间如何共享 VMM 分配。默认 "auto" 会在跨节点且设备
# 支持时选用 fabric 句柄，否则用 POSIX fd；"fabric"/"fd" 可强制指定其中一种。
_HANDLE_TYPE_ENV = "MOONEP_MEM_HANDLE_TYPE"  # 存放环境变量名字符串


def _use_fabric_for_group(group: dist.ProcessGroup | None) -> bool:
    # 判断该进程组是否应使用 fabric 句柄：读取环境变量并（必要时）all-gather 各 rank 的支持情况
    mode = os.environ.get(_HANDLE_TYPE_ENV, "auto").lower()  # 读取环境变量配置的句柄模式，默认 auto，统一转小写
    assert mode in ("auto", "fabric", "fd"), (  # 校验模式取值合法，否则断言失败并报错
        f"{_HANDLE_TYPE_ENV} must be one of auto/fabric/fd, got {mode!r}"  # 断言失败时的错误信息
    )
    if mode == "fd":  # 如果显式指定使用 POSIX fd 模式
        return False  # 直接返回不使用 fabric

    supported = bool(nvl_fabric_supported())  # 查询本设备是否支持 fabric 句柄
    world_size = (dist.get_world_size(group=group)  # 获取该进程组的世界大小（组未初始化时为 1）
                  if dist.is_available() and dist.is_initialized() else 1)  # 分布式未初始化时退化为单进程
    if world_size == 1:  # 单进程场景无需集合通信
        unsupported = [] if supported else [0]  # 本机支持则列表为空，否则记 rank 0 不支持
    else:  # 多 rank 场景需要收集每个 rank 的支持情况
        local = torch.tensor([supported], dtype=torch.uint8, device="cuda")  # 本 rank 的支持标志（0/1）张量
        gathered = torch.empty(world_size, dtype=torch.uint8, device="cuda")  # 预分配收集结果的输出张量
        dist.all_gather_into_tensor(gathered, local, group=group)  # 将所有 rank 的支持标志收集到 gathered
        unsupported = (gathered == 0).nonzero().flatten().tolist()  # 找出所有不支持 fabric 的 rank 编号列表

    if mode == "fabric":  # 如果显式强制使用 fabric 模式
        assert not unsupported, (  # 断言所有 rank 都支持，否则直接报错
            f"{_HANDLE_TYPE_ENV}=fabric, but fabric memory handles are "  # 错误信息第一行
            f"unsupported on group ranks {unsupported}."  # 错误信息第二行，列出不支持 fabric 的 rank
        )
        return True  # 强制 fabric 模式，返回 True

    return not unsupported  # auto 模式：仅当所有 rank 都支持 fabric 时才使用 fabric


def _all_gather_shareables(
    local_handle: torch.Tensor,  # 本 rank 的 fabric 句柄张量
    group: dist.ProcessGroup | None,  # 目标进程组（None 表示默认组）
) -> torch.Tensor:
    """Gather every rank's fabric handle into a uint8[world_size, 64] CPU tensor."""  # 文档字符串：收集所有 rank 的 fabric 句柄为 uint8[world_size, 64] 的 CPU 张量
    # 收集组内每个 rank 的 fabric 句柄
    world_size = dist.get_world_size(group=group)  # 获取进程组的世界大小
    gathered = torch.empty(world_size, _FABRIC_HANDLE_BYTES,  # 预分配 [world_size, 64] 的输出张量
                           dtype=torch.uint8, device="cuda")  # 类型为 uint8，放在 CUDA 上供集合通信使用
    dist.all_gather_into_tensor(  # 执行 all-gather，把每个 rank 的句柄按行拼到 gathered
        gathered, local_handle.cuda().view(1, _FABRIC_HANDLE_BYTES), group=group)  # 输入为本 rank 句柄 reshaped 成 1x64
    return gathered.cpu()  # 拷回 CPU 并返回收集结果


def _broadcast_shareable(
    local_handle: torch.Tensor | None,  # 源 rank 的句柄张量（非源 rank 传 None）
    owner_rank: int,  # 持有句柄的源 rank 编号（组内 rank）
    group: dist.ProcessGroup | None,  # 目标进程组（None 表示默认组）
) -> torch.Tensor:
    """Broadcast one rank's fabric handle to the whole group."""  # 文档字符串：把某一个 rank 的 fabric 句柄广播给整个组
    # 把一个 rank 的句柄广播给全组
    if local_handle is not None:  # 如果本进程就是持有者
        buf = local_handle.cuda()  # 直接把已有句柄张量搬到 CUDA 上作为广播源
    else:  # 否则（非持有者）
        buf = torch.empty(_FABRIC_HANDLE_BYTES, dtype=torch.uint8, device="cuda")  # 分配接收缓冲区
    # 有组时把组内 rank 换算成全局 rank；无组时组内 rank 就是全局 rank
    src = dist.get_global_rank(group, owner_rank) if group is not None \
        else owner_rank  # 无组时组内 rank 就是全局 rank
    dist.broadcast(buf, src=src, group=group)  # 从源 rank 广播句柄到全组
    return buf.cpu()  # 拷回 CPU 并返回


def pad_to_granularity(nbytes: int) -> int:
    """Round up nbytes to VMM granularity."""  # 文档字符串：将字节数向上取整到 VMM 分配粒度
    # 把字节数向上对齐到 VMM 分配粒度
    gran = get_vmm_granularity()  # 查询当前设备的 VMM 分配粒度
    return ((nbytes + gran - 1) // gran) * gran  # 向上取整到粒度的整数倍并返回


def pad_dim0_for_alignment(chunk_shape: list[int], dtype: torch.dtype) -> int:
    """Compute the padded dim0 so that chunk bytes are aligned to VMM granularity.

    Returns the padded dim0 value (>= chunk_shape[0]).
    """  # 文档字符串：计算填充后的 dim0，使整个 chunk 的字节数对齐到 VMM 粒度
    # 计算填充后的 dim0，使 chunk 总字节数对齐到 VMM 粒度
    inner_size = dtype.itemsize  # 内层单个元素的字节大小（dtype 的大小）
    for d in chunk_shape[1:]:  # 遍历除 dim0 以外的所有维度
        inner_size *= d  # 累乘得到每行的字节数

    nbytes = chunk_shape[0] * inner_size  # 计算整个 chunk 未填充时的总字节数
    padded_bytes = pad_to_granularity(nbytes)  # 将总字节数向上对齐到 VMM 粒度
    padded_dim0 = padded_bytes // inner_size  # 反推出填充后的 dim0（行数）
    # Ensure exact alignment
    # 确保严格对齐
    while padded_dim0 * inner_size % get_vmm_granularity() != 0:  # 若仍未对齐（整除反推可能不精确）则继续加行
        padded_dim0 += 1  # 每次增加一行直到字节数对齐粒度
    return padded_dim0  # 返回填充后的 dim0（保证 >= 原 chunk_shape[0]）


def _exchange_ipc_fds(
    local_fd: int | None,  # 本进程要发送的导出 fd（发送者必传，其他 rank 传 None）
    sender_ranks: list[int],  # 会发送 fd 的 rank 编号列表
    local_rank: int,  # 本进程在组内的 rank 编号
    world_size: int,  # 组内 rank 总数
    group: dist.ProcessGroup | None,  # 目标进程组（None 表示默认组）
) -> dict[int, int]:
    """Pass POSIX fds between ranks via per-rank unix datagram sockets.

    Ranks listed in ``sender_ranks`` must pass their exported fd as
    ``local_fd`` (other ranks pass None). Every rank receives one fd from
    each sender; the kernel dups the fd into the receiving process
    (SCM_RIGHTS), so the returned fds are owned by this process and must be
    closed by the caller after import. The sender's own fd may be closed as
    soon as this function returns — a trailing barrier guarantees all peers
    have already received their copy.
    """  # 文档字符串：通过每 rank 一个 UNIX 数据报套接字在 rank 间传递 POSIX fd（SCM_RIGHTS）；返回的 fd 归本进程所有，导入后需由调用者关闭
    # 通过每个 rank 一个的 UNIX 数据报套接字在 rank 之间传递 POSIX fd
    # Group-rank 0 creates a shared dir; broadcast the path to the group.
    # 组内 rank 0 创建共享目录，然后把路径广播给整个组。
    if local_rank == 0:  # 如果本进程是组内 rank 0
        dir_path = tempfile.mkdtemp(prefix="moonep_ipc_")  # 创建临时目录用于存放各 rank 的套接字文件
    else:  # 其他 rank
        dir_path = None  # 先置为空，等待广播
    obj = [dir_path]  # 用单元素列表包装以便 broadcast_object_list 广播
    src = dist.get_global_rank(group, 0) if group is not None else 0  # 计算组内 rank 0 对应的全局 rank
    dist.broadcast_object_list(obj, src=src, group=group)  # 把目录路径（对象列表）从 rank 0 广播到全组
    dir_path = obj[0]  # 取出广播后的目录路径

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)  # 创建 UNIX 域数据报套接字
    sock.bind(os.path.join(dir_path, f"rank_{local_rank}"))  # 把套接字绑定到共享目录下以本 rank 命名的地址文件
    sock.settimeout(120)  # 设置接收超时 120 秒，防止对端异常时永久阻塞
    # All sockets must be bound before any send.
    # 所有套接字必须先全部绑定完成，才允许开始发送。
    dist.barrier(group=group)  # 组内栅栏同步，确保所有 rank 的套接字都已绑定

    try:  # 进入 try/finally，保证套接字和临时目录一定被清理
        if local_fd is not None:  # 如果本进程是发送者
            payload = struct.pack("<i", local_rank)  # 消息体：小端 int32 打包的本 rank 编号（供接收者识别来源）
            anc = [(socket.SOL_SOCKET, socket.SCM_RIGHTS,  # 构造辅助数据：SOL_SOCKET 层的 SCM_RIGHTS 控制消息
                    struct.pack("<i", local_fd))]  # 控制消息内容为小端 int32 打包的本地 fd，内核会把它复制进接收进程
            for dst in range(world_size):  # 遍历组内所有目标 rank
                sock.sendmsg([payload], anc, 0,  # 发送消息体 + fd 辅助数据
                             os.path.join(dir_path, f"rank_{dst}"))  # 目标地址为共享目录下对应 rank 的套接字文件

        fds = {}  # 收到的 fd 字典：键为发送者 rank，值为接收到的 fd
        try:  # 内层 try：接收失败时清理已收到的 fd
            while len(fds) < len(sender_ranks):  # 循环直到收齐每个发送者的 fd
                msg, ancdata, _flags, _addr = sock.recvmsg(16, socket.CMSG_SPACE(4))  # 接收消息体和 SCM_RIGHTS 辅助数据
                src_rank = struct.unpack("<i", msg[:4])[0]  # 从消息体解包出发送者的 rank 编号
                for level, ctype, cdata in ancdata:  # 遍历收到的辅助数据项
                    if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:  # 找到携带 fd 的控制消息
                        fds[src_rank] = struct.unpack("<i", cdata[:4])[0]  # 解包出 fd 并记录到字典
                        break  # 找到后跳出辅助数据遍历
                else:  # for 循环未 break，说明本条消息没带 fd
                    raise RuntimeError("received IPC message without an fd")  # 视为协议错误，直接抛异常
        except Exception:  # 接收过程中发生任何异常
            # SCM_RIGHTS fds are plain ints owned by this process; sock.close()
            # below does not close them, so drop any already received.
            # SCM_RIGHTS 收到的 fd 是本进程持有的普通整数，关闭套接字并不会关闭它们，
            # 因此异常退出时要把已收到的 fd 全部关闭，避免泄漏。
            for fd in fds.values():  # 遍历已收到的所有 fd
                os.close(fd)  # 逐个关闭，防止 fd 泄漏
            raise  # 重新抛出原异常
    finally:  # 无论成功与否都执行清理
        sock.close()  # 关闭本 rank 的套接字
        # Everyone has received their fds; safe to tear down the sockets and
        # for senders to close their original fd.
        # 走到这里说明所有人都已收到 fd；此时可以安全拆除套接字，
        # 发送者也可以关闭自己的原始 fd。
        dist.barrier(group=group)  # 尾部栅栏：保证所有 peer 都已收到副本后才继续
        if local_rank == 0:  # 由组内 rank 0 负责清理
            shutil.rmtree(dir_path, ignore_errors=True)  # 删除共享临时目录及其中的套接字文件
    return fds  # 返回收集到的 {发送者rank: fd} 字典


def _map_nvl_dist_tensor(
    chunk_shape: list[int],  # 单个 rank 显存块的形状（dim0 已按粒度对齐）
    dtype: torch.dtype,  # 张量元素类型
    shareable: torch.Tensor,  # 本 rank 导出的可共享句柄（fd 标量张量或 64 字节 fabric 张量）
    keepalive: torch.Tensor,  # 挂到结果张量上防止物理内存释放的保活张量
    local_rank: int,  # 本进程在组内的 rank 编号
    world_size: int,  # 组内 rank 总数
    group: dist.ProcessGroup | None,  # 目标进程组（None 表示默认组）
    use_fabric: bool,  # 是否使用 fabric 句柄方式共享
) -> torch.Tensor:
    # 把所有 rank 的显存块映射成一段连续虚拟地址，返回完整张量
    if use_fabric:  # fabric 句柄方式：直接 all-gather 各 rank 的句柄
        shareables = _all_gather_shareables(shareable, group)  # 收集全组的 fabric 句柄到 CPU 张量
        full_tensor = nvl_dist_map(  # 调用 C++ 扩展把所有 rank 的块映射为连续 VA
            chunk_shape=chunk_shape,  # 传入单块形状
            dtype=dtype,  # 传入元素类型
            shareables=shareables,  # 传入收集到的句柄矩阵
            local_rank=local_rank,  # 本 rank 编号
            world_size=world_size,  # rank 总数
            use_fabric=True,  # 标记使用 fabric 句柄
        )
    else:  # POSIX fd 方式：需要通过 SCM_RIGHTS 交换 fd
        local_fd = int(shareable.item())  # 从标量张量取出本进程导出的 fd
        try:  # try/finally：无论映射成功与否都要关闭本地 fd
            fds = _exchange_ipc_fds(local_fd, list(range(world_size)),  # 通过 UNIX 套接字交换 fd（每个 rank 都是发送者）
                                    local_rank, world_size, group)  # 传入组内信息
        finally:  # 交换完成后
            os.close(local_fd)  # 关闭自己导出的原始 fd（各进程已拿到副本）
        all_fds = [fds[r] for r in range(world_size)]  # 按 rank 顺序整理收到的 fd 列表
        try:  # try/finally：映射后关闭所有导入的 fd
            full_tensor = nvl_dist_map(  # 调用 C++ 扩展把所有 rank 的块映射为连续 VA
                chunk_shape=chunk_shape,  # 传入单块形状
                dtype=dtype,  # 传入元素类型
                shareables=torch.tensor(all_fds, dtype=torch.int64),  # 把 fd 列表包装成 int64 张量
                local_rank=local_rank,  # 本 rank 编号
                world_size=world_size,  # rank 总数
                use_fabric=False,  # 标记使用 fd 方式
            )
        finally:  # 映射完成后
            for fd in all_fds:  # 遍历所有导入的 fd
                os.close(fd)  # 逐个关闭（已导入成句柄，不再需要 fd）
    full_tensor._keepalive = keepalive  # 把保活张量挂在结果上，防止物理内存被提前释放
    return full_tensor  # 返回映射好的完整张量


def create_nvl_dist_tensor(
    chunk_shape: list[int],  # 单个 rank 显存块的形状（必须已对齐 VMM 粒度）
    dtype: torch.dtype,  # 张量元素类型
    local_rank: int,  # 本进程在组内的 rank 编号
    world_size: int,  # 组内 rank 总数
    group: dist.ProcessGroup | None = None,  # 目标进程组（默认 None 表示默认组）
) -> torch.Tensor:
    """Allocate an NVLink distributed tensor with all-RW access.

    chunk_shape MUST already be padded to VMM granularity alignment.
    Use pad_dim0_for_alignment() to compute the padded dim0.

    `local_rank` and `world_size` must match the given `group` (or the default
    group when `group is None`). All ranks in the group exchange memory handles.
    """  # 文档字符串：分配一个全 rank 可读写的 NVLink 分布式张量；chunk_shape 必须已按 VMM 粒度对齐
    # 分配一个所有 rank 均可读写的 NVLink 分布式张量
    use_fabric = _use_fabric_for_group(group)  # 决定本组使用 fabric 还是 fd 方式共享
    keepalive, shareable, owned_handle = nvl_dist_alloc(  # 分配本地显存块并导出可共享句柄
        shape=chunk_shape, dtype=dtype, use_fabric=use_fabric)  # 传入形状、类型与共享方式
    try:  # try/finally：映射完成后释放持有的分配句柄
        return _map_nvl_dist_tensor(  # 交换句柄并把全组显存映射为连续 VA
            chunk_shape, dtype, shareable, keepalive,  # 传入形状、类型、导出句柄与保活张量
            local_rank, world_size, group, use_fabric,  # 传入组内信息与共享方式
        )
    finally:  # 无论成功与否
        nvl_release_mem_handle(owned_handle)  # 释放本 rank 持有的底层分配句柄（物理内存由映射保活）


def create_nvl_dist_multicast_tensor(
    chunk_shape: list[int],  # 单个 rank 显存块的形状（必须已对齐 VMM 粒度）
    dtype: torch.dtype,  # 张量元素类型
    local_rank: int,  # 本进程在组内的 rank 编号
    world_size: int,  # 组内 rank 总数
    group: dist.ProcessGroup | None = None,  # 目标进程组（默认 None 表示默认组）
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate an NVLink distributed tensor and its multicast view together.

    Returns `(full_tensor, mc_view)`. `full_tensor` is the regular all-rank RW
    VMM mapping; `mc_view` is a single-chunk multicast VA whose `.data_ptr()` can
    be used by `multimem.st` to fan out writes to every rank's local chunk.

    The owned allocation handle needed by `cuMulticastBindMem` stays internal to
    this helper and is released after both mappings have been created.
    """  # 文档字符串：同时分配 NVLink 分布式张量及其组播视图，返回 (full_tensor, mc_view)
    # 同时创建 NVLink 分布式张量和它的组播视图
    use_fabric = _use_fabric_for_group(group)  # 决定本组使用 fabric 还是 fd 方式共享
    keepalive, shareable, owned_handle = nvl_dist_alloc(  # 分配本地显存块并导出可共享句柄（持有句柄后面绑组播要用）
        shape=chunk_shape, dtype=dtype, use_fabric=use_fabric)  # 传入形状、类型与共享方式
    try:  # try/finally：两种映射都建好后释放持有句柄
        full_tensor = _map_nvl_dist_tensor(  # 先建立全组单播连续 VA 映射
            chunk_shape, dtype, shareable, keepalive,  # 传入形状、类型、导出句柄与保活张量
            local_rank, world_size, group, use_fabric,  # 传入组内信息与共享方式
        )
        mc_view = _create_nvl_multicast_view(  # 再在已有 chunk 上叠加组播 VA 映射
            full_tensor, owned_handle, local_rank, world_size, group,  # 传入单播张量、持有句柄与组内信息
            use_fabric,  # 传入共享方式
        )
        return full_tensor, mc_view  # 返回单播张量与组播视图的二元组
    finally:  # 两种映射创建完毕后
        nvl_release_mem_handle(owned_handle)  # 释放底层分配句柄（物理内存由各映射保活）


def _create_nvl_multicast_view(
    meta_buf: torch.Tensor,  # 已建好的全组单播 NVLink 张量（每 rank 一个 chunk）
    owned_handle: int,  # 本 rank 由 nvl_dist_alloc 返回的持有分配句柄（绑组播必需）
    local_rank: int,  # 本进程在组内的 rank 编号
    world_size: int,  # 组内 rank 总数
    group: dist.ProcessGroup | None = None,  # 目标进程组（默认 None 表示默认组）
    use_fabric: bool = False,  # 是否使用 fabric 句柄方式共享
) -> torch.Tensor:
    """Overlay a multicast (NVSwitch SHARP) mapping on an existing NVL chunk.

    Binds each rank's own chunk physical memory (the chunk it `cuMemCreate`d
    inside `meta_buf`) to a single multicast object, then maps a multicast VA.
    A `multimem.st` to the returned VA fan-outs the hardware-replicated write to
    every rank's local chunk at the same offset.

    Registration is one-shot and persistent — the caller keeps the returned
    tensor alive (same lifetime as `meta_buf`). No extra device memory is used
    beyond an additional virtual address mapping over the existing chunk.

    The returned tensor's `.data_ptr()` is the multimem address, laid out as a
    single chunk (write `mc[off]` hits every rank's chunk `[off]`).
    """  # 文档字符串：在已有 NVL chunk 上叠加组播（NVSwitch SHARP）映射；写入返回 VA 会由硬件扇出到每个 rank 的本地 chunk
    # 在已有的 NVL chunk 上叠加组播视图
    assert nvl_multicast_supported(), "Multicast not supported on this device"  # 断言当前设备支持组播
    chunk_elems = meta_buf.numel() // world_size  # 计算单个 rank chunk 的元素数
    size_bytes = chunk_elems * meta_buf.element_size()  # 计算单个 chunk 的字节数
    is_root = local_rank == 0  # 判断本进程是否为组内 root（rank 0）

    # Root creates the multicast object and shares it with all ranks.
    # 由 root 创建组播对象，并把它的句柄共享给所有 rank。
    if is_root:  # 如果本进程是 root
        mc_handle, mc_shareable = nvl_multicast_create(  # 创建组播对象并导出可共享句柄
            size_bytes, world_size, use_fabric=use_fabric)  # 传入单 chunk 字节数、设备数与共享方式
    else:  # 非 root
        mc_handle, mc_shareable = 0, None  # 占位值，等待从 root 导入

    try:  # try/except：交换/导入失败时释放 root 持有的组播句柄
        if use_fabric:  # fabric 方式：root 广播 64 字节句柄
            root_handle = _broadcast_shareable(mc_shareable, 0, group)  # 把 root 的 fabric 句柄广播到全组
            if not is_root:  # 非 root 进程
                mc_handle = nvl_multicast_import(root_handle, use_fabric=True)  # 从广播来的句柄导入组播对象
        else:  # fd 方式：root 通过 SCM_RIGHTS 发送 fd
            local_fd = int(mc_shareable.item()) if is_root else None  # root 取出导出的 fd，非 root 传 None
            try:  # try/finally：交换后 root 关闭自己的 fd
                fds = _exchange_ipc_fds(local_fd, [0], local_rank, world_size,  # 只有 root（组内 rank 0）发送 fd
                                        group)  # 传入组内信息
            finally:  # 交换完成后
                if is_root:  # 仅 root 需要关闭
                    os.close(local_fd)  # 关闭自己导出的原始 fd
            root_fd = fds[0]  # 取出从 root 收到的 fd
            try:  # try/finally：导入后关闭 fd
                if not is_root:  # 非 root 进程
                    mc_handle = nvl_multicast_import(  # 从 root 的 fd 导入组播对象
                        torch.tensor(root_fd, dtype=torch.int64),  # 把 fd 包装成 int64 张量
                        use_fabric=False)  # 标记 fd 方式
            finally:  # 导入完成后
                os.close(root_fd)  # 关闭导入用的 fd
    except Exception:  # 交换或导入过程中发生异常
        # The multicast object handle is a CUmemGenericAllocationHandle; if the
        # exchange or import fails there is no owner left to release it.
        # 组播对象句柄是 CUmemGenericAllocationHandle；若交换或导入失败，
        # 就没有其他持有者能释放它，因此 root 需在此释放。
        if is_root:  # 仅 root 持有原始句柄
            nvl_release_mem_handle(mc_handle)  # 释放组播对象句柄，避免泄漏
        raise  # 重新抛出原异常

    # All ranks add their device before any bind, then barrier.
    # 所有 rank 先把设备加入组播组，之后才能 bind，这里用栅栏分隔。
    nvl_multicast_add_device(mc_handle)  # 把本 rank 的 GPU 加入组播对象
    dist.barrier(group=group)  # 栅栏：确保所有 rank 都已 add_device 后才允许 bind

    mc_view = nvl_multicast_bind_map(  # 把本 rank 的本地 chunk 物理内存绑到组播对象并映射组播 VA
        mc_handle, owned_handle, size_bytes, world_size)  # 传入组播句柄、持有分配句柄、字节数与设备数
    dist.barrier(group=group)  # 栅栏：确保所有 rank 都完成 bind/map 后才返回
    return mc_view  # 返回组播视图张量（data_ptr 即 multimem 地址）


def create_nvl_single_owner_tensor(
    shape: list[int],  # 张量形状（必须已按 VMM 粒度对齐）
    dtype: torch.dtype,  # 张量元素类型
    owner_rank: int,  # 物理内存所在的 owner rank 编号
    local_rank: int,  # 本进程在组内的 rank 编号
    group: dist.ProcessGroup | None = None,  # 目标进程组（默认 None 表示默认组）
) -> torch.Tensor:
    """Allocate a VMM tensor on one GPU, visible to all ranks via NVLink.

    The physical memory resides on owner_rank's GPU.  All ranks get an RW
    mapping so the resulting tensor can be read/written from any rank (remote
    accesses go over NVLink).  shape must already be padded to VMM granularity
    (use pad_dim0_for_alignment).
    """  # 文档字符串：在单个 GPU 上分配 VMM 张量，通过 NVLink 让所有 rank 均可读写
    # 创建一块由单一 owner 持有、全组经 NVLink 可见的 VMM 张量
    world_size = dist.get_world_size(group=group)  # 获取进程组的世界大小
    use_fabric = _use_fabric_for_group(group)  # 决定本组使用 fabric 还是 fd 方式共享
    is_owner = local_rank == owner_rank  # 判断本进程是否为 owner

    if is_owner:  # 只有 owner 分配物理内存并导出句柄
        keepalive, shareable, owned_handle = nvl_dist_alloc(  # 在 owner 的 GPU 上分配并导出可共享句柄
            shape=shape, dtype=dtype, use_fabric=use_fabric)  # 传入形状、类型与共享方式
        nvl_release_mem_handle(owned_handle)  # 立即释放持有句柄（不需要组播绑定，物理内存由映射保活）
    else:  # 非 owner
        shareable = None  # 没有本地句柄，等待广播/接收

    if use_fabric:  # fabric 方式：owner 直接广播句柄
        owner_handle = _broadcast_shareable(shareable, owner_rank, group)  # 把 owner 的 fabric 句柄广播到全组
        tensor = nvl_dist_map(  # 把 owner 的块映射进本进程的 VA 区
            chunk_shape=shape, dtype=dtype,  # 传入形状与类型
            shareables=owner_handle.view(1, _FABRIC_HANDLE_BYTES),  # 单个句柄 reshaped 成 1x64
            local_rank=0, world_size=1, use_fabric=True,  # 只映射一个块，按单块处理
        )
    else:  # fd 方式：owner 通过 SCM_RIGHTS 发送 fd
        local_fd = int(shareable.item()) if is_owner else None  # owner 取出导出的 fd，非 owner 传 None
        try:  # try/finally：交换后 owner 关闭自己的 fd
            fds = _exchange_ipc_fds(local_fd, [owner_rank], local_rank,  # 只有 owner 发送 fd
                                    world_size, group)  # 传入组内信息
        finally:  # 交换完成后
            if is_owner:  # 仅 owner 需要关闭
                os.close(local_fd)  # 关闭自己导出的原始 fd
        owner_fd = fds[owner_rank]  # 取出从 owner 收到的 fd
        try:  # try/finally：映射后关闭 fd
            tensor = nvl_dist_map(  # 把 owner 的块映射进本进程的 VA 区
                chunk_shape=shape, dtype=dtype,  # 传入形状与类型
                shareables=torch.tensor([owner_fd], dtype=torch.int64),  # 把 fd 包装成单元素 int64 张量
                local_rank=0, world_size=1, use_fabric=False,  # 只映射一个块，按单块处理
            )
        finally:  # 映射完成后
            os.close(owner_fd)  # 关闭导入用的 fd

    if is_owner:  # 只有 owner 有保活张量
        tensor._keepalive = keepalive  # 挂到结果上防止物理内存被提前释放
    return tensor  # 返回可在任意 rank 读写的结果张量
