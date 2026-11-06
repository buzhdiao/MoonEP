"""Generate the README figures explaining MoonEP's weight/grad buffer layout.

Outputs:
  figure/weight_buffer.png  - compact per-rank weight and prefetch views
  figure/grad_buffer.png    - compact per-rank grad and reduce views
"""
# 中文说明：本脚本用 matplotlib 绘制 README 中解释 MoonEP 权重/梯度缓冲区布局的示意图。
# 输出两个文件：
#   figure/weight_buffer.png —— 紧凑排布的每 rank 权重与前取（prefetch）视图；
#   figure/grad_buffer.png   —— 紧凑排布的每 rank 梯度与归约（reduce）视图。

# 导入 matplotlib 的 pyplot 接口，用于创建画布/坐标轴并保存图片
import matplotlib.pyplot as plt
# FancyArrowPatch：可定制样式的箭头；Rectangle：矩形色块（缓冲条/填充块）
from matplotlib.patches import FancyArrowPatch, Rectangle


# —— 以下为配色与几何常量（数据坐标单位）——
# YELLOW：本地专家权重 / 本地梯度条的颜色
YELLOW = "#FBE7A1"   # local experts / local grad
# PINK：prefetch 槽位 / reduce 缓冲条的颜色
PINK = "#F3D6DA"     # prefetch slots / reduce buffer
# RED：跨层共享的物理显存池条的颜色
RED = "#C05046"      # shared physical pool
# EDGE：矩形描边色
EDGE = "#333333"
# ARROW：箭头与总线线条颜色
ARROW = "#555555"
# TEXT：文字颜色
TEXT = "#222222"

# R：图中画出的 EP rank 数（示意 4 个 rank）
R = 4
# LOCAL_W：本地载荷条（专家权重/梯度）的宽度
LOCAL_W = 1.9
# BUFFER_W：通信缓冲条（prefetch/reduce）的宽度
BUFFER_W = 1.9
# BAR_H：每个条形的高度
BAR_H = 0.55
# PAD_W：VMM 填充（padding）块的默认宽度
PAD_W = 0.38
# PAD：填充块的颜色
PAD = "#E8EAED"


def arrow(ax, x1, y1, x2, y2, rad=0.0, style="-|>", lw=1.1, ls="-"):
    # 画一条从 (x1,y1) 到 (x2,y2) 的箭头；rad 控制 arc3 曲率，style/lw/ls 为
    # 箭头样式、线宽与线型，颜色统一用 ARROW
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle=style, mutation_scale=10,
                                 linewidth=lw, linestyle=ls, color=ARROW))


def padding(ax, x, y, width=PAD_W):
    # 画一个表示 VMM 填充的小矩形：斜线纹理（hatch="///"），灰蓝色底
    ax.add_patch(Rectangle((x, y), width, BAR_H, facecolor=PAD,
                           edgecolor="#89919B", hatch="///", linewidth=0.8))


def draw_rank_bar(ax, x0, y, local_text, buffer_text, fontsize=8):
    # Local payload ends at a block boundary; the pool starts there.
    # 画单个 rank 的一行条形：左侧黄色本地载荷 + 右侧粉色通信缓冲，
    # 两端再补上 VMM 填充块（本地载荷结束于块边界，缓冲池从那里开始）。
    # 左端（条形起点之前）的填充块
    padding(ax, x0 - PAD_W, y)
    # 右端（本地+缓冲结束之后）的填充块
    padding(ax, x0 + LOCAL_W + BUFFER_W, y)
    # 画黄色本地载荷矩形（从 x0 起宽 LOCAL_W）
    ax.add_patch(Rectangle((x0, y), LOCAL_W, BAR_H,
                           facecolor=YELLOW, edgecolor=EDGE, linewidth=1.0))
    # 在黄色矩形中央写本地载荷文字（如 "local experts [epn, H, H']"）
    ax.text(x0 + LOCAL_W / 2, y + BAR_H / 2, local_text,
            ha="center", va="center", fontsize=fontsize, color=TEXT)
    # 画粉色通信缓冲矩形（紧随本地载荷之后，宽 BUFFER_W）
    ax.add_patch(Rectangle((x0 + LOCAL_W, y), BUFFER_W, BAR_H,
                           facecolor=PINK, edgecolor=EDGE, linewidth=1.0))
    # 在粉色矩形中央写缓冲文字（如 "prefetch buffer [epn, H, H']"）
    ax.text(x0 + LOCAL_W + BUFFER_W / 2, y + BAR_H / 2, buffer_text,
            ha="center", va="center", fontsize=fontsize, color=TEXT)


def draw_rank_rows(ax, x0, y_top, pitch, title, local_text, buffer_text,
                   show_rank_labels):
    # 画一个面板：自上而下 R 行 rank 条 + 底部"连续 GEMM 视图"标注 + 右侧汇总总线。
    # bar_w：单个 rank 行的总宽度（本地 + 缓冲）
    bar_w = LOCAL_W + BUFFER_W
    # 在面板顶部中央写标题（如 "layer ℓ" / "layer ℓ+1"）
    ax.text(x0 + bar_w / 2, y_top + BAR_H + 0.42, title,
            ha="center", va="center", fontsize=11, color=TEXT)
    # 收集每行条形的垂直中心，供后面画总线连线
    centers = []
    for rank in range(R):
        # 第 rank 行的 y 坐标：自 y_top 向下按 pitch 递减
        y = y_top - rank * pitch
        # 记录该行中心高度
        centers.append(y + BAR_H / 2)
        if show_rank_labels:
            # 仅第一个面板在左侧标注 "EP rank N"
            ax.text(x0 - PAD_W - 0.15, y + BAR_H / 2, f"EP rank {rank}",
                    ha="right", va="center", fontsize=9, color=TEXT)
        # 画该 rank 的一行条形
        draw_rank_bar(ax, x0, y, local_text, buffer_text)

    # —— 画右侧"共享物理池"总线：每行经一段水平线汇入竖直总线 ——
    tail_right = x0 + bar_w
    bus_x = tail_right + PAD_W + 0.3
    for center in centers:
        # 从每行右端（跳过填充块）画水平线到总线
        ax.plot([tail_right + PAD_W, bus_x], [center, center], color=ARROW, lw=1.1)
    # 画竖直总线（连接最上与最下中心）
    ax.plot([bus_x, bus_x], [centers[-1], centers[0]], color=ARROW, lw=1.1)
    # 在每条水平线与总线的交点画一个小圆点
    ax.plot([bus_x] * R, centers, "o", ms=3.5, color=ARROW)
    # 最下面一行的 y 坐标
    bottom = y_top - (R - 1) * pitch
    # 在条形组下方画一个双端竖线箭头（|-|），表示整组的连续跨度
    ax.annotate("", xy=(x0, bottom - 0.14),
                xytext=(x0 + bar_w, bottom - 0.14),
                arrowprops=dict(arrowstyle="|-|", color=ARROW, lw=1))
    # 在箭头下方标注：所有 rank 拼成连续的 GEMM 视图 [2*epn, H, H']
    ax.text(x0 + bar_w / 2, bottom - 0.36,
            "Contiguous GEMM view [2*epn, H, H']", ha="center", fontsize=8)
    # 返回总线 x 位置与各行中心，供调用方继续连线
    return bus_x, centers


def draw_all_rank_view(ax, x0, y, title, buffer_text, width=1.7):
    # 画底部"全 rank 视图"面板：R 个 rank 的缓冲并排成 [R, epn, H, H']，
    # 并标注 rank 步长（stride）的取值方式。
    # 面板标题
    ax.text(x0 + R * width / 2, y + BAR_H + 0.28, title,
            ha="center", va="center", fontsize=10, color=TEXT)
    for rank in range(R):
        # 每个 rank 格子的载荷宽度 = 格宽 - 填充宽
        payload_w = width - PAD_W
        # 画该 rank 的粉色载荷矩形
        ax.add_patch(Rectangle((x0 + rank * width, y), payload_w, BAR_H,
                               facecolor=PINK, edgecolor=EDGE, linewidth=1.0))
        # 载荷右侧补一个填充块（对齐间隙）
        padding(ax, x0 + rank * width + payload_w, y)
        # 在格子中央写 "rank N + 缓冲名"
        ax.text(x0 + rank * width + payload_w / 2, y + BAR_H / 2,
                f"rank {rank}\n{buffer_text}", ha="center", va="center",
                fontsize=8, color=TEXT)
    # 在面板下方画双向箭头，标示相邻 rank 之间的步长
    ax.annotate("", xy=(x0, y - 0.2), xytext=(x0 + width, y - 0.2),
                arrowprops=dict(arrowstyle="<->", color=ARROW, lw=1))
    # 箭头下写 "rank stride"
    ax.text(x0 + width / 2, y - 0.47, "rank stride", ha="center", fontsize=8)
    # 补充说明：stride(0) = 对齐后 rank 字节数 / 元素大小；每个 rank 的载荷保持连续
    ax.text(x0 + R * width / 2, y - 0.83,
            "stride(0) = aligned rank bytes / element size; each rank payload stays contiguous",
            ha="center", fontsize=8.5, color=TEXT)


def buffer_figure(path, buffer_name, local_name, notes):
    # 通用绘图流程：先画顶部共享物理池与两个 layer 面板，再画底部全 rank 视图，
    # 最后加图例说明并保存。buffer_name 为 "prefetch"/"reduce"，local_name 为
    # "local experts"/"local grad"，notes 为底部两行注释文字。
    # 创建 12x7.2 英寸画布与坐标轴
    fig, ax = plt.subplots(figsize=(12, 7.2))
    # 让坐标轴铺满整张图（去掉白边）
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # 数据坐标范围：x 0..15，y 0.1..9.4
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0.1, 9.4)
    # 不显示坐标轴刻度与边框
    ax.axis("off")

    # 顶部中央的红色"物理显存池"矩形的位置与尺寸
    physical_x, physical_y, physical_w, physical_h = 5.2, 7.95, 4.6, 0.55
    # 画红色物理池矩形
    ax.add_patch(Rectangle((physical_x, physical_y), physical_w, physical_h,
                           facecolor=RED, edgecolor=EDGE, linewidth=1.0))
    # 在矩形中央写两行文字："xx 物理池（每 rank）／跨层共享"
    ax.text(physical_x + physical_w / 2, physical_y + physical_h / 2,
            f"{buffer_name} physical pool, per rank\nshared across layers", ha="center", va="center",
            fontsize=9, color="white", weight="bold")

    # 两个 layer 面板首行的 y 坐标与行距
    y_top = 6.65
    pitch = 0.82
    # 两个面板的 (x 起点, 标题)：layer ℓ 与 layer ℓ+1
    panels = ((1.65, "layer ℓ"), (8.15, "layer ℓ+1"))
    # 记录每个面板的总线位置与各行中心
    buses = []
    for panel_idx, (x0, title) in enumerate(panels):
        # 画该 layer 面板的 R 行 rank 条与总线；仅左面板显示 rank 标签
        bus_x, centers = draw_rank_rows(
            ax, x0, y_top, pitch,
            title,
            f"{local_name}\n[epn, H, H']",
            f"{buffer_name} buffer\n[epn, H, H']",
            show_rank_labels=panel_idx == 0,
        )
        # 保存总线信息
        buses.append((bus_x, centers))
        # 从物理池底部向该面板首行画一条弧线箭头，表示缓冲映射自共享物理池
        # （左面板从池的左 1/4 处出发、右面板从右 3/4 处出发，弧向相反）
        arrow(ax,
              physical_x + (0.25 if panel_idx == 0 else 0.75) * physical_w,
              physical_y,
              x0 + LOCAL_W + BUFFER_W / 2,
              y_top + BAR_H,
              rad=-0.08 if panel_idx == 0 else 0.08)

    # 底部全 rank 视图面板的位置与格子宽度
    view_x, view_y, view_w = 3.7, 2.6, 1.9
    # 画 [R, epn, H, H'] 的全 rank 缓冲视图
    draw_all_rank_view(
        ax, view_x, view_y,
        f"{buffer_name}-buffer view on every rank  [R, epn, H, H']",
        f"{buffer_name} buffer",
        width=view_w,
    )
    # 从左面板总线底部向全 rank 视图左端画箭头（视图同样映射自物理池）
    arrow(ax, buses[0][0], buses[0][1][-1],
          view_x + 0.2, view_y + BAR_H, rad=-0.12)
    # 从右面板总线底部向全 rank 视图右端画箭头
    arrow(ax, buses[1][0], buses[1][1][-1],
          view_x + R * view_w - 0.2, view_y + BAR_H, rad=0.12)

    # 图例：在右上角画一个填充块样例并标注 "VMM padding（在逻辑张量视图之外）"
    padding(ax, 4.0, 8.82)
    ax.text(4.55, 9.07, "VMM padding (outside logical tensor views)",
            va="center", fontsize=9, color=TEXT)

    # 在底部并排写两行说明文字（notes 必须恰好两条）
    for y, note in zip((1.15, 0.65), notes, strict=True):
        ax.text(7.5, y, note,
                ha="center", va="center", fontsize=8.5, color=TEXT)

    # 以 200 dpi、白底保存图片，并关闭画布释放内存
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def weight_buffer_figure(path):
    # 生成权重/前取缓冲示意图：本地名 "local experts"、缓冲名 "prefetch"，
    # 附注说明：本地权重占 VMM 块末尾、prefetch 载荷从下一块边界开始；
    # GEMM 视图跳过首尾填充，全 rank 池视图靠 rank stride 跳过填充。
    buffer_figure(
        path, "prefetch", "local experts",
        (
            "Local weights occupy the end of their VMM block; prefetch payload starts at the next block boundary.",
            "The GEMM view skips prefix/tail padding. The all-rank pool view skips padding via its rank stride.",
        ),
    )


def grad_buffer_figure(path):
    # 生成梯度/归约缓冲示意图：本地名 "local grad"、缓冲名 "reduce"，
    # 附注说明：每份完整 fp32 本地梯度占一个对齐物理块、grad 载荷结束处即
    # reduce 载荷开始处；GEMM 梯度视图内部无填充，只有逻辑参数梯度参与框架归约。
    buffer_figure(
        path, "reduce", "local grad",
        (
            "One aligned physical block per full fp32 local grad; grad payload ends where the reduce payload starts.",
            "The GEMM grad view has no internal padding; only logical parameter grads enter framework reduction.",
        ),
    )


if __name__ == "__main__":
    # 脚本直接运行时：依次生成两张 README 用图
    weight_buffer_figure("figure/weight_buffer.png")
    grad_buffer_figure("figure/grad_buffer.png")
