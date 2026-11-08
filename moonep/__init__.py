# 从子模块 api 导入 MoonEP 的核心入口类 Buffer（封装 EP 通信缓冲区/连接池的创建与销毁）
from .api import (
    Buffer,
)
# 从子模块 planning 导入 MoE 通信规划器 MoonEPCommPlan
from .planning import MoonEPCommPlan

# 声明 `from moonep import *` 时对外暴露的公开接口：仅 Buffer 与 MoonEPCommPlan 两个符号
__all__ = [
    "Buffer",
    "MoonEPCommPlan",
]
