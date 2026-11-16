import torch
from torch import nn
import torch.nn.functional as F

"""
    NOTE: Qwen3采用SwiGLU激活函数:
    引入门控机制：用一个独立的门矩阵 W_gate 来控制信息的通过量，而不是简单地对所有维度施加相同的激活函数。
"""
class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)   # 前一半是 gate，后一半是 up。 放在一起是为了合并gate_up_proj投影。
        return F.silu(x) * y    # Sigmoid Linear Unit : silu(x) = x * sigmoid(x) 作为门控函数，控制信息流动
