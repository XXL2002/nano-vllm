"""
    旋转位置编码: RoPE (Rotary Position Embedding)
"""

from functools import lru_cache
import torch
from torch import nn

# 旋转位置编码
# 前向时把最后一维按前后两半拆成(x1,x2)，第i维和第i+d/2维组成一对，做二维旋转
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)      # 沿最后一个维度将 Tensor 均分为两部分
    # 对x1和x2对应的位置应用旋转变换，cos和sin是预先计算好的旋转角度的余弦和正弦值
    y1 = x1 * cos - x2 * sin                        # 矩阵第一行
    y2 = x2 * cos + x1 * sin                        # 矩阵第二行
    return torch.cat((y1, y2), dim=-1).to(x.dtype)  # torch.cat() 沿最后一个维度拼接


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,                     # 每个注意力头的维度
        rotary_dim: int,                    # 旋转位置编码的维度，通常等于 head_size
        max_position_embeddings: int,       # 模型可以处理的最大序列长度，决定了位置编码的范围
        base: float,                        # 旋转位置编码的基数
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size

        # NOTE: RoPE（旋转位置编码）的预计算过程, 包括每个维度对应的旋转频率、每个位置对应的旋转角度、所有位置的正弦和余弦，将结果保存为 Buffer，供后续 Query 和 Key 使用。
        # 计算每个维度的旋转频率，公式为 1 / (base^(i/rotary_dim))，其中 i 为偶数索引
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 生成所有 Token 的位置索引
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # NOTE: 计算每个位置和每个维度的旋转角度，使用爱因斯坦求和接口，得到一个二维矩阵 freqs，其中行表示位置，列表示维度
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # i：取第一个向量的第 i 个元素。j：取第二个向量的第 j 个元素。ij：输出一个二维矩阵，分别以 i 和 j 作为行、列索引。
        
        # 预计算 cos 和 sin 
        cos = freqs.cos()
        sin = freqs.sin()
        # 拼接 cos 和 sin，并在第一个维度上增加一个维度，方便后续广播使用
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # 将 cache 注册为模块的 Buffer，而不是可训练参数
        self.register_buffer("cos_sin_cache", cache, persistent=False)  # persistent=False 表示该 Buffer 不会被保存到模型的状态字典中，通常用于临时计算或缓存数据

    # RoPE 是逐元素的小算子，编译后可以融合成少量 kernel，减少显存读写
    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


# 参数相同时返回同一个对象，所以 28 层共享一份 cos_sin_cache，不会每层各存一张表。
@lru_cache(1)   # 1表示缓存最近一次调用的结果
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
