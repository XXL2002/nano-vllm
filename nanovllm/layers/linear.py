"""
    线性层: 包含张量并行信息, 支持复制式、列并行、行并行、合并列并行、QKV合并等多种形式
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


"""
    线性层基类，包含张量并行信息
"""
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        # 张量并行信息
        self.tp_dim = tp_dim                    # 权重沿哪个维度切分
        self.tp_rank = dist.get_rank()          # 当前进程在张量并行中的rank编号
        self.tp_size = dist.get_world_size()    # 当前张量并行的总进程数

        # 权重初始化
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader  # 将当前 Linear 层的 weight_loader() 方法绑定到 weight 参数对象上，因为 nano-vllm 加载模型权重时需要知道每个参数应该如何加载。
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# 复制式线性层：每个进程拥有完整的权重矩阵 W
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


"""
    NOTE: 列并行线性层
        W 形状为 (output_size, input_size)，在列维度上进行切分，每个进程拥有 W 的一部分列。
        输入 X 形状为 (batch_size, input_size)，输出 Y = X @ W^T 形状为 (batch_size, output_size)
        沿着列维度切分，则每个进程计算 Y 的一部分列，最后拼接即可，无需通信。
"""
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()     # 获取 Tensor Parallel 的总进程数
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)     # 0 表示沿着列维度切分权重矩阵 W

    # 加载当前 GPU 对应的权重分片
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data                     # 获取当前参数的张量数据
        shard_size = param_data.size(self.tp_dim)   # 获取分片大小，例如X行
        start_idx = self.tp_rank * shard_size       # 计算当前进程在张量并行中的起始索引
        # 提取当前进程对应的权重分片
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)    # narrow(dim, start, length) 表示沿指定维度，从 start 开始截取 length 个元素
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


"""
    合并列并行线性层：将多个列并行线性层合并为一个大线性层，合并存储和计算，并分别加载它们的权重分片
    例如: FFN第一层 gate_up:     把 SwiGLU 的 gate、up 合并，权重加载按整数 shard_id (0=gate、1=up) 索引 output_sizes
"""
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],    # 记录每个独立线性层的输出维度大小
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)   # 初始化时各个独立线性层的输出维度大小相加，得到合并后的总输出维度大小

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        # 计算当前独立线性层权重在合并矩阵中的起始位置
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size     # loaded_shard_id 表示当前加载的是第几个独立 Linear
        # 计算当前独立线性层的本地分片大小
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 定位目标分片在合并矩阵中的位置
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)

        # 从完整权重中提取当前 Rank 的分片
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


"""
    把 Q、K、V 三个投影合并成一个大矩阵，输出维是 (num_heads + 2 * num_kv_heads) * head_dim
    按头并行
"""
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size  # 即q占用的空间
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size # 即q+k占用的空间
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


"""
    NOTE: 行并行线性层
        W 形状为 (output_size, input_size)，在行维度上进行切分，每个进程拥有 W 的一部分行。
        输入 X 形状为 (batch_size, input_size)，输出 Y = X @ W^T 形状为 (batch_size, output_size)
        沿着行维度切分，则每个进程计算 Y 的一部分行，最后需要 all-reduce 汇总结果。
    例如: Attention 输出投影 o_proj, 把各卡的头拼回来，需要 all_reduce
         FFN 第二层 down_proj, 把各卡的 up 投影结果拼回来，需要 all_reduce
"""
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:    # bias 参数是 1D 张量，不需要切分，直接复制即可
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)   # 获取当前进程在张量并行中的分片大小，即列数
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # NOTE: 行并行线性层的前向计算需要 all-reduce 汇总结果
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)  # 只有 rank 0 的进程才有 bias，其他 rank 的进程没有 bias
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
