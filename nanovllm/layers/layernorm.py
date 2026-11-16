import torch
from torch import nn


class RMSNorm(nn.Module):
    """
        NOTE: LLM 普遍用 RMSNorm 替代 LayerNorm。
        RMSNorm 去掉了减均值和偏置，只按均方根缩放，计算更少，效果相当
    """
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # 不含残差的归一化
    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()                               # 上升精度，避免在低精度下计算均方根时出现数值不稳定
        # 1) 计算均方值, 这里的var是均方值，RMSNorm不需要减去均值
        var = x.pow(2).mean(dim=-1, keepdim=True)   # dim=-1表示对最后一个维度求均值，keepdim=True 保持维度不变
        # 2) 归一化，乘以均方根的倒数
        x.mul_(torch.rsqrt(var + self.eps))         # rsqrt 是 Reciprocal Square Root，即平方根的倒数
        x = x.to(orig_dtype).mul_(self.weight)      # 还原精度，并乘以可学习的缩放参数
        return x

    # 含残差的归一化
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype

        x = x.float().add_(residual.float())    # 残差连接
        residual = x.to(orig_dtype)             # NOTE: 这里的 residual 是归一化前的残差，后续会被传递到下一层, 其他的和 rms_forward 一样
        
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
