"""
    logits -> token: rank 0 拿到 logits 后, Sampler 按每条序列自己的温度采样出下一个 token。
    Temperature Sampling(温度采样)，通过调整 Logits 的温度、计算概率分布，再利用指数分布随机数完成采样，最终为 Batch 中每个请求选出一个 Token
"""
import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) # 每行除以各自的温度
        probs = torch.softmax(logits, dim=-1)   # 转换为概率分布
        """
            NOTE: 使用指数分布随机数进行采样，其与经典的 Gumbel-Max Sampling 等价，可以实现符合概率分布P的随机采样。
            1. torch.empty_like(probs).exponential_(1) 生成与 probs 形状相同的指数分布随机数矩阵, E_i ~ Exp(1)
            2. clamp_min_(1e-10) 将指数分布随机数矩阵中的最小值限制为 1e-10, 避免除以零
            3. probs.div_(E) 将概率分布除以指数分布随机数
            4. argmax(dim=-1) 选择得分最大的 Token
        """
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
