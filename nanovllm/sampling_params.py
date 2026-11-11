"""
    采样超参数
"""
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64    # 最多生成多少个 token，默认 64。既是停止条件，也用于 benchmark 时固定生成量。
    ignore_eos: bool = False    # 是否忽略 EOS 继续生成。bench.py 把它设为 True，保证每条序列都生成满 max_tokens，吞吐数据才可比。

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
