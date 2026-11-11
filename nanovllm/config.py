'''
全局配置类，包含模型路径、最大 token 数量、最大序列数、最大上下文长度、显存利用率、张量并行大小等参数。
'''
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str                              # 本地模型目录（必填） 
    max_num_batched_tokens: int = 16384     # 单步最多处理的 token 总数（Token Budget）
    max_num_seqs: int = 512                 # 单步最多并发的序列数
    max_model_len: int = 4096               # 支持的最大上下文长度，即单个请求允许的最大序列长度，包含prompt和生成的token在内的总长度
    gpu_memory_utilization: float = 0.9     # 允许占用的显存比例
    tensor_parallel_size: int = 1           # 张量并行 GPU 数
    enforce_eager: bool = False             # True 则禁用 CUDA Graph
    hf_config: AutoConfig | None = None     # 在 __post_init__ 里加载的 HF 配置
    eos: int = -1                           # EOS token id（初始化后由 tokenizer 填入）
    
    # NOTE: 256 是 nano-vllm 当前默认值。vLLM 使用自己的 PagedAttention 内核，常见默认 block_size=16，两者的默认值不要混为一谈。
    kvcache_block_size: int = 256           # 每个 KV 块容纳的 token 数

    # NOTE: 引擎不按最大长度给每条请求预留 KV，而是先加载权重、用一次最大规模的 prefill 测出激活峰值，再把剩余配额全部切成 KV 块。
    num_kvcache_blocks: int = -1            # KV 块总数（运行时按显存推算）

    def __post_init__(self):
        assert os.path.isdir(self.model)            # 必须是存在的本地目录
        assert self.kvcache_block_size % 256 == 0   # 块大小需 256 的整数倍，来自底层算子要求：flash-attn 的 flash_attn_with_kvcache 文档写明，传入 block_table（分页 KV）时 page_block_size 必须是 256 的倍数。
        assert 1 <= self.tensor_parallel_size <= 8  # TP 度在 1~8 之间
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
