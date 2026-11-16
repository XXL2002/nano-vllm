"""
    注意力层：先把新算出的 K/V 写进 KV Cache, 再做注意力计算。
    注意力计算在 prefill 和 decode 两个阶段调用的是不同的 FlashAttention 接口。
"""
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# 用 Triton 把单个 TOKEN 的 K/V 写进 Cache
# 根据 slot_mapping 找到每个 Token 在 KV Cache 中的物理存储位置，然后将它对应的 K、V 向量写入该位置
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,         # Key 在 Token 维度上的步长
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,   # Token 到物理 Cache Slot 的映射
    D: tl.constexpr,    # 编译器常量： 每个 Token 的 K/V 向量总维度   D = num_heads * head_dim, num_heads为本 rank 的 KV 头数
):
    # 计算槽位
    idx = tl.program_id(0)          # 每个 program 处理一个 token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return           # -1 表示这个 token 不需要写（padding）

    # 用传入的 stride 定位第 idx 个 token 的 D 个元素
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 然后将它们写入 KV Cache 中对应的 slot
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


# 包装函数：根据 slot_mapping 将一批 Token 的 K/V 写进 KV Cache
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape      # num_heads 实际是本 rank 的 KV 头数， N 为需要写入 KV Cache 的 Token 总数量
    D = num_heads * head_dim                # 一个 token 在一层里的 K（或 V）向量长度
    # 确保每个 token 的 D 个元素在内存里连续，kernel 才能一次读写一整段
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 启动 N 个 Triton Program Instance, 每个 Program 处理一个 Token 的 K/V 写入
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
    """
        NOTE: 也可以用 PyTorch 写同样的逻辑，大致是 k_cache.view(-1, D)[slot_mapping] = key.view(N, D)。
        问题出在 slot == -1 的位置：CUDA Graph 回放时 decode batch 会补齐到固定大小，补出来的 token 的 slot 是 -1，这些位置不能写。
        PyTorch 里要先用掩码筛掉它们，筛选结果的长度随数据变化，不适合放进 CUDA Graph。
        Triton kernel 每个 program 处理一个 token，遇到 -1 直接返回，张量形状始终固定，可以被 Graph 捕获。
    """


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale      # 1 / sqrt(head_dim)
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():     # cache 已分配才写
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:      # a) prefill 阶段，走变长flash_attention, 侧重并行处理多个新 Token
            if context.block_tables is not None:    # prefix cache
                """
                    NOTE: 若存在 Prefix Cache, 说明部分历史 K、V 已经保存在 KV Cache 中本次并没有计算
                    此时不再使用计算得到的部分 k、v, 而是从 KV Cache 中读取完整的 k、v 参与注意力计算，保证历史信息完整
                """
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        
        else:                       # b) decode 阶段，走 KV Cache 版本的 flash_attention, 侧重利用 KV Cache 高效读取历史信息 (K/V全部来自 KV Cache)
            # q.unsqueeze(1)补充长度维度：Q: [N,H_q,D] -> [N,1,H_q,D]
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
