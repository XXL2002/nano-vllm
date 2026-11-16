"""
    Qwen3 的词表约 15 万行, Embedding 和 LMHead 的权重都很大, 也要切分。
    Nano-VLLM 将模型的词嵌入矩阵按照词表维度切分到不同 GPU, 每张 GPU 只保存一部分 Token 的 Embedding, 通过 AllReduce 合并结果, 得到完整的 Embedding 输出。
    词表并行输出层(LM Head), 负责将 Transformer 最后一层的 Hidden States 转换为词表上的 Logits, 通过ALLGather操作将各 GPU 的 Logits 汇总, 以便后续 Sampling 选择下一个 Token。
    NOTE: 切输入维就 reduce，切输出维就 gather
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,    # 词表大小, 例如 150000
        embedding_dim: int,     # Embedding 维度, 例如 4096
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()          # 当前进程在张量并行中的rank编号
        self.tp_size = dist.get_world_size()    # 当前张量并行的总进程数
        assert num_embeddings % self.tp_size == 0

        # 计算每张 GPU 的词表范围
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        # 创建并加载本地 Embedding 权重
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    # 加载当前 GPU 的 Embedding 权重分片
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)         # 每张 GPU 的词表大小，即 num_embeddings_per_partition
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # 前向计算: 根据输入的 Token ID 查找对应的 Embedding, 并通过 AllReduce 汇总结果
    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 通过掩码判断每个 Token ID 是否属于当前 GPU的词表范围, 如果不属于则置为0
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)   # 平移到本地下标，越界的先置 0
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y   # 不属于本卡的行清零
            dist.all_reduce(y)          # 将每张卡都查一次表，但只有负责该 token 的卡查出非零向量，其余卡输出 0，all_reduce 求和后就是完整结果
        return y


"""
    词表并行输出层(Vocabulary-Parallel Language Model Head), 
    负责将 Transformer 最后一层的 Hidden States 转换为词表上的 Logits, 以便后续 Sampling 选择下一个 Token。
    其中, logits是每个Token的预测概率分布, 形状为 (batch_size, seq_len, vocab_size).

    每张 GPU 只计算自己负责的词表部分的 Logits，最后通过 Gather 将各 GPU 的结果拼接起来。
"""
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()     # 获取当前推理 Step 的上下文信息

        # prefill只需要计算最后一个 Token 的 Logits, decode阶段无需特殊处理
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1     # cu_seqlens_q 保存的是累计长度， 例如 [0, 4, 10] 表示第一条序列长度为4，第二条序列长度为6，最后一个Token的索引为3和9
            x = x[last_indices].contiguous()                # contiguous() 使上述末尾元素连续存储，便于后续的 Gather 操作
        logits = F.linear(x, self.weight)

        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None # rank 0 的进程创建一个列表来存储所有 GPU 的 Logits，其他 rank 不需要创建
            dist.gather(logits, all_logits, 0)  # gather到 rank 0 的进程
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None   # 沿最后一个维度拼接
        return logits
