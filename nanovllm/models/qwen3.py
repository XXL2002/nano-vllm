import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,          # 旋转位置编码的基数
        rope_scaling: dict | None = None,   # 旋转位置编码的缩放参数, 用于处理长序列
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)       # reshape 成 [token, 头, head_dim]
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:   # Qwen3 的 Q/K/V 都不加偏置
            # RMSNorm 作用在最后一维 head_dim 上，所以把张量 reshape 成 [token, H, head_dim] 后直接送进去，就是逐头归一化
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)     # 归一化后再加旋转位置
        o = self.attn(q, k, v)                      # 形状为 (token, H_q, head_dim)
        output = self.o_proj(o.flatten(1, -1))      # o.flatten(1, -1) 展平o的后两维, 形状为 (token, H_q * head_dim), 送进行输出线性层
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    """
        NOTE: Qwen3 使用 Pre-Norm + 残差结构：归一化放在子层之前，残差直接跨过子层。
        每层：先归一化 + 注意力 -> 再归一化 + MLP
    """
    def forward(
        self,
        positions: torch.Tensor,        # 当前 Token 的位置索引，用于 RoPE 位置编码
        hidden_states: torch.Tensor,    # 当前层输入的隐藏状态
        residual: torch.Tensor | None,  # 残差连接中暂存的张量，第一层为 None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) 先归一化 + 注意力
        if residual is None:    # 第一层：残差从输入初始化
            # residual = hidden_states  保持原值用于残差连接
            # hidden_states = norm(hidden_states)   归一化
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:                   # 非第一层：残差来自上一层的输出
            # NOTE: nano-vllm 把残差相加放进了 norm 内部
            # residual = hidden_states + residual  保持原值之和用于残差连接
            # hidden_states = norm(hidden_states + residual)  归一化
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        
        # 2) 再归一化 + MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    nano-vllm 把 q/k/v 合并成了 qkv_proj、gate/up 合并成了 gate_up_proj，而 HuggingFace 的 checkpoint 里它们是分开的五个权重。
    加载时 loader.py 遍历 safetensors 里的权重名，按子串匹配这张映射表，把原始名字翻译成合并后的参数名和 shard_id
    """
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:  # Qwen3-0.6B 这类小模型的 Embedding 和 LMHead 共用同一张权重表
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 计算 Transformer，得到 Hidden States
        return self.model(input_ids, positions)

    """
        NOTE: compute_logits 单独调用 LMHead, forward 只算到 hidden state, logits 由 compute_logits 单独计算, 降低 CUDA Graph 捕获和重放的复杂度。
        这样 CUDA Graph 只需要捕获到 hidden state 为止, LMHead 的「prefill 只取最后 token」逻辑和后面的采样都放在 Graph 之外。
        - LMHead: 需要根据当前 Batch 的请求数量，动态选取每个请求最后一个 Token 的 Hidden State;
        - Sampling: Sampling 还涉及 Temperature、采样策略等操作。
    """
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 将 Hidden States 转换为词表 Logits
        return self.lm_head(hidden_states)
