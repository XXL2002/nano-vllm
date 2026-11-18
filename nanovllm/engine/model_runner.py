import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size   # 张量并行总进程数
        self.rank = rank
        self.event = event

        # 初始化分布式通信环境 通过nccl建立通信组
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        # 设置默认数据类型
        default_dtype = torch.get_default_dtype()   # 保存原始默认数据类型
        torch.set_default_dtype(hf_config.dtype)
        # 设置默认设备
        torch.set_default_device("cuda")
        # 加载模型
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        # 初始化 Sampler
        self.sampler = Sampler()
        """
            预热模型，使用虚拟输入执行一次模型前向计算: 
                1. 提前触发 CUDA Kernel 初始化。
                2. 触发 TorchInductor 等编译优化。
                3. 预热 FlashAttention、Triton 等计算路径。
                4. 统计 GPU 显存峰值，为后续 KV Cache 分配提供依据
        """
        self.warmup_model()
        # 分配 KV Cache
        self.allocate_kv_cache()
        # CUDA Graph 捕获
        if not self.enforce_eager:
            self.capture_cudagraph()
        # 恢复默认配置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 初始化共享内存和子进程通信
        if self.world_size > 1:
            if rank == 0:           # Rank 0 创建共享内存
                # 创建一个名称为 nanovllm 的共享内存区域，大小为：2^20 = 1 MiB 主进程后续会将 RPC 指令写入这里。例如 self.write_shm("run", seqs, True)
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()  # 等待所有 rank 进程到达此处，确保共享内存已创建

            else:                   # 其他 Rank 连接共享内存
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()     # 进入子进程循环，等待 Rank 0 发来的 RPC 指令并执行

    # 资源释放：关闭共享内存、销毁进程组、清理 CUDA Graph
    def exit(self):
        # 清理共享内存
        if self.world_size > 1:
            self.shm.close()    # 当前进程关闭自己对共享内存的映射
            dist.barrier()      # 等待所有 rank 进程到达此处，确保共享内存已关闭
            if self.rank == 0:
                self.shm.unlink()   # Rank 0 删除共享内存对象，释放资源
        # 删除 CUDA Graph 以及 Graph Pool
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        # 等待 GPU 工作完成
        torch.cuda.synchronize()
        # 销毁分布式通信环境
        dist.destroy_process_group()

    # rank > 0 在 __init__ 末尾进入循环等待
    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    # rank > 0 的子进程从共享内存读取 RPC 指令
    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()      # 把 Event 恢复成未触发状态。
        return method_name, args

    # rank 0 将 RPC 指令写入共享内存，并唤醒其他 rank 进程
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0   # 检查调用者身份
        data = pickle.dumps([method_name, *args])       # 序列化方法名和参数
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")     # 将长度转换成 4 字节的小端序整数，写入共享内存的前 4 个字节
        self.shm.buf[4:n+4] = data
        for event in self.event:    # rank 0 持有所有 worker 的 Event
            event.set()

    # Rank 0 广播一个方法调用，让各个进程同步执行
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)      # 先广播给 worker
        method = getattr(self, method_name, None)
        return method(*args)                        # 再自己执行


    # 用一批假序列跑一次最大规模的 prefill
    # 这时 KV Cache 还没分配，k_cache.numel() 为 0，3.3 里 Attention 的写 Cache 分支被跳过，测出的峰值只包含权重和激活。
    def warmup_model(self):
        # 清理显存缓存并重置峰值统计
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()    # 峰值统计从这里重新开始
        # 确定预热请求的长度和数量
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        # 创建一批假序列并执行预热
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        # 再次清理显存缓存
        torch.cuda.empty_cache()

    # 测出峰值后, 根据显存利用率和峰值计算 KV Cache 的可分配块数, 并在 GPU 上分配 KV Cache 张量
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config

        # 1. 获取 GPU 显存使用情况
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]        # PyTorch 自上次重置峰值统计以来的最大显存分配量
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 当前 PyTorch 已分配的显存量  NOTE: peak - current 预留模型推理过程中可能产生的临时显存

        # 2. 计算单个 Block 需要多少显存
        # 计算当前 GPU 负责的 KV Head 数量，以及每个 Head 的维度
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # M_block = 2 * L * B * H_kv * D_h * sizeof(dtype) 表示所有层各增加一个 Block 的总大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        
        # 3. 计算可以分配多少个 Block
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        # 4. 一次性分配所有 KV Cache
        # 一个 6 维大张量装下所有层、所有块的 K 和 V, 形状为[2, L, N_blocks, B, H_kv, D_h]，其中 2 表示 K 和 V
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        
        # 5. 将 KV Cache 绑定到每个 Attention 层
        layer_id = 0
        for module in self.model.modules():     # 遍历模型中的所有子模块
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id] # 创建 Tensor 视图，共享 self.kv_cache 的底层显存
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1


    # 将多个请求长度不同的 Block Table 补齐为相同长度，再转换为 GPU 上的二维 Tensor，供 FlashAttention 访问 KV Cache
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    """
        prefill 序列张量化：变长序列拼在一起
            主要工作：
                1. 收集本轮需要计算的 Token ID 和位置索引。
                2. 构造变长 Attention 所需的 cu_seqlens_q 和 cu_seqlens_k。
                3. 根据 Block Table 计算 slot_mapping。
                4. 将数据传输到 GPU，并设置 Attention 上下文。
    """
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []          # 本轮需要计算的 Token ID
        positions = []          # Token 在原始序列中的位置
        cu_seqlens_q = [0]      # 每个请求的 Query 边界
        cu_seqlens_k = [0]      # 每个请求的 Key/Value 边界
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []       # Token 到 KV Cache 物理 Slot 的映射
        block_tables = None     # 逻辑 Block 到物理 Block 的映射
        for seq in seqs:
            start = seq.num_cached_tokens           # 已经计算并缓存的 Token 数
            seqlen_q = seq.num_scheduled_tokens     # 本轮需要计算的新 token 数
            end = start + seqlen_q                  # 本轮计算结束后的序列长度
            seqlen_k = end                          # Key/Value 的长度 = 已缓存的 Token + 本轮计算的新 token

            # 收集 Token ID 和位置 用于RoPE编码和模型输入
            input_ids.extend(seq[start:end])        # Q 只包含本轮要算的 token
            positions.extend(range(start, end))

            # 构造 cu_seqlens_q 和 cu_seqlens_k 用于变长 Attention
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            # 计算最大序列长度，用于 FlashAttention 的 Kernel 调用
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:    # warmup 的假序列没有块，不写 Cache
                continue
            # 计算本轮 Token 涉及哪些逻辑 Block
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            # 把 [start, end) 映射成物理 slot（源码按块成段 extend，这里逐 token 展开）
            for i in range(start_block, end_block):
                # 计算物理 Slot 起始位置
                slot_start = seq.block_table[i] * self.block_size   # 逻辑块号 -> 物理块号 -> 物理 Slot 起始位置
                if i == start_block:    # 如果是第一个 Block，跳过已经缓存的位置
                    slot_start += start % self.block_size
                if i != end_block - 1:  # 如果不是最后一个 Block，取整块
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:                   # 如果是最后一个 Block，取到本轮计算结束的位置（end - i * self.block_size）
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))    # 将计算出的物理 Slot 加入映射列表

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # 有历史 KV：前缀命中或 chunked 续算
            """
                NOTE: 何时带 block_tables?
                      没有历史时 Q 和 K 一样长, kernel 直接用本轮新算出的 k/v
                      只要有一条序列的 K 比 Q 长，说明它的前半段 KV 已经在 Cache 里，这时才准备 block_tables, Attention 改从 Cache 读完整的 K/V
            """
            block_tables = self.prepare_block_tables(seqs)
        
        """
            将数据传输到 GPU:
                - pin_memory=True  : 将 CPU Tensor 放在页锁定内存中，方便 GPU 进行 DMA 传输。
                - non_blocking=True: 允许异步进行 CPU 到 GPU 的数据传输（满足相应内存和设备条件时）。
        """
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 设置 Attention 上下文
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables) # True 表示当前是 Prefill
        return input_ids, positions


    """
        准备 Decode 阶段模型输入和 KV Cache 元数据，主要工作为：
            1. 收集每个请求最后一个 Token。
            2. 计算该 Token 的位置索引。
            3. 计算它在 KV Cache 中的物理写入位置。
            4. 准备历史 KV Cache 的 Block Table。
    """
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        """
            NOTE: context_lens 记录每条序列的长度，用于 KV Cache 版本的 FlashAttention
                告诉 flash_attn_with_kvcache 每条序列要读多少个历史 K/V。
                它把新 token 也算进去，因为 Attention 先执行 store_kvcache 把新 token 的 K/V 写进 Cache, 再做注意力计算。
        """
        context_lens = []
        for seq in seqs:
            """
                NOTE: 读 last_token 而不是 token_ids[-1]: 
                    TP>1 时 worker 收到的 decode 序列只有 last_token, 没有完整的 token_ids
            """
            input_ids.append(seq.last_token)    # 收集每条序列的最后一个 Token（之前 Token 的 K、V 已经保存在 KV Cache 中，不需要重新计算）
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))       # 包含这个新 token 在内的总长度
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1) # 调度器在 may_append 里已经为跨块的 token 分好了新块，所以新 token 一定落在 block_table[-1] 里
        
        # 将数据传输到 GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 准备历史 KV Cache 的 Block Table
        block_tables = self.prepare_block_tables(seqs)
        # 设置 Attention 上下文
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # 运行时按实际 batch 大小挑一张不小于它的图，把数据拷进静态缓冲区再回放
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # 判断使用哪种执行模式
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            # 根据实际 Batch Size 选择 Graph
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)] # next() 返回第一个满足条件的元素，x >= bs 表示选择的 Graph 的 Batch Size 不小于实际 Batch Size
            # 更新静态输入 Tensor
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # 更新 KV Cache 元数据
            graph_vars["slot_mapping"].fill_(-1)    # 补齐的位置 slot = -1：不写 Cache
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()      # 补齐的位置长度为 0：不读 Cache
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 重放 CUDA Graph
            graph.replay()
            # CUDA Graph 只捕获到 Hidden States 为止, 还要取出有效的 Hidden States 并计算 Logits
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    """
        流程: 
            张量化序列并 set_context → run_model(eager 或 CUDA Graph 回放)
            → compute_logits → rank 0 采样 → reset_context
    """
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids


    """
        提前为不同的 Decode Batch Size 捕获 CUDA Graph，后续推理时根据实际 Batch Size 选择合适的 Graph，直接 Replay，减少 CPU 发起 GPU Kernel 的开销.
            主要工作：
                1. 创建 CUDA Graph 所需的静态输入和输出 Tensor
                2. 确定需要捕获哪些 Batch Size
                3. 对每种 Batch Size 进行预热和 CUDA Graph 捕获
                4. 保存静态 Tensor，供后续 Replay 使用
    """
    @torch.inference_mode()     # torch.inference_mode() 表示函数内部执行推理模式，不记录梯度，也不进行 Autograd 相关操作
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)             # CUDA Graph 最大支持捕获的 Batch Size
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size    # 一个请求最多需要多少个 KV Cache Block
        
        # 创建静态输入和输出 Tensor
        input_ids = torch.zeros(max_bs, dtype=torch.int64)  
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 确定需要捕获哪些 Batch Size
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

        # 初始化 Graph 容器
        self.graphs = {}        # 保存不同 Batch Size 对应的 CUDA Graph
        self.graph_pool = None  # 保存 CUDA Graph 使用的显存池句柄: 通过共享 Graph Pool，可以让不同 Graph 复用显存分配空间

        # 逐个捕获 CUDA Graph
        for bs in reversed(self.graph_bs):  # NOTE: reversed() 让大 Batch Size 的 Graph 先捕获，后续小 Batch Size 的 Graph 可以复用大 Batch Size Graph 分配的显存池
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])

            # 预热
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup

            # 捕获 CUDA Graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture 调用model()，捕获范围只到 Transformer 输出 Hidden States 为止, compute_logits 和采样在图外执行
            
            if self.graph_pool is None:     # 第一次捕获完成后，获取当前 Graph 使用的显存池供后续复用
                self.graph_pool = graph.pool()
            
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存静态 Tensor，供后续 Replay 使用
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
