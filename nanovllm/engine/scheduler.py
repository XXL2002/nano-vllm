from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs                         # 一轮最多几条序列
        self.max_num_batched_tokens = config.max_num_batched_tokens     # 一轮最多几个 token
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()     # 还没 prefill 完的序列（新请求、chunked 未算完、被抢占的）
        self.running: deque[Sequence] = deque()     # prefill 已完成、正在逐 token decode 的序列

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    """
        prefill 优先调度：只要 prefill 分支选出了至少一条序列，本轮就做 prefill;
                        一条都选不出来 (waiting 为空，或队头序列的块不够)，才去 running 做 decode。
        返回: (seqs, is_prefill)，即选中的这批序列，以及这一轮是 prefill 还是 decode
    """
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0  # 本轮已调度的Token数

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens    # 剩余 token 预算
            if remaining == 0:
                break
            if not seq.block_table:             # 首次调度：查前缀缓存 + 确认块够用
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:     # 放不下
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break   # 剩余 token 预算不够，且已经选了至少一条序列，放弃新增
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            # 允许对「本轮第一条」做 chunked prefill：只算预算内的部分
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:  # 本轮调度后 prefill 结束
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):       # 需要抢占
                if self.running:
                    self.preempt(self.running.pop()) # 牺牲 running 队尾的序列
                else:
                    self.preempt(seq)                # 没有可牺牲的 running 序列了，只能抢占自己
                    break
                """
                    NOTE: while ... else 的 else 只在循环条件变为假 (can_append 成功) 时执行，被 break 跳出则不执行。
                    牺牲自己的分支以 break 结束，所以这条序列不会进入本轮 batch,它已经在 preempt 里被插回了 waiting 队头。
                """
            else:                                               # 无需抢占
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)  # 真的跨块就取新块
                scheduled_seqs.append(seq)

        assert scheduled_seqs
        # NOTE: 把选中的序列按原顺序放回running队头
        self.running.extendleft(reversed(scheduled_seqs))   # deque.extendleft() 是从左侧依次插入元素，插入顺序会反转, 因此需要reversed()
        return scheduled_seqs, False


    # 抢占
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING     # 打回等待队列
        seq.is_prefill = True                   # 下次要重新 prefill
        self.block_manager.deallocate(seq)      # 释放它占的所有 KV 块，num_cached_tokens 清零
        self.waiting.appendleft(seq)            # 插回队头，下一轮优先重新调度


    # 模型算完、采样出 token 后，引擎调用 postprocess 把结果写回序列并判断是否结束
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)     # 本轮新填满的块登记哈希
            seq.num_cached_tokens += seq.num_scheduled_tokens   # 本轮算过的 token 记为已缓存
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue    # chunked prefill 还没算完：丢弃本轮采样结果
            seq.append_token(token_id)              # 追加采样出的新 token
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 结束就释放 KV
                self.running.remove(seq)
