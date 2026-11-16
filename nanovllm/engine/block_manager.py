from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

# 每个 Block 对应 KV Cache 里一段物理显存
class Block:

    def __init__(self, block_id):
        self.block_id = block_id    # 物理块编号
        self.ref_count = 0          # 引用计数：几条序列在共享这块
        self.hash = -1              # 块内容的哈希（-1 表示还没填满、没登记）
        self.token_ids = []         # 块内 token（哈希命中后再逐个比对，防止哈希碰撞导致复用错块）

    # 块被填满、算好哈希时登记
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # 从空闲队列取出、分配给新序列时
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

# BlockManager 持有全部物理块、一个空闲队列和一张哈希表，对外提供分配、追加、回收三组操作
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()                  # 前缀缓存：哈希 → 块号
        self.free_block_ids: deque[int] = deque(range(num_blocks))      # 空闲块队列（先进先出）
        self.used_block_ids: set[int] = set()                           # 已分配块集合

    @classmethod    # @classmethod 装饰器将方法绑定到类而不是实例上
    # 哈希链前缀哈希 + 本块 token
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()  # 初始化一个 64 位哈希对象
        if prefix != -1:    # 如果有前缀哈希，就把它的字节表示喂给哈希对象（prefix是前一块的哈希值）
            h.update(prefix.to_bytes(8, "little"))  # 将整数转换为 8 字节的小端序字节表示
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 分配一个空闲块，返回块号。调用方要保证 free_block_ids 非空
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()    # 取出空闲最久的块
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 这块以前登记过哈希，且表里还指向它：要被覆盖了，先把登记删掉
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()   # ref_count = 1，清空 hash 和 token_ids
        self.used_block_ids.add(block_id)
        return block_id

    # 回收一个块并放回空闲队列末尾
    # NOTE: 不清空块的 hash 和 token_ids，哈希表里的登记也还在。这块真正被 _allocate_block 取走覆盖之前，后来的请求仍然可以命中它。
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    """
        分配前检查：沿哈希链逐块查前缀缓存，统计命中了几块，再判断空闲块够不够装下剩下的部分:
            - 放不下返回 -1, 否则返回命中块数。
            - 确认放得下之后，调度器再调 allocate(seq, num_cached_blocks) 把物理块挂到 seq.block_table 上。
    """
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks         # 需要新分配的块数（总块数 - 命中块数）
        for i in range(seq.num_blocks - 1):     # NOTE: 最后一块不查，永远重算（不缓存，即使填满了也需要重算获取logits）
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:  # 哈希没命中或命中但内容不一致
                break                   # 第一次没命中就停
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1     # 命中在用的块：不占空闲块
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    """
        分配( 用于prefill 前一次性分够块)：沿哈希链逐块查前缀缓存，命中就挂上去、没命中就分配新块，直到把 seq.num_blocks 块都挂上去。
    """
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        # 1. 先挂上命中块
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]

            if block_id in self.used_block_ids: # 命中块已经被别的序列占用，增加引用计数
                block.ref_count += 1
            else:                               # 已释放但还没被覆盖：从空闲队列里摘回来
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            
            seq.block_table.append(block_id)
        # 2. 再分配新块
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    # 回收 (用于序列结束或被抢占)
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):  # 从最后一块往前放，前缀缓存里可能有命中块，先释放后面的块，前面的块可能还在被别的序列占用
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:    # 没人用了才放回空闲队列队尾
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # 追加前检查
    def can_append(self, seq: Sequence) -> bool:
        # 只有「最后一个 token 正好落在新块的第一格」时才需要一个空闲块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    # 追加(用于 decode 时每步可能新增一块)：如果最后一个 token 正好落在新块的第一格，就分配一个空闲块挂上去
    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    # 登记：块填满后调用 hash_blocks
    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size                                # 本轮之前已填满的块数
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size     # 本轮之后已填满的块数
        if start == end: return                                                         # 本轮没有新填满的块
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1           # 接上链的前一环作为前缀哈希
        for i in range(start, end): # 遍历本轮新填满的块
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id                                   # 此处直接覆盖旧登记，NOTE: 因此_allocate_block 删除登记前要先确认 hash_to_block_id.get(block.hash) == block_id
