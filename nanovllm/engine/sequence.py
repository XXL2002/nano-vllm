from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams

# 枚举生命周期
class SequenceStatus(Enum):
    WAITING = auto()    # 在等待队列里，还没开始算
    RUNNING = auto()    # 正在被调度执行
    FINISHED = auto()   # 已结束（命中 EOS 或达到 max_tokens）


# 请求状态机
class Sequence:
    block_size = 256    # 类变量：所有序列共享同一块大小
    counter = count()   # 类变量：自增 ID 生成器

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)            # prompt + 已生成的 token（拷贝一份，不改调用方的列表）
        self.last_token = token_ids[-1]             # 最后一个 token，decode 时作为模型输入
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0                  # 已经算完、KV 写进 Cache 的 token 数
        self.num_scheduled_tokens = 0               # 本轮被调度计算的 token 数
        self.is_prefill = True                      # 实例属性：是否还在 prefill 阶段
        self.block_table = []                       # 逻辑块 → 物理 block_id 的映射, 相当于页表
        # 采样超参直接摊平成实例属性（不另存 sampling_params 对象）
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property   # @property 将类中的方法转换为属性，使调用者可以像访问变量一样访问方法，而不需要加括号
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):   # 需要几个块（向上取整）
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):    # 最后一块装了几个 token
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):     # 获取第 i 块的 token_ids
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 跨进程传递时，pickle 会调用 __getstate__ 方法获取对象的必要状态信息。 pickle是一个用于序列化和反序列化对象的模块。
    def __getstate__(self):
        # prefill 需要整段 token；decode 只需要最后一个 token
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    # 对应的反序列化
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):    # prefill 阶段传过来的是整段 token_ids
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:   # decode 阶段传过来的是最后一个 token
            self.token_ids = []
            self.last_token = last_state
