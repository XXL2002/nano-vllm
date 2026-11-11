import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 1) 从 kwargs 里挑出 Config 认识的字段，构造全局配置
        config_fields = {field.name for field in fields(Config)}    # fields(Config)：Python dataclasses 提供的函数，用于获取 Dataclass 中定义的所有字段。
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        # 把块大小同步给 Sequence
        Sequence.block_size = config.kvcache_block_size

        # 2) 张量并行：用 spawn 启动 rank>0 的 worker 子进程
        self.ps = []        # 存放子进程的列表
        self.events = []    # 存放子进程的事件对象列表，用于进程间通信和同步
        """
        NOTE: 为什么用 spawn 而不是 Linux 默认的 fork?
            CUDA 运行时不支持在 fork 出的子进程里重新初始化。
            父进程一旦初始化过 CUDA，fork 出的子进程再用 GPU 就会报错。
            spawn 启动的是全新的解释器进程，每个 rank 各自初始化自己的 GPU。
        """
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            """
            NOTE: 主从进程靠共享内存 + Event 通信：rank>0 的 ModelRunner 在子进程的 __init__ 末尾进入循环，等待任务。
                  主进程把方法名和参数 pickle 后写进共享内存，再 set 每个 worker 的 Event 通知它读取。
            """
            event = ctx.Event() # 创建一个进程同步事件，用于主进程和当前子进程之间的通知与同步。
            process = ctx.Process(target=ModelRunner, args=(config, i, event))  # 创建一个新的子进程，运行 ModelRunner 类的实例，传入配置、rank 和事件对象作为参数。
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank 0 的 ModelRunner 跑在主进程
        self.model_runner = ModelRunner(config, 0, self.events)

        # 3) 分词器只在主进程持有
        # NOTE: 编码和解码都是 CPU 任务，TP worker 只需要 token id，不需要各自加载一份分词器。
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id    # 把EOS的实际ID回填到config

        # 4) 调度器 + 退出清理
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)  # NOTE: 向 Python 注册一个程序退出回调函数，当 Python 解释器正常退出时，自动调用 self.exit()，完成资源清理

    # NOTE: 退出时通过同一套 RPC [Remote Procedure Call（远程过程调用）] 给子进程发 "exit"，再 join 回收，避免残留进程。
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner   # 删除当前 LLMEngine 对 ModelRunner 对象的引用。
        for p in self.ps:
            p.join()    # 阻塞当前主进程，直到对应的子进程执行结束

    # 将一条 prompt 变成一个 Sequence 放进调度器。prompt 可以是字符串（内部先 encode）也可以是现成的 token id 列表
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    # 一次完整的「调度 → 前向 → 写回」
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()    # 调度器返回本轮要处理的序列列表和是否是 prefill 阶段的标志
        """
            NOTE: num_tokens 用正负号区分阶段。prefill 时返回正数，即这一步实际计算的 token 总量（开启 chunked prefill 后可能只是 prompt 的一部分）；
            decode 时返回负的 batch 大小 -len(seqs)。generate 根据符号分别计算 Prefill 和 Decode 吞吐（tok/s）显示在进度条上。
            这一行的位置不能挪到 postprocess 之后，否则 num_scheduled_tokens 已经被清零，prefill 吞吐会一直显示为 0。
        """
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)


        token_ids = self.model_runner.call("run", seqs, is_prefill) # 跑模型 + 采样
        self.scheduler.postprocess(seqs, token_ids, is_prefill)     # 写回、判断结束
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    # NOTE: 主循环： add_request -> step -> post_process
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)    # tqdm 是一个用于在命令行中显示进度条的库
        # 单个 SamplingParams （不是列表类型）复制给每条 prompt；也可以传一个等长列表
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}    # 此处为字典
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()  # 记录当前时间，用于计算每步的吞吐量
            output, num_tokens = self.step()    # 跑一步，拿回本步完成的序列
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 序列完成的先后顺序和提交顺序不同，按 seq_id 排序还原
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]    # 字典->token_ids列表，按 seq_id 排序
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs] # token_ids 列表->字典列表，包含解码后的文本和对应的原始token_ids
        return outputs
