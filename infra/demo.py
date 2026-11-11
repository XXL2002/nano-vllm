import os
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")   # 本地模型目录

# 1) 构造引擎：本地路径 + 全局开关
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1) # enfoerce_eager=True 表示强制使用Eager Execution（即时执行模式），禁用CUDA Graph优化

# 2) 定义采样策略
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# 3) Chat 模型先套对话模板，再交给引擎
tokenizer = AutoTokenizer.from_pretrained(path) # 通过 Hugging Face Transformers 加载指定模型对应的 Tokenizer（分词器）
questions = ["用一句话介绍 PagedAttention", "100 以内有哪些质数"]

"""
    NOTE: 模板 对于Qwen3-0.6B模型，使用了一个简单的对话模板，包含用户和助手的角色标记。
    用户输入的问题被包装在 <|im_start|>user 和 <|im_end|>user 标签之间，而助手的回答则以 <|im_start|>assistant 开始。
    模板帮助模型理解对话的结构，从而生成更符合预期的回答。
    """
prompts = [
    tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                  tokenize=False, add_generation_prompt=True)   # tokenize=False 表示返回格式化后的字符串，而不是 Token ID; add_generation_prompt=True添加助手回复的起始标记<|im_start|>assistant，告诉模型接下来该生成助手的回答
    for q in questions
]

# 4) 批量生成：每个元素是 {"text": ..., "token_ids": ...}
outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print("\n")
    print(f"Prompt: {prompt!r}")
    print(f"Completion: {output['text']!r}")
