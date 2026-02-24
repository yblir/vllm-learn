# -*- coding: utf-8 -*-
# @Time    : 2025/3/21 下午2:10
# @Author  : yblir
# @File    : vllm_inference.py
# explain  :
# =======================================================
import os
import warnings
warnings.filterwarnings("ignore")

# import sys
# from pathlib2 import Path
# os.environ["VLLM_NO_KERNEL"] = "1"
# sys.path.insert(0, str(Path.cwd() / 'vllm'))
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["VLLM_USE_V1"] = "1"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# import torch
# print(torch.cuda.is_available())

# model_path = '/mnt/e/checkpoints/Qwen2.5-3B-Instruct'
model_path = '/media/xk/D6B8A862B8A8433B/data/qwen2-15b-instruct'

params = {"repetition_penalty": 1.1,
          "temperature"       : 0.7,
          'n'                 : 2,
          "top_p"             : 0.8,
          "top_k"             : 20, }

sample_params = SamplingParams(**params)
llm = LLM(model=model_path,
          dtype='half',
          # dtype='float16'
          # 把模型层均分到n个gpu上, 而不是运行n个完整模型
          tensor_parallel_size=1,
          # gpu利用率最大70%
          # gpu_memory_utilization=0.7,
          )
tokenizer = AutoTokenizer.from_pretrained(model_path, )

# 构造模板
prompt = '介绍下京杭大运河'
messages = [
    {'role': 'system', 'content': '你是一个诗人'},
    {'role': 'user', 'content': prompt}
]

text = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)

messages2 = [
    {'role': 'system', 'content': '你是一个诗人'},
    {'role': 'user', 'content': 'how far you go'}
]

text2 = tokenizer.apply_chat_template(conversation=messages2, tokenize=False, add_generation_prompt=True)

messages3 = [
    {'role': 'system', 'content': '你是一个诗人'},
    {'role': 'user', 'content': '中国首都城市什么名字'}
]

text3 = tokenizer.apply_chat_template(conversation=messages3, tokenize=False, add_generation_prompt=True)
# print(text)
outputs = llm.generate(
        # 当tokenizer.apply_chat_templat中 tokenize为 False 时激活prompts
        # prompts=[text, text2, text3],
        prompts=[text, text2],
        # 当tokenizer.apply_chat_templat中 tokenize为 True 时激活prompt_token_ids,与prompts二选一
        # prompt_token_ids=[text,text2,text3],
        # 使用OCR技术抽取3000万底库图片文本，搭建ES检测系统，根据关键字检索出符合需求的文本对应的图片，经过图片分辨率，组合关键字等方案二次提取
        # 后，获得1.5万图片，最后经过人工挑选后获得3000张图片作为数据集。在2x8共16张2080ti分布式部署qwen2.5-vl-72B模型作为教师模型，精心
        # 编写提示词后对指定字段结构化输出，将结果作为训练label。训练阶段，使用2x8共16张TITAN RTX对qwen2.5-vl-7B分别进行分布式全量和lora微调，
        # 实验结果表明，在当前数据规模下，两种训练方式效果没有明显区别。评测阶段使用DeepSeek-R1-Distill-Qwen-32B对gt_label和预测la
        # bel再次进行结构化规整，整理后label，使用Levenshtein Distance方案计算准确率，在全量微调下，准确率为0.82，比qwen2.5-vl-72B
        # 少0.04，比原版qwen2.5-vl-7B高0.05。在部署阶段，将固定prompt+OCR提取的关键信息组成新的prompt，改写vllm源码缓存固定prompt的kv-cache
        # 优化模型推理速度。
        sampling_params=sample_params
)

for output in outputs:
    # prompt = output.prompt
    # print(prompt)
    # print(output)
    # print('------------------------------------------')
    for i, item in enumerate(range(1)):
        print(output.outputs[i].text)
        # print(output.outputs[i].token_ids)
    print('------------------------------------------\n')