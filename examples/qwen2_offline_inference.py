import os
import sys
from ctypes import cdll

# cdll.LoadLibrary('/mnt/e/PyCharm/insteresting/vllm-0.5.4/cmake/libtorch.so')
# cdll.LoadLibrary('/mnt/e/PyCharm/insteresting/vllm-0.5.4/cmake/_C.abi3.so')
# cdll.LoadLibrary('/mnt/e/PyCharm/insteresting/vllm-0.5.4/cmake/_core_C.abi3.so')
# cdll.LoadLibrary('/mnt/e/PyCharm/insteresting/vllm-0.5.4/cmake/_moe_C.abi3.so')

sys.path.append('/mnt/e/PyCharm/insteresting/vllm-0.5.4/')

from vllm_module import LLM, SamplingParams
# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# model_path = '/mnt/e/PyCharm/PreTrainModel/qwen2_15b_instruct'
model_path = '/media/xk/D6B8A862B8A8433B/data/qwen2-15b-instruct'

params = {"repetition_penalty": 1.1,
          "temperature"       : 0.7,
          'n'                 :4,
          "top_p"             : 0.8,
          "top_k"             : 20, }
sample_params = SamplingParams(**params)
llm = LLM(model=model_path,
          dtype='half'
            # dtype='float16'
          # 把模型层均分到n个gpu上, 而不是运行n个完整模型
          # tensor_parallel_size=1
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
        prompts=[text,text2,text3],

        # 当tokenizer.apply_chat_templat中 tokenize为 True 时激活prompt_token_ids,与prompts二选一
        # prompt_token_ids=[text,text2,text3],

        sampling_params=sample_params
)

for output in outputs:
    # prompt = output.prompt
    # print(prompt)
    # print(output)
    # print('------------------------------------------')
    for i,item in enumerate(range(4)):
        print(output.outputs[i].text)
        print(output.outputs[i].token_ids)
    print('------------------------------------------\n')
    # generated_text = output.outputs[0].text
    # print(generated_text)
