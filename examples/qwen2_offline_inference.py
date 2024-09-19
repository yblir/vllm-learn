import os
import sys
import time

sys.path.append('/mnt/e/PyCharm/insteresting/vllm-0.5.4/')

from vllm_module import LLM, SamplingParams
# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, set_seed

set_seed(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# model_path = '/mnt/e/PyCharm/PreTrainModel/qwen2_15b_instruct'
model_path = '/mnt/e/PyCharm/PreTrainModel/Meta-Llama-3.1-8B-Instruct'
# model_path = '/mnt/e/PyCharm/PreTrainModel/qwen2_7b'
# model_path = '/media/xk/D6B8A862B8A8433B/data/qwen2-15b-instruct'

params = {
          # "repetition_penalty": 1.1,
          "temperature"       : 0.6,
          'max_tokens'        : 64,
          'n'                 : 1,
          "top_p"             : 0.9,
          "top_k"             : 50,
}
sample_params = SamplingParams(**params)
llm = LLM(model=model_path,
          dtype='half',
          # enable_prefix_caching= True,
          # max_tokens=64,
            # dtype='float16'
          # 把模型层均分到n个gpu上, 而不是运行n个完整模型
          # tensor_parallel_size=1
          max_model_len=1000,
          # gpu利用率最大70%
          # gpu_memory_utilization=0.5,
          )
tokenizer = AutoTokenizer.from_pretrained(model_path, )

# 构造模板
prompt = '介绍下京杭大运河'
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': prompt}
]

text = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)

messages2 = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'how far you go'}
]

text2 = tokenizer.apply_chat_template(conversation=messages2, tokenize=False, add_generation_prompt=True)

messages3 = [
    {'role': 'system', 'content': '你是一个诗人'},
    {'role': 'user', 'content': '用100字左右，描述下台风贝碧嘉'}
]

text3 = tokenizer.apply_chat_template(conversation=messages3, tokenize=False, add_generation_prompt=True)

# 京杭大运河是中国历史上最长的运河，也是世界上最古老的、最长的、最繁忙的运河之一。它连接了中国的首都北京和杭州，总长约1,776公里，跨越了山东、河北
# It seems like you're referencing the popular song "How Far I'll Go" from the Disney movie Moana. The song's lyrics speak about a young girl's desire to explore the world beyond her island home and discover her true identity.
# It seems like you're referencing the popular song "How Far I'll Go" from the Disney movie Moana. The song's lyrics speak about a young girl's desire to explore the world beyond her island home and discover her true identity.

# It seems like you're referencing the popular song "How Far I'll Go" from the Disney movie Moana. The song's lyrics speak about a young girl's desire to explore the world beyond her island home and discover her true identity.

# It seems like you're referencing the popular song "How Far I'll Go" from the Disney movie Moana. The song's lyrics speak about a young girl's desire to explore the world beyond her island home and discover her true identity.
# 台风贝碧嘉（Bebinca），又称为台风贝比娜，是一场强烈的台风，于2021年8月发生在太平洋西北部。贝碧嘉是本年第9号热带气旋，也是本年最强的
# 台风贝碧嘉（Bebinca），是2019年太平洋台风季的一场台风。它于9月14日在菲律宾海形成，初期为热带低气压，后于9月15日增强为热带风暴，9月16
# It seems like you're referencing a song title. "How Far I'll Go" is a popular song from the Disney movie Moana (2016). The song, written by Lin-Manuel Miranda, is Moana's big musical number where she sings about her desire to explore the ocean and discover her true identity.
# 台风贝碧嘉（Bebinca），是2019年太平洋台风季的一场台风。它于9月14日在菲律宾海形成，初期为热带低气压，后于9月15日增强为热带风暴，9月16
# 台风贝碧嘉（Bebinca），是2019年太平洋台风季的一场台风。它于9月14日在菲律宾海形成，初期为热带低气压，后于9月15日增强为热带风暴，9月16
t1=time.time()
outputs = llm.generate(
        # 当tokenizer.apply_chat_templat中 tokenize为 False 时激活prompts
        prompts=[text3,text2,],
        # prompts=[text2],    # 46
        # 当tokenizer.apply_chat_templat中 tokenize为 True 时激活prompt_token_ids,与prompts二选一
        # prompt_token_ids=[text,text2,text3],

        sampling_params=sample_params
)
print("耗时 =",time.time()-t1)
for output in outputs:
    # prompt = output.prompt
    # print(prompt)
    print(output.outputs[0].text)

    # print('------------------------------------------')
    # for i,item in enumerate(range(3)):
    #     print(output.outputs[i].text)
        # print(output.outputs[i].token_ids)
    # print('------------------------------------------\n')

