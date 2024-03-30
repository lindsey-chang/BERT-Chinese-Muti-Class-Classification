import erniebot
import json
import random
import time
import re

with open('./input_word.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

erniebot.api_type = 'aistudio'
erniebot.access_token = "9e6d196a904ceb6fb788421afc913cacc2298567"  # 此处需要将你的token也就是 https://aistudio.baidu.com/index 里面获取

save_json = []

for line in lines:
    out_data = {}
    line = line.strip()
    out_data['input_word'] = line
    input_prompt = f'你现在是一名熟悉汽车各种功能的专家，你需要帮助用户提供驾驶车辆需要了解的各种知识，你需要给出专业、可靠、有逻辑的回答，同时用词还需要具有亲和力。在保持原有句子意思的前提下，改进提问方式，使其符合人类用语习惯，请一定给我生成50个不同的泛化,只回答改进后的句子：{line},需要返回的就是我们人类对于这条指令可能给出的相应的口语化命令'
    out_set = set()
    count = 1
    while len(out_set) < 500:
        print(f"第{count}次，处理词：{line}")
        response_input = erniebot.ChatCompletion.create(
            model='ernie-3.5',
            messages=[{
                'role': 'user',
                'content': input_prompt
            }]
        )
        result = response_input.result
        for temp in result.split("\n"):
            temp = re.sub(r'^\d+\.\s', '', temp)
            out_set.add(temp)
        count+=1
    input_prompt = f'你现在是一名熟悉汽车各种功能的专家，你需要帮助用户提供驾驶车辆需要了解的各种知识，你需要给出专业、可靠、有逻辑的回答，同时用词还需要具有亲和力。在保持原有句子意思的前提下，改进提问方式，使其符合人类用语习惯，请一定给我生成500个不同的泛化,只回答改进后的句子：{line},需要返回的就是我们人类对于这条指令可能给出的相应的口语化命令'
    out_data['input_prompt'] = input_prompt
    out_data['out_set'] = list(out_set)
    save_json.append(out_data)

with open('./out_data.json', 'w', encoding='utf-8') as file:
    json.dump(save_json, file, ensure_ascii=False, indent=4)

