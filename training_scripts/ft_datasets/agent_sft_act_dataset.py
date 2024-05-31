import os

import torch
import tqdm
from torch.utils.data import Dataset

import copy
import json
from typing import Dict
from training_scripts.ft_datasets.agent_sft_common import PERSONA_PROMPT_DICT, agent_tokenize


ANSWER_TYPE_PROMPT = {
    'act_selection_baseline_dst_emb': (
        'Please summary the following dialog into what the user want and what system provided.\n'
        'Here is the dialog:\n{history}\n'
        'Please give your summary:\n'
    ),
    'act_selection_baseline_dst': (
        'You are a local guide online, primarily handling local services like:\n'
        'find the user\'s place (such as attraction, hotel, train, restaurant or hospital), and calling taxis, contacting the police, or other convenient services.\n'
        'Your service is efficient and of high quality, earning widespread praise from the local community.\n'
        'Given the conversion history, Your task is to help determine whether the next response can be directly replied to or not.\n'
        'Please output the current_service based on the user last utterence.\n'
        'Please noted that your responses are not used in the action selection, except the hotel name and restaurant name that you provided.\n'
        'And also please output all the services\' information that need pay attention to from the whole conversion.\n'
        'Here is the conversion history:\n{history}\n'
        'the user lastest utterence: \n{user_utterence}\n'
        'The output should be in JSON format like {{"current_service": xxx, "slots": {{"service": {{"slot_key": "slot_val"}}}}}}\n'
        'Please give your decision:\n'
    ),
    'act_selection_baseline_dst_2.4': (
        'You are a local guide online, primarily handling local services like:\n'
        'find the user\'s place (such as attraction, hotel, train, restaurant or hospital), and calling taxis, contacting the police, or other convenient services.\n'
        'Your service is efficient and of high quality, earning widespread praise from the local community.\n'
        'Given the conversion history, Your task is to help determine whether the next response can be directly replied to or not.\n'
        'Please output the current_service based on the user last utterence.\n'
        'Please noted that your responses are not used in the action selection, except the hotel name and restaurant name that you provided.\n'
        'And also please output all the services\' information that need pay attention to from the whole conversion.\n'
        'Here is the conversion history:\n{history}\n'
        'the user lastest utterence: \n{user_utterence}\n'
        'The output should be in JSON format like {{"slots": {{"service": {{"slot_key": "slot_val"}}}}}}\n'
        'Please give your decision:\n'
    ),
}
ANSWER_TYPE_PROMPT['default'] = ANSWER_TYPE_PROMPT['act_selection_baseline_dst']


class AgentActDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=1024, do_padding=True, debug=False):
        type = 'train'
        input_files = [
            f'{dataset_config.root}/{dataset_config.dataset_dir}/{type}.{task}.json'
            for task in ['act']
        ]
        print(json.dumps(input_files, indent=2), flush=True)
        self.datas = []
        for input_file in input_files:
            datas = [json.loads(data) for data in open(input_file) if data.strip()]
            if partition == 'train':
                self.datas.extend(datas[0:1000])
            else:
                self.datas.extend(datas[0:100])

        self.max_words = max_words
        self.do_padding = do_padding
        self.tokenizer = tokenizer
        self.debug = debug

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        item: Dict = self.datas[index]
        prompt, label = AgentActDataset.prompting(item)
        return agent_tokenize(self.tokenizer, prompt, label, self.max_words, self.do_padding)

    @staticmethod
    def prompting(item: Dict):
        type = item['type']
        history = [x.replace('USER', 'user').replace('SYSTEM', 'you') for x in item['history']]

        if type == 'act_selection_baseline_dst_emb':
            prompt = ANSWER_TYPE_PROMPT[type].format(
                history=json.dumps(history, indent=2),
            )
            label = json.dumps(item['label'])

        elif type == 'act_selection':
            # persona = PERSONA_PROMPT_DICT.get(item['action'], PERSONA_PROMPT_DICT['default'])
            persona = PERSONA_PROMPT_DICT['default']
            prompt = ANSWER_TYPE_PROMPT[type].format(
                persona=persona,
                history=json.dumps(history[0:-1], indent=2),
                user_utterence=history[-1].replace('user: ', '')
            )
            label = json.dumps(item['label'])
        else:
            prompt = ANSWER_TYPE_PROMPT[type].format(
                history=json.dumps(history[0:-1], indent=2),
                user_utterence=history[-1].replace('user: ', '')
            )

            # 训练时不需要 current service，直接预测 slot 就可以
            if type == 'act_selection_baseline_dst_2.4' and 'current_service' in item['label']:
                del item['label']['current_service']

            label = json.dumps(item['label'])
        return prompt, label


if __name__ == '__main__':
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')

    items = json.load(open('./datas/agent_sft_act_data.json'))
    for item in items:
        output = AgentActDataset.tokenize(item, tokenizer, 1024, True)

