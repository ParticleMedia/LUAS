import collections
import copy
import multiprocessing
import os
import json
import time

import tqdm

from training_scripts.ft_datasets.agent_sft_act_dataset import AgentActDataset

from text_generation import Client


def call_tgi(prompt, tgi_server="http://YOUR_TGI_IP:YOUR_TGI_PORT"):
    client = Client(tgi_server)
    response = client.generate(prompt=prompt,
                               temperature=1.0,
                               best_of=2,
                               max_new_tokens=500,
                               repetition_penalty=1,
                               do_sample=False)
    return response.generated_text


def is_valid_action_response(output):
    try:
        output = json.loads(output)
        if 'slots' not in output:
            return False
        for k, v in output['slots'].items():
            if not isinstance(v, list) and not isinstance(v, dict):
                return False
        return True
    except:
        return False


def is_valid_api_response(output):
    try:
        json.loads(output)
        return True
    except:
        return False


def run(tgi_svr, output_file):
    sout = open(output_file, 'w')

    datas = [data for data in open(f'{input_dir}/{split}.act.json')]
    for data in tqdm.tqdm(datas):
        act_obj = json.loads(data)
        key = f'{act_obj["dialog_id"]}_{act_obj["turn_id"]}'

        if key not in key2sample:
            counter['sample_is_missing'] += 1
            continue

        prompt, label = AgentActDataset.prompting(act_obj)

        output = call_tgi(prompt, tgi_svr)
        if not is_valid_action_response(output):
            counter['decision_maker_error'] += 1
            continue

        act_output = json.loads(output)

        sout.write(json.dumps({
            'key': key,
            'pred_act': act_output,
            'real_act': act_obj['label']
        }) + '\n')
        sout.flush()
        counter['success'] += 1
    print(tgi_svr, json.dumps(counter, indent=2))


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--input_dir', type=str, default='agent_sft.woz.2.4.limit_1k')
    args.add_argument('--split', type=str, default='dev')
    args.add_argument('--host', type=str, default='http://0.0.0.0:8002')
    args = args.parse_args()

    input_dir = ''

    key2sample = {}
    for filename in [f'{args.split}.act.json']:
        for data in open(f'{args.input_dir}/{filename}'):
            obj = json.loads(data)
            key = f'{obj["dialog_id"]}_{obj["turn_id"]}'
            key2sample[key] = obj

        counter = collections.defaultdict(float)
        tgi_svr2output_file = {
            'http://YOUR_TGI_IP:YOUR_TGI_PORT': f'output.json',
        }

        pool = multiprocessing.Pool(len(tgi_svr2output_file))
        for tgi_svr, output_file in tgi_svr2output_file.items():
            pool.apply_async(
                func=run,
                args=(tgi_svr, output_file)
            )
        pool.close()
        pool.join()
