import argparse
import collections
import json

import asyncio
import aiohttp
from tqdm.asyncio import tqdm

from training_scripts.ft_datasets.agent_sft_act_dataset import AgentActDataset


async def call_vllm(sem, pbar, idx, key, prompt, vllm_svr):
    req_obj = {
        "prompt": prompt,
        "best_of": 3,
        "temperature": 0.1,
        "max_tokens": 500,
        "use_beam_search": True
    }

    async with sem:
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{vllm_svr}/generate', json=req_obj) as response:
                output = await response.text()
                pbar.update(1)
                try:
                    output = json.loads(output)["text"][0][len(prompt):]
                except:
                    output = ""
                # print(idx, key, output)
                return idx, key, output

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


async def run_generations(vllm_svr, datas):
    generation_tasks = []
    sem = asyncio.Semaphore(30)
    pbar = tqdm(total=len(datas))

    for idx, data in enumerate(datas):
        act_obj = json.loads(data)
        key = f'{act_obj["dialog_id"]}_{act_obj["turn_id"]}'

        prompt, _ = AgentActDataset.prompting(act_obj)

        generation_tasks.append(asyncio.create_task(call_vllm(sem, pbar, idx, key, prompt, vllm_svr)))

    generations = await asyncio.gather(*generation_tasks)
    generations = sorted(generations, key=lambda x: x[0])

    return generations


def run(vllm_svr, output_file):
    open(output_file, "w").close()
    sout = open(output_file, "a")
    datas = [data for data in open(f'{input_dir}/{args.split}.act.json')]
    generations = asyncio.run(run_generations(vllm_svr, datas))

    for data_idx, data_key, output in generations:
        if not is_valid_action_response(output):
            counter['decision_maker_error'] += 1
            continue

        act_output = json.loads(output)

        sout.write(json.dumps({
            'key': data_key,
            'pred_act': act_output,
            'real_act': json.loads(datas[data_idx])['label']
        }) + '\n')
        sout.flush()
        counter['success'] += 1
    print(vllm_svr, json.dumps(counter, indent=2))
    sout.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='agent_sft.woz.2.4.limit_1k')
    args.add_argument('--split', type=str, default='dev')
    args.add_argument('--host', type=str, default='http://0.0.0.0:8002')
    args.add_argument('--tag', type=str, default='')
    args = args.parse_args()

    input_dir = f'./{args.dataset}/'

    counter = collections.defaultdict(float)
    vllm_svr2output_file = {
        args.host: f'{input_dir}/{args.split}.act.pred.vllm.13b.2e-5{args.tag}.json',
    }

    for vllm_svr, output_file in vllm_svr2output_file.items():
        run(vllm_svr, output_file)