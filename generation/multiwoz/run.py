import argparse
import json
import multiprocessing
from generation.multiwoz.gen_utils import *

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--part', type=int, default=9)
    args = args.parse_args()

    os.environ['OPENAI_API_KEY'] = 'nb-8jh4Y8klEqhMTXjWn-LdfUmCYhVUIVwL4VHncJWjhNmekG1MIY55DgfIs1_iCxkL2HM'
    os.environ['OPENAI_BASE_URL'] = 'http://cooper.k8s.nb-prod.com/v1'

    hour = datetime.datetime.now().strftime('%Y-%m-%d_%H')
    part = str(args.part + 100)[1:]

    services_list, service2preference_list = [], []
    for i, data in enumerate(open('configs/preference_templates.json')):
        if i % 64 != args.part:
            continue
        obj = json.loads(data)
        services_list.append(obj['services'])
        service2preference_list.append(obj['service2preference'])

    cache = Cache(f'./datas/{hour}.part_{part}.txt')

    bos, eos = 0, len(services_list)
    services_list = services_list[bos:eos]
    service2preference_list = service2preference_list[bos:eos]

    run_batch(cache, services_list, service2preference_list)

