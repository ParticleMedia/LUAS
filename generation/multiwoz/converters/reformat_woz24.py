import collections
import json
import os


def get_meta(turn):
    service2slots = collections.defaultdict(dict)
    for service, slots in turn['metadata'].items():
        for slot_key, slot_val in slots['semi'].items():
            if not slot_val or slot_val == 'not mentioned':
                continue
            service2slots[service][slot_key] = slot_val
    return service2slots


if __name__ == '__main__':
    # download data from
    # https://github.com/smartyfh/MultiWOZ2.4.git
    input_dir = './MultiWOZ2.4/data/mwz2.4/'
    target_domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi']

    replaces = {
        'concert hall': 'concerthall',
        'night club': 'nightclub'
    }

    output_dir = './woz.2.4.real'
    os.makedirs(output_dir, exist_ok=True)

    def clean_data(x):
        for fc, tc in {
            ' -s': 's',
            ' .': '.',
            ' ,': ',',
            ' ?': '?',
            ' !': '?',
            ' :': ':'
        }.items():
            x = x.replace(fc, tc)
        return x

    for type in ['dev', 'test', 'train']:
        sout = open(f'{output_dir}/{type}.act.json', 'w')
        for dialog in json.load(open(f'{input_dir}/{type}_dials.json')):
            did = dialog['dialogue_idx']
            domains = dialog['domains']
            if len([x for x in domains if x not in target_domains]) > 0:
                continue
            history = []
            for i, turn in enumerate(dialog['dialogue']):
                user_turn, system_turn = turn['transcript'], turn['system_transcript']
                user_turn = clean_data(user_turn)
                system_turn = clean_data(system_turn)

                if system_turn:
                    history.append(f'SYSTEM: {system_turn}')
                history.append(f'USER: {user_turn}')

                domain = turn['domain']

                states = collections.defaultdict(dict)
                for state in turn['belief_state']:
                    slot_key, slot_val = state['slots'][0]
                    service = slot_key.split('-')[0]
                    slot_key = slot_key.split('-')[1].replace(' ', '')

                    if slot_val == 'dontcare':
                        if service not in states:
                            states[service] = {}
                        continue

                    slot_val = replaces.get(slot_val, slot_val)
                    states[service][slot_key] = slot_val

                if domain not in states:
                    states[domain] = {}

                sample = {
                    'dialog_id': did.replace('.json', '').lower(),
                    'turn_id': i * 2 + 1,
                    'type': 'act_selection_baseline_dst_2.4',
                    'history': history,
                    'label': {"slots": states}
                }
                sout.write(json.dumps(sample) + '\n')





