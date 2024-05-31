import json
import os
from collections import defaultdict

from generation.multiwoz.db import DataBase as DB




def load_dialog_acts():
    act_file = f'{data_root}/dialog_acts.json'
    data = json.load(open(act_file))
    act2count = defaultdict(float)
    act2count_v2 = defaultdict(float)
    dialog2turn2actions = defaultdict(dict)
    for dialogue_id, dialog_turns in data.items():
        for turn_id, turn_info in dialog_turns.items():
            dialog_acts = []
            for act in list(turn_info['dialog_act'].keys()):
                act2count[act] += 1
                # if 'Booking' in act:
                #     dialog_acts.append('Booking')
                # elif 'general' in act:
                #     dialog_acts.append('General')
                # else:
                dialog_acts.append(act)
                if int(turn_id) % 2 == 0:
                    act2count_v2[dialog_acts[-1]] += 1
            dialog2turn2actions[dialogue_id][turn_id] = list(set(dialog_acts))
    print(json.dumps(act2count, sort_keys=True, indent=2))
    print(json.dumps(act2count_v2, sort_keys=True, indent=2))
    return dialog2turn2actions


def parse_frames(frames, speaker, turn_acts, service2slot_vals):
    # request 为用户主动要求更新
    services = set()
    for frame in frames:
        service = frame.get('service', '')

        if not service or not speaker:
            continue

        state = frame.get('state', {})
        slot_k2vs = {
            k: v for k, v in state.get('slot_values', {}).items()
            if 'dontcare' not in v
        }
        active_intent = state.get('active_intent', 'NONE')

        if active_intent != 'NONE' and speaker == 'USER':
            slot_k2vs = {k: vs for k, vs in slot_k2vs.items() if vs}
            if slot_k2vs or service == 'taxi':
                services.add(service)
                service2slot_vals[service] = slot_k2vs
    return services


if __name__ == '__main__':
    # download from https://github.com/budzianowski/multiwoz.git

    db = DB()

    data_root = './multiwoz/data/MultiWOZ_2.2'

    dialog2turn2acts = load_dialog_acts()

    for type in ['train', 'dev', 'test']:
        data_dir = f'{data_root}/{type}/'
        data_out_dir = f'{data_root}/{type}_sft'
        os.makedirs(data_out_dir, exist_ok=True)

        target_services = {
            "attraction",
            "hotel",
            "restaurant",
            "train",
            "taxi"
        }

        bad_logs = open(f'{data_out_dir}/bad.logs.txt', 'w')

        service2field_config = {
            'attraction': ['area', 'type', 'name', 'entrance fee', 'openhours'],
            'restaurant': ['area', 'food', 'pricerange', 'type', 'address', 'introduction', 'phone', 'postcode'],
            'hotel': ['area', 'internet', 'parking', 'pricerange', 'stars', 'type', 'address', 'phone', 'postcode'],
            'train': ['departure', 'destination', 'day', 'arriveby', 'leaveat', 'price', 'trainid', 'duration'],
        }

        service2samples = defaultdict(list)
        for input_file in sorted(os.listdir(data_dir)):
            fout = open(f'{data_out_dir}/{input_file}', 'w')
            input_file = f'{data_dir}/{input_file}'
            for dialog in json.load(open(input_file)):
                did = dialog.get('dialogue_id')
                services = dialog['services']
                if len([x for x in services if x not in target_services]) > 0:
                    continue
                new_dialog = []
                bad_dialog = False
                service2slot_vals = defaultdict(dict)

                dialog_acts = dialog2turn2acts[did]

                turns = dialog.get('turns', [])
                for i, turn in enumerate(turns):
                    turn_id = turn.get('turn_id', -1)
                    turn_acts = dialog_acts.get(turn_id, '')
                    turn_services = [x.split('-')[0].lower() for x in turn_acts if x.split('-')[0].lower() in target_services]

                    if did == 'SNG0771.json':
                        print(turn_id, turn_services)

                    speaker = turn.get('speaker', '')
                    utterance = turn.get('utterance', '')
                    frames = turn.get('frames', [])

                    services = parse_frames(frames, speaker, turn_acts, service2slot_vals)
                    if not turn_services:
                        turn_services.extend(services)

                    new_dialog.append({
                        'turn_id': turn_id,
                        'speaker': speaker,
                        'actions': turn_acts,
                        'utterance': utterance
                    })

                    if speaker == 'SYSTEM' and i < len(turns) - 1 and not frames and '?' in utterance:
                        continue

                    elif turn_services and speaker == 'USER':
                        # By User Update
                        items = {}
                        api_configs = []
                        for service in turn_services:
                            slot_values = service2slot_vals[service]
                            if not slot_values:
                                continue
                            # new slot is detected
                            api_configs.append({
                                'service': service,
                                'active_intent': f'find_{service}',
                                'slot_values': slot_values
                            })

                        new_dialog.append({
                            'turn_id': f'{int(turn_id)}:follow_by_user_select_api',
                            'speaker': 'SYSTEM',
                            'utterance': f'GenAPIConfig',
                            'reference': api_configs
                        })
                        new_dialog.append({
                            'turn_id': f'{int(turn_id)}:follow_by_user_call_api',
                            'speaker': 'SYSTEM',
                            'utterance': 'DoAPICall',
                            'reference': []
                        })

                output = {
                    'dialogue_id': dialog['dialogue_id'],
                    'services': dialog['services'],
                    'turns': new_dialog
                }
                if bad_dialog:
                    continue

                fout.write(json.dumps(output) + '\n')
                fout.flush()
                for service in services:
                    if len(service2samples[service]) < 5:
                        service2samples[service].append(output)
                        break
            fout.close()
        print(json.dumps(service2samples, indent=2), file=open(f'{data_out_dir}/samples.json', 'w'), flush=True)
        bad_logs.close()