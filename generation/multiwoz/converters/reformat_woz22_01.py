import collections

import os, sys, json


GEN_API_CONFIG = 'GenAPIConfig'
DO_API_CALL = 'DoAPICall'

SPEAKER_USER = 'USER'
SPEAKER_SYSTEM = 'SYSTEM'


def get_action(actions, all_actions):
    output = 'default'
    for action in actions:
        output = action.split('-')[0].lower()
        if output != 'booking' and output != 'general':
            break
        if 'Recommend' in action:
            # recommend 是最高优先级，排他
            break
    if output == 'booking':
        for actions in all_actions[::-1]:
            for action in actions:
                output = action.split('-')[0].lower()
                if output != 'booking' and output != 'general':
                    return output
    return output


def simplify_params(history_api_params_list):
    out = collections.defaultdict(dict)
    for api_params_list in history_api_params_list:
        for api_params in api_params_list:
            service  = api_params['service']
            slots = {
                k.replace(f'{service}-', ''): v[0] for k, v in api_params['slot_values'].items() if v
            }
            out[service].update(slots)
    return out


def enrich_api_params(api_params):
    api_params_list = []
    for service, slots in api_params.items():
        api_params = {
            'service': service,
            'active_intent': f'find_{service}',
            'slot_values': {f'{service}-{k}': [v] for k, v in slots.items()}
        }
        api_params_list.append(api_params)
    return api_params_list


def convert_sft_types(dialog):
    dialog_id = dialog.get('dialogue_id').replace('.json', '').lower()
    services = dialog.get('services', [])

    target_services = {
        "attraction",
        "hotel",
        "restaurant",
        "train",
        "taxi"
    }
    if [x for x in services if x not in target_services]:
        return {}
    if not services:
        return {}

    turns = dialog.get('turns', [])
    api_generate_turns = []
    rag_generate_turns = []
    casual_generate_turns = []

    all_actions = []
    all_utterances = []
    idx = 0
    while idx < len(turns):
        speaker = turns[idx].get('speaker', '')
        utterance = turns[idx].get('utterance', '')

        if utterance == GEN_API_CONFIG:
            api_generate_turn = turns[idx]
            api_call_turn = turns[idx+1]
            turn = turns[idx+2]
            turn_id = turn.get('turn_id', '')
            api_generate_turns.append([
                turn_id, turn.get('actions', []), api_generate_turn.get('reference', []), 'api'
            ])
            rag_generate_turns.append([
                turn_id, turn.get('actions', []), api_call_turn.get('reference', {})
            ])
            all_actions.append(turn.get('actions', []))
            all_utterances.append(turn['speaker'] + ': ' + turn['utterance'])
            idx += 3
        else:
            if speaker == SPEAKER_SYSTEM:
                turn = turns[idx]
                turn_id = turn.get('turn_id', '')
                casual_generate_turns.append([
                    turn_id, turn.get('actions', []), turn.get('asked_slots', {}), 'chat'
                ])
                all_actions.append(turn.get('actions', []))
            all_utterances.append(speaker + ': ' + utterance)
            idx += 1

    api_generate_datas = []
    history_api_params_list = []

    for api_generate_turn in sorted(api_generate_turns + casual_generate_turns, key=lambda x:int(x[0])):
        turn_id, actions, api_params_list, turn_type = api_generate_turn
        turn_id = int(turn_id)

        if turn_type == 'api':
            history_api_params_list.append(api_params_list)

        action = get_action(actions, all_actions)

        api_generate_datas.append({
            'dialog_id': dialog_id,
            'turn_id': turn_id,
            'type': 'api_generation',
            'action': action,
            'label_type': 'Query',
            'history': all_utterances[0: turn_id],
            'label': simplify_params(history_api_params_list)
        })

        if dialog_id == 'pmul0187':
            print(turn_id, api_generate_datas[-1]['label'])

    casual_generation_datas =[]
    for casual_generate_turn in casual_generate_turns:
        turn_id, actions, asked_slots, _ = casual_generate_turn
        turn_id = int(turn_id)
        casual_generation_datas.append({
            'dialog_id': dialog_id,
            'turn_id': turn_id,
            'type': 'casual_generation_no_slots',
            'action': get_action(actions, all_actions),
            'label_type': 'Normal',
            'history': all_utterances[0: turn_id],
            'label': all_utterances[turn_id].replace("SYSTEM: ", "")
        })
        if asked_slots:
            casual_generation_datas[-1]['type'] = 'casual_generation'
            casual_generation_datas[-1]['asked_slots'] = asked_slots

    rag_generate_datas = []
    for rag_generate_turn in rag_generate_turns:
        turn_id, actions, search_results = rag_generate_turn
        turn_id = int(turn_id)
        rag_generate_datas.append({
            'dialog_id': dialog_id,
            'turn_id': turn_id,
            'type': 'rag_generation',
            'action': get_action(actions, all_actions),
            'history': all_utterances[0: turn_id],
            'search_results': search_results,
            'label': all_utterances[turn_id].replace("SYSTEM: ", "")
        })

    return {
        'api': api_generate_datas,
        'rag': rag_generate_datas,
        'casual': casual_generation_datas,
    }

if __name__ == '__main__':
    data_root = './multiwoz/data/MultiWOZ_2.2'

    for type in ['train', 'dev', 'test']:
        input_dir = f'{data_root}/{type}_sft'
        output_dir = f'./woz.2.2.real'

        os.makedirs(output_dir, exist_ok=True)

        key2fout = {
            'api': open(f'{output_dir}/{type}.api.json', 'w'),
            'rag': open(f'{output_dir}/{type}.rag.json', 'w'),
            'casual': open(f'{output_dir}/{type}.casual.json', 'w'),
            'act': open(f'{output_dir}/{type}.act.json', 'w')
        }
        for filename in os.listdir(input_dir):
            if 'dialogues_' not in filename:
                continue
            for dialog in open(f'{input_dir}/{filename}'):
                dialog = json.loads(dialog)
                for key, datas in convert_sft_types(dialog).items():
                    if not datas:
                        continue
                    datas = sorted(datas, key=lambda x:x['turn_id'])
                    for data in datas:
                        key2fout[key].write(json.dumps(data) + '\n')
                        key2fout[key].flush()

                    if key in ['api']:
                        for data in datas:
                            data['type'] = 'act_selection_baseline_dst'
                            slots = {k: v for k, v in data.get('label', {}).items()}
                            data['label'] = {'current_service': data['action'], 'slots': slots}
                            del data['label_type']
                            key2fout['act'].write(json.dumps(data) + '\n')
        [f.close() for f in key2fout.values()]
