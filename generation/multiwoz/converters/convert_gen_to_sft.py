import argparse
import json
import os
import collections
import random


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


def simplify_params(history_api_params_list, remove_hotel_name, remove_restaurant_name, remove_attraction_name):
    out = collections.defaultdict(dict)
    for api_params_list in history_api_params_list:
        for api_params in api_params_list:
            service  = api_params['service']
            slots = {}
            for slot_key, slot_val in api_params['slot_values'].items():
                if not slot_val or 'trainid' in slot_key or service not in slot_key:
                    continue
                if remove_hotel_name and slot_key == 'hotel-name':
                    continue
                if remove_attraction_name and slot_key == 'attraction-name':
                    continue
                if remove_restaurant_name and slot_key == 'restaurant-name':
                    continue
                slots[slot_key.split('-')[-1]] = slot_val[0].lower()
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




def md5hash(text):
    import hashlib
    return hashlib.md5(f'{text}'.encode("utf-8")).hexdigest()


def convert_sft_types(dialog):
    # dialog_id = dialog.get('dialogue_id').replace('.json', '').lower()
    dialog_id = md5hash(json.dumps(dialog))
    services = dialog.get('services', [])

    count_log['all'] += 1

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

    def has_remove_name(turns, sevice):
        name, remove = '', -1
        for turn in turns:
            if turn['utterance'] == 'GenAPIConfig' and turn['reference'][0]['active_intent'] == f'find_{sevice}':
                name = turn['reference'][0]['slot_values'].get(f'{sevice}-name', [''])[0]
        for turn in turns:
            if ':' in turn['turn_id']:
                continue
            if remove == -1 and name and name in turn['utterance'] and turn['speaker'] == 'SYSTEM':
                remove = 1
            if name and name in turn['utterance'] and turn['speaker'] == 'USER':
                remove = 0
        if remove == 1:
            count_log[f'remove_{sevice}_name'] += 1
        return remove == 1

    remove_hotel_name = False # has_remove_name(turns, 'hotel')
    remove_attraction_name = False # has_remove_name(turns, 'attraction')
    remove_restaurant_name = False # has_remove_name(turns, 'restaurant')

    bad_dialog = False
    for turn in turns:
        if ('-name' in turn['utterance']
                or '-type' in turn['utterance']
                or '-area' in turn['utterance']
                or not turn['utterance']
        ):
            bad_dialog = True
    if bad_dialog:
        print('bad dialog', flush=True)

    def clean_utterance(utterance):
        for mark in ['[EOD]', '[EOF]', '[BOOKED]', '[RECOM]']:
            utterance = utterance.replace(mark, '')
            utterance = utterance.replace(mark.lower(), '')
        if "I recommend" in utterance:
            # utterance = utterance.replace('I recommend', random.choice(replace_phrases))
            utterance = utterance.replace("'", '')
        if 'recommend' in utterance:
            utterance.replace("'", '')
            utterance.replace("\"", '')
        utterance = utterance.replace('Wi-Fi', 'wifi')
        return utterance.strip() if '2.2' in output_dir else utterance.strip().lower()

    while not bad_dialog and idx < len(turns):
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
            all_utterances.append(turn['speaker'] + ': ' + clean_utterance(turn['utterance']))
            idx += 3
        else:
            if speaker == SPEAKER_SYSTEM:
                turn = turns[idx]
                turn_id = turn.get('turn_id', '')
                api_generate_turns.append([
                    turn_id, turn.get('actions', []), turn.get('reference', []), 'api'
                ])
                all_actions.append(turn.get('actions', []))
            all_utterances.append(speaker + ': ' + clean_utterance(utterance))
            idx += 1

    api_generate_datas = []
    history_api_params_list = []

    for api_generate_turn in sorted(api_generate_turns, key=lambda x:int(x[0])):
        turn_id, actions, api_params_list, turn_type = api_generate_turn
        turn_id = int(turn_id) - 1

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
            'label': simplify_params(history_api_params_list, remove_hotel_name, remove_restaurant_name, remove_attraction_name)
        })

    casual_generation_datas =[]
    for casual_generate_turn in casual_generate_turns:
        turn_id, actions, asked_slots, _ = casual_generate_turn
        turn_id = int(turn_id) - 1
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
        turn_id = int(turn_id) - 1
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


def convert():
    os.makedirs(output_dir, exist_ok=True)

    type = 'train'
    key2fout = {
        'api': open(f'{output_dir}/{type}.api.json', 'w'),
        'rag': open(f'{output_dir}/{type}.rag.json', 'w'),
        'act': open(f'{output_dir}/{type}.act.json', 'w')
    }
    n_dialog = 0
    for dialog in open(input_file):
        try:
            dialog = json.loads(dialog)
        except:
            continue
        for key, datas in convert_sft_types(dialog).items():
            if not datas:
                continue
            datas = sorted(datas, key=lambda x: x['turn_id'])
            for data in datas:
                key2fout[key].write(json.dumps(data) + '\n')
                key2fout[key].flush()

            if key in ['api']:
                for i, data in enumerate(datas):
                    data['type'] = 'act_selection_baseline_dst'
                    slots = {k: v for k, v in data.get('label', {}).items()}
                    data['label'] = {'current_service': data['action'], 'slots': slots}
                    if i == len(datas) - 1:
                        data['label']['current_service'] = 'general'
                    del data['label_type']
                    key2fout['act'].write(json.dumps(data) + '\n')
        n_dialog += 1
    [f.close() for f in key2fout.values()]
    print(json.dumps(count_log, indent=2))


if __name__ == '__main__':
    input_file = './../datas/multiwoz.json'

    args = argparse.ArgumentParser()
    args.add_argument('--woz22', action='store_true')
    args.add_argument('--woz24', action='store_true')
    args = args.parse_args()

    if args.woz22:
        count_log = collections.defaultdict(float)
        output_dir = './woz.2.2.gen'
        convert()

    if args.woz24:
        count_log = collections.defaultdict(float)
        output_dir = './woz.2.4.gen'
        convert()



