import copy
import datetime
import json
import os.path
import random
import sys
import time

import tqdm

from fuzzywuzzy import fuzz

from collections import *
from generation.multiwoz.gpt_utils import *
from generation.multiwoz import gen_config as config
from generation.multiwoz.db import DataBase as MultiWozDB


__dir__ = os.path.split(os.path.realpath(__file__))[0]

woz_db = MultiWozDB()

class Cache:
    def __init__(self, output_file):
        self.root = './cache'
        os.makedirs(os.path.dirname(f'{self.root}/{output_file}'), exist_ok=True)
        self.sout = open(f'{self.root}/{output_file}', 'a')

    def write(self, data):
        self.sout.write(data)
        self.sout.flush()


def prepare_services():
    services = random.choice(config.service_combinations).split(',')
    services = list(services)
    random.shuffle(services)
    return list(services)


def update_preference(service2preference):
    for service, preference in service2preference.items():
        if random.random() >= 0.1:
            continue
        config_slot_keys = config.rollback_service2prefierence[service]
        update_slot_keys = [
            slot_key for slot_key in preference if slot_key in config_slot_keys
        ]
        if not update_slot_keys:
            continue
        update_slot_key = random.choice(update_slot_keys)
        update_slot_val = random.choice([
            slot_val for slot_val in config.rollback_service2prefierence[service][update_slot_key]
            if slot_val != preference[update_slot_key]
        ])
        print(f'random update [{update_slot_key}] '
              f'from [{preference[update_slot_key]}] to [{update_slot_val}]', flush=True)
        preference[update_slot_key] = update_slot_val

    print(f'user preference (from template) = {json.dumps(service2preference)}', flush=True)

    return service2preference


def prepare_preference(services):
    service2slots = {}
    for service in services:
        if service == 'taxi':
            continue
        if not service2slots or service == 'train':
            service2slots[service] = random.choice(woz_db.service2db[service])
        elif random.random() <= 0.5:
            service2slots[service] = random.choice(woz_db.service2db[service])
        else:
            areas = {slots['area'] for slots in service2slots.values() if 'area' in slots}
            if len(areas) == 1:
                same_area_slots = [
                    slots for slots in woz_db.service2db[service] if slots['area'] in areas
                ]
                if same_area_slots:
                    slots = random.choice(same_area_slots)
                else:
                    slots = random.choice(woz_db.service2db[service])
            else:
                slots = random.choice(woz_db.service2db[service])
            service2slots[service] = slots

    service2preference = {
        service: {k: slot.get(k, '') for k in config.service2preference[service]}
        for service, slot in service2slots.items() if service in config.service2preference
        if service != 'taxi'
    }
    for service, preference in service2preference.items():
        if random.random() >= 0.1:
            continue
        config_slot_keys = config.rollback_service2prefierence[service]
        update_slot_keys = [
            slot_key for slot_key in preference if slot_key in config_slot_keys
        ]
        if not update_slot_keys:
            continue
        update_slot_key = random.choice(update_slot_keys)
        update_slot_val = random.choice([
            slot_val for slot_val in config.rollback_service2prefierence[service][update_slot_key]
            if slot_val != preference[update_slot_key]
        ])
        print(f'random update [{update_slot_key}] '
              f'from [{preference[update_slot_key]}] to [{update_slot_val}]', flush=True)
        preference[update_slot_key] = update_slot_val

    if 'train' in service2preference:
        while service2preference['train']['departure'] == service2preference['train']['destination']:
            service2preference['train']['destination'] = random.choice(list(config.service2schema_key2vals['train']['train-destination']))

        minutes = int(20 * random.random()) - 10
        minutes = datetime.timedelta(minutes=minutes)
        if random.random() < 0.5:
            leaveat = service2preference['train']['leaveat']
            leaveat = (datetime.datetime.strptime(leaveat, '%H:%M') + minutes).strftime('%H:%M')
            service2preference['train']['leaveat'] = leaveat
            del service2preference['train']['arriveby']
        else:
            arriveby = service2preference['train']['arriveby']
            arriveby = (datetime.datetime.strptime(arriveby, '%H:%M') + minutes).strftime('%H:%M')
            service2preference['train']['arriveby'] = arriveby
            del service2preference['train']['leaveat']
        service2preference['train']['bookpeople'] = '1' if random.random() < 0.1 else str(int(random.random() * 8) + 1)

    # 10% 的直推概率，首轮出现概率下降 10%
    name_prob_scale = 1.
    if 'restaurant' in service2preference:
        minutes = int(720 * random.random())
        minutes = datetime.timedelta(minutes=minutes)
        service2preference['restaurant']['booktime'] = (datetime.datetime.strptime('09:00', '%H:%M') + minutes).strftime('%H:%M')
        service2preference['restaurant']['bookday'] = random.choice(config.bookdays)
        service2preference['restaurant']['bookpeople'] = '1' if random.random() < 0.1 else str(int(random.random() * 8) + 1)

        if 'type' in service2preference['restaurant']:
            del service2preference['restaurant']['type']
        if 'name' in service2preference['restaurant'] and random.random() > (0.16 * name_prob_scale):
            # 78 -> 12
            del service2preference['restaurant']['name']

    if 'hotel' in service2preference:
        service2preference['hotel']['bookday'] = random.choice(config.bookdays)
        service2preference['hotel']['bookstay'] = str(int(random.random() * 8) + 1)
        service2preference['hotel']['bookpeople'] = '1' if random.random() < 0.1 else str(int(random.random() * 8) + 1)
        if 'name' in service2preference['hotel'] and random.random() > (0.25 * name_prob_scale):
            # 72 -> 18
            del service2preference['hotel']['name']

    if 'attraction' in service2preference:
        if 'name' in service2preference['attraction'] and random.random() > (0.3 * name_prob_scale):
            # 72 -> 18
            del service2preference['attraction']['name']

    for slot_key, ratio in {'bookday': 0.4, 'bookpeople': 0.6}.items():
        if random.random() > ratio:
            continue
        values = [preference[slot_key] for preference in service2preference.values() if slot_key in preference]
        if slot_key == 'bookday' and 'train' in service2preference and 'day' in service2preference['train']:
            values.append(service2preference['train']['day'])

        if len(values) <= 1:
            continue
        value = random.choice(values)
        for _service, _preference in service2preference.items():
            if slot_key in _preference:
                _preference[slot_key] = value
            if slot_key == 'bookday' and 'train' == _service and 'day' in _preference:
                _preference['day'] = value

    if 'taxi' in services:
        preference = {}
        preference['departure'] = random.choice(config.taxi_departures)
        preference['destination'] = random.choice(config.taxi_destinations)
        while preference['departure'] == preference['destination']:
            preference['destination'] = random.choice(config.taxi_destinations)

        # 通常 leave at 和 arrive by 只设置一个即可
        key = 'leaveat' if random.random() > 0.5 else 'arriveby'
        preference[key] = random.choice(config.taxi_times)

        if 'leaveat' in preference and random.random() < 0.1:
            arrivebys = [x for x in config.taxi_times if x > preference['leaveat']]
            preference['arriveby'] = random.choice(arrivebys)

        if 'arriveby' in preference and random.random() < 0.1:
            leaveats = [x for x in config.taxi_times if x < preference['arriveby']]
            preference['leaveat'] = random.choice(leaveats)
        service2preference['taxi'] = preference

    print(f'user preference = {json.dumps(service2preference)}', flush=True)

    del_services = set()
    service2slot_key_drop_ratio = config.turn_service2slot_key_drop_ratio[-1]
    for service, preference in service2preference.items():
        if service == 'train' or service == 'taxi':
            continue
        del_slot_keys = set()
        for slot_key, slot_val in preference.items():
            if 'book' in slot_key:
                continue
            if (not slot_val
                  or slot_key not in service2slot_key_drop_ratio[service]
                  or random.random() <= service2slot_key_drop_ratio[service][slot_key]):
                del_slot_keys.add(slot_key)

        if len([x for x in del_slot_keys if 'book' not in x]) == len([x for x in preference if 'book' not in x]):
            del_services.add(service)
        else:
            for del_slot_key in del_slot_keys:
                del preference[del_slot_key]

        if 'name' in preference:
            del_slot_keys = [slot_key for slot_key in preference if slot_key != 'name' and 'book' not in slot_key]
            for del_slot_key in del_slot_keys:
                del preference[del_slot_key]

    for service in ['train', 'restaurant', 'hotel']:
        service2remove_booking_prob = {
            'train': 0.05,
            'hotel': 0.15,
            'restaurant': 0.15,
        }
        if random.random() < service2remove_booking_prob[service] and service in service2preference:
            del_slot_keys = [slot_key for slot_key in service2preference[service] if 'book' in slot_key]
            for del_slot_key in del_slot_keys:
                del service2preference[service][del_slot_key]

    for del_service in del_services:
        del service2preference[del_service]
    print(f'user preference (after delete) = {json.dumps(service2preference)}', flush=True)

    return service2preference


def update_taxi_slots(preference, service2preference_gen_latest):
    up_serivices = [service for service in ['hotel', 'train', 'restaurant', 'attraction'] if service in service2preference_gen_latest]
    if not up_serivices:
        return

    up_service = random.choice(up_serivices)
    if 'train' == up_service:
        if random.random() < 0.5:
            if 'train-departure' in service2preference_gen_latest['train']:
                preference['destination'] = service2preference_gen_latest['train']['train-departure']
            if 'train-leaveat' in service2preference_gen_latest['train']:
                preference['arriveby'] = service2preference_gen_latest['train']['train-leaveat']
        else:
            if 'train-destination' in service2preference_gen_latest['train']:
                preference['departure'] = service2preference_gen_latest['train']['train-destination']
            if 'train-arriveby' in service2preference_gen_latest['train']:
                preference['leaveat'] = service2preference_gen_latest['train']['train-arriveby']

    elif f'{up_service}-name' in service2preference_gen_latest[up_service]:
        name = service2preference_gen_latest[up_service][f'{up_service}-name']
        preference['departure' if random.random() < 0.5 else 'destination'] = name
        if (up_service == 'restaurant'
                and name == preference['destination']
                and 'restaurant-booktime' in service2preference_gen_latest['restaurant']):
            preference['arriveby'] = service2preference_gen_latest['restaurant']['restaurant-booktime']
            if 'leaveat' in preference:
                del preference['leaveat']

    print(f'[taxi] update taxi perference into {preference}', flush=True)


def prepare_asking_slot_keys_from_preference(service2preference):
    service2asking_slot_keys = collections.defaultdict(list)

    service2slot_key_drop_ratio = config.turn_service2slot_key_drop_ratio[-1]

    for service, preference_kvs in service2preference.items():
        if service not in service2slot_key_drop_ratio:
            continue

        del_slot_keys = set()
        for slot_key, slot_val in list(preference_kvs.items()):
            if slot_key not in service2slot_key_drop_ratio[service]:
                del_slot_keys.add(slot_key)
            if random.random() < service2slot_key_drop_ratio[service][slot_key]:
                del_slot_keys.add(slot_key)

        if len(del_slot_keys) > 2:
            del_slot_keys = set(list(del_slot_keys)[0:2])

        if service == 'train':
            if 'leaveat' in del_slot_keys and 'arriveby' in del_slot_keys:
                del_slot_keys.remove(random.choice(['arriveby', 'leaveat']))
            elif 'leaveat' not in del_slot_keys and 'arriveby' not in del_slot_keys and random.random() > 0.1:
                del_slot_keys.add(random.choice(['arriveby', 'leaveat']))

        elif service == 'taxi':
            del_slot_keys = []

        elif service in {'hotel', 'restaurant', 'attraction'}:
            del_slot_keys.add('name')

        service2asking_slot_keys[service] = [
            f'{service}-{slot_key}' for slot_key in preference_kvs.keys()
            if slot_key not in del_slot_keys
        ]
    print(f'system asking = {json.dumps(service2asking_slot_keys)}', flush=True)
    return service2asking_slot_keys


def set_dst_preference(turn, service, preference, preference_dst, asking_slots=[], need_booking_slot=False):
    def random_choice(preference, force_n=-1):
        service2slot_key_drop_ratio_base = config.turn_service2slot_key_drop_ratio[-1]
        service2slot_key_drop_ratio = {} if turn >= config.n_tt else config.turn_service2slot_key_drop_ratio[turn]

        if force_n > 0:
            preference = list(preference.items())
            if len(preference) <= 1:
                return preference
            random.shuffle(preference)
            output_preference = {}
            for kv in preference[:force_n]:
                output_preference[kv[0]] = kv[1]
            return output_preference

        elif not service2slot_key_drop_ratio:
            return copy.deepcopy(preference)

        else:
            slot_key_drop_ratio = service2slot_key_drop_ratio[service]
            output_preference = {}
            slot_key_with_prob = []

            for slot_key, slot_val in preference.items():
                if slot_key not in slot_key_drop_ratio:
                    continue
                turn_prob = 1 - slot_key_drop_ratio[slot_key]
                final_prob = 1 - service2slot_key_drop_ratio_base[service][slot_key]
                scale_prob = turn_prob / final_prob
                scale_drop_prob = 1. - scale_prob

                if 'leaveat' in slot_key or 'arriveby' in slot_key:
                    scale_prob = min(1., turn_prob * 1.75)
                    scale_drop_prob = 1 - scale_prob

                slot_key_with_prob.append([slot_key, scale_prob])
                if random.random() <= scale_drop_prob:
                    continue
                output_preference[slot_key] = slot_val

            if not output_preference:
                accu_prob = 0.
                for i in range(len(slot_key_with_prob)):
                    slot_key_with_prob[i][1] = accu_prob + slot_key_with_prob[i][1]
                    accu_prob = slot_key_with_prob[i][1]

                rand = random.random() * accu_prob
                for i in range(len(slot_key_with_prob)):
                    if rand <= slot_key_with_prob[i][1]:
                        slot_key = slot_key_with_prob[i][0]
                        output_preference[slot_key] = preference[slot_key]
                        break

            if not output_preference:
                output_preference = copy.deepcopy(preference)

            return output_preference

    if asking_slots:
        asking_slot_keys = []
        for asking_slot in asking_slots:
            parts = asking_slot.split('-')
            if len(parts) != 2 or parts[0] != service:
                continue
            slot_key = parts[1]
            if slot_key not in preference or slot_key in preference_dst:
                continue
            asking_slot_keys.append(slot_key)

        if asking_slot_keys:
            for slot_key in asking_slot_keys:
                preference_dst[slot_key] = preference[slot_key]
            return

    avaliable_preference = {k: v for k, v in preference.items() if 'book' not in k and k not in preference_dst}
    avaliable_preference_book = {k: v for k, v in preference.items() if 'book' in k and k not in preference_dst}

    if avaliable_preference:
        avaliable_preference = random_choice(avaliable_preference)
        preference_dst.update(avaliable_preference)
        print(f'[{service}] add new dst slots = {avaliable_preference}')

        if need_booking_slot:
            n = 1 if random.random() <= 0.5 else (2 if random.random() < 0.8 else 3)
            avaliable_preference_book = random_choice(avaliable_preference_book, n)
            preference_dst.update(avaliable_preference_book)

    elif avaliable_preference_book:
        avaliable_preference_book = random_choice(avaliable_preference_book)
        preference_dst.update(avaliable_preference_book)

    # 首轮有一定的空槽概率
    if turn == 0 and random.random() <= 0.1:
        preference_dst.clear()


def rollback_preference(service, preference, preference_dst, preference_gen, preference_gen_latest):
    if service not in config.rollback_service2prefierence:
        return {}

    config_slot_keys = config.rollback_service2prefierence[service]
    update_slot_keys = [
        slot_key.split('-')[-1] for slot_key in preference_gen_latest if slot_key.split('-')[-1] in config_slot_keys
    ]
    if not update_slot_keys:
        return {}

    for _ in range(10):
        preference_update = {slot_key.split('-')[-1]: slot_val for slot_key, slot_val in preference_gen_latest.items()}
        update_slot_key = random.choice(update_slot_keys)
        update_slot_val = random.choice([
            slot_val for slot_val in config.rollback_service2prefierence[service][update_slot_key]
            if slot_val != preference_update[update_slot_key]
        ])
        if not update_slot_val:
            continue
        preference_update[update_slot_key] = update_slot_val
        api_config = {
            'service': service,
            'active_intent': f'find_{service}',
            'slot_values': {slot_key: [slot_val] for slot_key, slot_val in preference_update.items()}
        }
        search_results = woz_db.search(**api_config)
        print(json.dumps(api_config), len(search_results))
        if len(search_results) > 0:
            preference[update_slot_key] = update_slot_val
            preference_dst[update_slot_key] = update_slot_val
            print(f'[{service}] update [{update_slot_key}] to from [{preference_dst[update_slot_key]}] '
                  f'to [{update_slot_val}]， with search result = {len(search_results)}')
            return {update_slot_key: update_slot_val}
    return {}


def update_dialog_status(service, dialog_status, preference_dst, preference_gen, preference_gen_latest, service_booking):
    slots = {slot_key: slot_val.lower() for slot_key, slot_val in dialog_status.get('slots', {}).items()}
    print(f'[{service}] dialog status: dst = {json.dumps(preference_dst, sort_keys=True)}', flush=True)
    print(f'[{service}] dialog status: predict = {json.dumps(slots, sort_keys=True)}', flush=True)

    slots_update = {}
    for slot_key in list(slots.keys()):
        slot_val = slots[slot_key]
        slot_key = slot_key.split('-')[-1]
        if service not in config.replace_services or slot_key not in config.replace_keys:
            continue
        to_slot_key = config.replace_keys[slot_key]
        if fuzz.partial_ratio(slot_val, preference_dst.get(to_slot_key, '')) >= 80:
            del slots[f'{service}-{slot_key}']
            slots_update[f'{service}-{to_slot_key}'] = slot_val
            print(f'[{service}] update preddict slot, from [{slot_key}] to [{to_slot_key}], value = {slot_val}')
    slots.update(slots_update)

    for slot_key, slot_val in slots.items():
        if (slot_key not in config.service2schema_keys[service]
                or ('name' not in slot_key
                    and slot_key in config.service2schema_key2vals[service]
                    and slot_val.lower() not in config.service2schema_key2vals[service][slot_key])):
            print(f'[{service}] remove preddict user slot [{slot_key}] = [{slot_val}], value is invalid.', flush=True)
            continue

        if 'name' in slot_key and 'name' not in preference_dst:
            print(f'[{service}] remove preddict user slot [{slot_key}] = [{slot_val}], value is not in user DST.', flush=True)
            continue

        if 'name' in slot_key and 'name' in preference_dst and slot_val != preference_dst['name']:
            print(f'[{service}] update preddict user slot [{slot_key}], from [{slot_val}] to [{preference_dst["name"]}]')
            slot_val = preference_dst['name']

        if not service_booking and 'name' in slot_key and 'name' not in preference_dst:
            print(f'[{service}] remove preddict user slot [{slot_key}] = [{preference_dst["name"]}], for no booking')
            continue

        preference_gen[slot_key].add(slot_val)
        preference_gen_latest[slot_key] = slot_val

    if service == 'hotel' and 'type' in preference_dst and 'hotel-type' not in preference_gen:
        preference_gen['hotel-type'].add(preference_dst['type'])
        preference_gen_latest['hotel-type'] = preference_dst['type']

    del_slot_keys = []
    for slot_key, slot_val in preference_gen_latest.items():
        slot_key = slot_key.split('-')[-1]

        if service in {'taxi', 'train'}:
            if 'trainid' not in slot_key and slot_key not in preference_dst and f'{service}-{slot_key}' in preference_gen_latest:
                del_slot_keys.append(f'{service}-{slot_key}')
                print(f'[{service}] slot update, remove [{slot_key}]', flush=True)

        checking_slots = {
            'type', 'area', 'stars',
            'internet', 'parking', 'pricerange',
            'departure', 'destination', 'leaveat',
            'bookday', 'bookstay', 'bookpeople', 'booktime'
        }
        if slot_key not in checking_slots:
            continue

        if slot_key not in preference_dst:
            del_slot_keys.append(f'{service}-{slot_key}')
            print(f'[{service}] slot update, remove [{slot_key}].')

        elif preference_dst[slot_key].lower() != slot_val.lower():
            print(f'[{service}] slot update, update [{slot_key}] with pre-setting value [{slot_val}] to [({preference_dst[slot_key]})]')
            preference_gen_latest[f'{service}-{slot_key}'] = preference_dst[slot_key]

    for del_slot_key in del_slot_keys:
        if del_slot_key in preference_gen_latest:
            del preference_gen_latest[del_slot_key]

    if preference_dst.get('stars', '') == '0' and 'hotel-stars' not in preference_gen_latest:
        preference_gen_latest['hotel-stars'] = '0'
        print(f'[{service}] slot insert [hotel-stars] = 0 to dst')

    print(f'[{service}] dialog status (after user): slots   = {json.dumps(preference_gen_latest, sort_keys=True)}', flush=True)


def update_dialog_status_whole(service, services, simulator, history, search_results,
                               preference_gen, preference_gen_latest, service_booking,
                               extra_slots_for_search):
    if service not in ['hotel', 'attraction', 'restaurant']:
        return
    names = [x['name'] for x in search_results]

    dialog_status = simulator(
        service=service,
        service_schemas={service: config.service2schema[service] for service in services},
        history=history,
        names=names,
        verbose=False
    )
    slot_key = f'{service}-name'
    name = dialog_status.get('name', '')
    name_norm = name.replace('the', '').replace('\'', '').strip()

    name2ratio = {
        n: max(
            fuzz.partial_ratio(name.lower(), n),
            fuzz.partial_ratio(name_norm.lower(), n)
        )
        for n in names
    }
    if not name2ratio:
        return

    max_matched_name, max_matched_ratio = max(name2ratio.items(), key=lambda x: x[1])
    if max_matched_ratio <= 75:
        return
    extra_slots_for_search[f'{service}-name'] = max_matched_name

    if not service_booking and service in ['hotel', 'restaurant', 'attraction']:
        return
    if slot_key not in preference_gen_latest or preference_gen_latest[slot_key] != max_matched_name:
        preference_gen[slot_key].add(max_matched_name)
        preference_gen_latest[slot_key] = max_matched_name
        print(f'[{service}] update [{service}-name] to [{max_matched_name}] after system response.')
    print(f'[{service}] dialog status (after system): slots   = {json.dumps(preference_gen_latest, sort_keys=True)}')


def is_preference_satifised(service, preference, preference_gen_latest):
    for slot_key, slot_val in preference.items():
        if ('book' not in slot_key
                and f'{service}-{slot_key}' not in preference_gen_latest):
            return False
    return True


def is_dialog_end(user_utterance, system_utterance):
    good_bye_words_in_system = [
        x for x in ['re welcome', 'oodbye', 'no problem', 'bye', 'glad', '[eod]'] if x in system_utterance.lower()
    ]
    good_bye_words_in_user = [
        x for x in ['thank', '[eof]'] if x in user_utterance.lower()
    ]
    return len(good_bye_words_in_user) > 0 and len(good_bye_words_in_system) > 0


def is_recommend_need(turns, user_utterance, service, service2preference, service2preference_gen_latest, search_results):
    if service == 'taxi':
        return False

    if len(search_results) <= 1:
        return False

    if ('recommend' in user_utterance or 'book one of' in user_utterance or '[RECOM]' in user_utterance) \
            and 'recommendation' not in user_utterance and len(turns) > 2:
        return True

    if 'an you help with' in user_utterance:
        return True

    if is_preference_satifised(
            service,
            service2preference[service],
            service2preference_gen_latest[service]
    ) and len(search_results) > 1:
        return True

    return False


def post_process_system_response(response):
    if '? If' in response:
        response = response.split('? If')[0] + '?'
    return response


def get_turn_user_asking_slot(service, preference_dst):
    if service == 'attraction':
        pre_set_keys = ["area", "type", "entrancefee", "openhours", "address", "phone", "postcode", ]
    elif service == 'restaurant':
        pre_set_keys = ["pricerange", "area", "food", "address", "phone", "postcode",]
    elif service == 'hotel':
        pre_set_keys = ["pricerange", "parking", "stars", "internet", "area", "address", "phone", "postcode"]
    else:
        pre_set_keys = []
    user_asking_slot_keys = [slot_key for slot_key in pre_set_keys if slot_key not in preference_dst]
    random.shuffle(user_asking_slot_keys)
    n = int(random.random() * 2) + 1
    return user_asking_slot_keys[0:n]


def clean_utterance(utterance):
    utterance = copy.deepcopy(utterance)
    for mark in ['[EOD]', '[EOF]', '[BOOKED]', '[RECOM]']:
        utterance = utterance.replace(mark, '')
        utterance = utterance.replace(mark.lower(), '')
    return utterance


def choice_template(service, preference_gen_latest):
    dst_keys = {slot_key.split('-')[-1] for slot_key in preference_gen_latest if 'name' not in slot_key}

    matched_templates = []
    for template, template_keys in config.service2templates[service].items():
        template_keys_matched = [key for key in template_keys if key in dst_keys]
        if min(len(dst_keys), len(template_keys)) == len(template_keys_matched) or not template_keys:
            matched_templates.append(template)
    if not matched_templates:
        return ""
    return random.choice(matched_templates)


def generate_dialog(services, service2preference, cache):
    service2preference = update_preference(service2preference)

    # for agent
    service2asking_slot_keys = prepare_asking_slot_keys_from_preference(service2preference)
    # for dst
    service2preference_dst = defaultdict(dict)
    service2preference_gen = defaultdict(dict)
    service2preference_gen_latest = defaultdict(dict)

    user_simulator = GPTUserSimulator()
    user_simulator_empty_preference = GPTEmptyPreferenceUserSimulator()
    user_response_rewrite_simulator = GPTUserUtterenceRewriteSimulator()
    user_update_preference_simulator = GPTUserUpdatePreferenceSimulator()

    dialog_state_simulator = GPTDialogStateSimulator()
    dialog_state_name_simulator = GPTDialogStateNameSimulator()

    action2simulator = {
        'asking': GPTSystemAskingResponseSimulator(),
        'chatting': GPTSystemChattingResponseSimulator(),
        'searching': GPTSystemSimulator(),
    }

    status, turns, turn_no = False, [], 1

    service2status = {}

    history = []
    user_history = []

    services = [x for x in services if x != 'taxi'] + [x for x in services if x == 'taxi']

    for service_idx, current_service in enumerate(services):
        service2preference_dst[current_service] = {}
        service2preference_gen[current_service] = defaultdict(set)
        service2preference_gen_latest[current_service] = {}
        service2status[current_service] = 'inform'

        preference = service2preference[current_service]
        preference_dst = service2preference_dst[current_service]
        preference_gen = service2preference_gen[current_service]
        preference_gen_latest = service2preference_gen_latest[current_service]
        system_asking_slot_keys = service2asking_slot_keys[current_service]
        system_asked_slot_keys = set()

        if 'taxi' == current_service:
            update_taxi_slots(preference, service2preference_gen_latest)

        set_dst_preference(0, current_service, preference, preference_dst)
        extra_slots_for_search = {}

        service_booking = True
        if (current_service in ['train', 'restaurant', 'hotel']
                and (random.random() < 0.05 or len([x for x in preference if 'book' in x]) == 0)):
            service_booking = False
        if current_service == 'attraction' and random.random() < 0.1:
            service_booking = False

        num_asking_turn = 1 if random.random() < 0.5 else 0
        if current_service == 'attraction':
            num_asking_turn += 1 if random.random() < 0.8 else 0
        print(f'[{current_service}] number asking turn = {num_asking_turn}, service booking = {service_booking}', flush=True)

        num_update_preference_turn = 1 if random.random() < 0.8 else 2
        search_result_size = -1

        turn_success = False
        for turn_idx in range(config.service2turn_num[current_service]):
            refuse_booking = not service_booking and search_result_size == 1

            if search_result_size == 1 and num_asking_turn > 0:
                num_asking_turn -= 1
                turn_user_asking_slot_keys = get_turn_user_asking_slot(current_service, preference_dst)
            else:
                turn_user_asking_slot_keys = []

            if not preference_dst:
                user_utterance = user_simulator_empty_preference(service=current_service)
                user_utterance = random.choice(user_utterance)

            elif search_result_size == 0:
                # update a preference to find another candidates
                update_slot_kv = {}
                if num_update_preference_turn > 0:
                    update_slot_kv = rollback_preference(
                        service=current_service,
                        preference=preference,
                        preference_dst=preference_dst,
                        preference_gen=preference_gen,
                        preference_gen_latest=preference_gen_latest
                    )
                    print(f'[{current_service}] update to new user preference = {json.dumps(update_slot_kv)}')
                    num_update_preference_turn -= 1
                if not update_slot_kv:
                    service2status[current_service] = 'booked'
                else:
                    service2status[current_service] = 'booking'

                user_utterance = user_update_preference_simulator(
                    service=current_service,
                    history=history,
                    preference=preference_dst,
                    preference_new=update_slot_kv
                )
            else:
                user_utterance = user_simulator(
                    service=current_service,
                    preference_dst=preference_dst,
                    preference_all=preference,
                    service_status=service2status.get(current_service, ''),
                    servcie2preference_dst=service2preference_dst,
                    service2preference_gen_latest=service2preference_gen_latest,
                    refuse_booking=refuse_booking,
                    history=history,
                    pre_service=services[service_idx - 1] if service_idx > 0 and turn_idx == 0 else '',
                    user_asking_slot_keys=turn_user_asking_slot_keys,
                    search_result_size=search_result_size,
                    verbose=True
                )

            if (not turn_user_asking_slot_keys
                    and current_service == 'attraction' and not service_booking
                    and service2status.get(current_service, '') == 'booked'):
                user_utterance = user_utterance.split('.')[-1]
                if 'thank' not in user_utterance.lower():
                    user_utterance = "Thanks for the help, that is all I need. [EOF]"

            user_utterance_list = user_response_rewrite_simulator(
                history=history, utterance=user_utterance, verbose=False
            )
            user_utterance_list.append(user_utterance)

            user_utterance_list = [x for x in user_utterance_list if 'heart' not in x]

            print(f"[{current_service}] user: {user_utterance}")
            print(f"[{current_service}] user: {json.dumps(user_utterance_list, indent=2)}")

            extra_marks = ' '.join([x for x in ['[EOF]', '[RECOM]'] if x in user_utterance])
            random.shuffle(user_utterance_list)
            user_utterance = user_utterance_list[0] + extra_marks
            print(f"[{current_service}] *** user ***: {user_utterance}")

            turns.append({
                "turn_id": str(turn_no),
                "speaker": "USER",
                "actions": [f"{current_service.capitalize()}-{service2status.get(current_service, 'inform').capitalize()}"],
                "utterance": clean_utterance(user_utterance),
                'utterance_list': user_utterance_list
            })
            history.append(clean_utterance(user_utterance))
            user_history.append(clean_utterance(user_utterance))

            for _ in range(3):
                dialog_status = dialog_state_simulator(
                    service=current_service,
                    service_schemas={service: config.service2schema[service] for service in services},
                    history=user_history,
                    verbose=False
                )
                if dialog_status:
                    break
                time.sleep(3)

            update_dialog_status(current_service, dialog_status, preference_dst, preference_gen, preference_gen_latest, service_booking)
            print('\n')

            api_config = {
                'service': current_service,
                'active_intent': f'find_{current_service}',
                'slot_values': {slot_key: [slot_val] for slot_key, slot_val in preference_gen_latest.items()}
            }
            api_config['slot_values'].update({k: [v] for k, v in extra_slots_for_search.items()})
            search_results = woz_db.search(**api_config)
            print(f'[{current_service}] system: search size = {len(search_results)}')

            kwargs = {
                'service': current_service,
                'history': history
            }
            if len(search_results) > 10:
                kwargs['search_results'] = search_results[0:10]
            search_result_size = len(search_results)

            asking_slots = []
            if len(search_results) > 3 or current_service == 'taxi':
                asking_slots = [
                    x for x in system_asking_slot_keys
                    if x not in preference_gen_latest and 'book' not in x and x not in system_asked_slot_keys
                ]

            if service2status.get(current_service, '') != 'booked' and (
                    len(search_results) == 1 or
                    is_preference_satifised(current_service, preference, preference_gen_latest)
            ):
                if current_service == 'taxi':
                    service2status[current_service] = 'booking'
                else:
                    service2status[current_service] = 'booking'

            system_service_status = service2status.get(current_service, 'inform')
            system_template_utterance = ""

            is_recommended_turn = False

            if ('[EOF]' in user_utterance or 'bye' in user_utterance) and service2status[current_service] == 'booked':
                action = 'chatting'

            elif asking_slots:
                n = int(random.random() * len(asking_slots)) + 1
                kwargs['asking_slots'] = asking_slots[0:n]
                dialog_status = {
                    'action': 'asking',
                    'slots': {k: '' for k in asking_slots},
                    'service': current_service,
                }
                print(f'[{current_service}] system: action(update) = {dialog_status}')
                action = 'asking'

            else:
                if is_recommend_need(turns, user_utterance, current_service,
                                     service2preference, service2preference_gen_latest, search_results):
                    system_service_status = 'recommend'

                if system_service_status == 'recommend' and '[RECOM]' in user_utterance:
                    is_recommended_turn = True
                    number = len(search_results)
                    search_results = [random.choice(search_results)]
                    template = choice_template(current_service, preference_gen_latest)

                    if current_service in {
                        'restaurant', 'hotel', 'attraction'
                    } and system_service_status == 'recommend' and random.random() < 0.9 and template:
                        print(f'[{current_service}] template = {template}')
                        params = copy.deepcopy(random.choice(search_results))
                        params['number'] = number

                        if current_service == 'hotel':
                            params['parking'] = 'parking' if params.get('parking') == 'yes' else 'no parking'
                            params['internet'] = 'free wifi' if params.get('internet') == 'yes' else 'no wifi'
                        if 'entrance fee' in params:
                            params['entrancefee'] = params['entrance fee']

                        system_template_utterance = template.format(**params)
                        if not service_booking and current_service in ['hotel', 'restaurant', 'attraction']:
                            pass
                        else:
                            preference_gen[f'{current_service}-name'].add(params['name'])
                            preference_gen_latest[f'{current_service}-name'] = params['name']
                            print(f'[{current_service}] template answer, update name = [{params["name"]}]', flush=True)
                        extra_slots_for_search[f'{current_service}-name'] = params['name']

                if '[eof]' in user_utterance.lower() and (
                        not service_booking or len([x for x in preference.keys() if 'book' in x]) == 0
                ) and current_service == 'train':
                    system_service_status = 'booking'

                kwargs['service_status'] = system_service_status
                kwargs['search_results'] = search_results
                kwargs['search_condition'] = {current_service: preference_gen_latest}
                print(f'[{current_service}] system: search size = {len(search_results)}')

                turns.append({
                    "turn_id": f'{str(turn_no)}::follow_by_user_select_api',
                    "speaker": "SYSTEM",
                    "actions": [f"{current_service.capitalize()}-{system_service_status.capitalize()}"],
                    "service": current_service,
                    "service-action": 'search',
                    "utterance": 'GenAPIConfig',
                    'reference': [{
                        'service': current_service,
                        'active_intent': f'find_{current_service}',
                        'slot_values': {slot_key: [slot_val] for slot_key, slot_val in preference_gen_latest.items()}
                    }]
                })
                turns.append({
                    "turn_id": f"{turn_no}:follow_by_user_call_api",
                    "speaker": "SYSTEM",
                    "actions": [f"{current_service.capitalize()}-{system_service_status.capitalize()}"],
                    "service": current_service,
                    "service-action": 'search',
                    "utterance": "DoAPICall",
                    "reference": search_results,
                })
                action = 'searching'

            if (current_service in ['train', 'restaurant', 'hotel'] and search_result_size == 1
                    and (len([x for x in preference if 'book' in x]) == 0 or refuse_booking)):
                kwargs['service_booking'] = False

            if not system_template_utterance:
                kwargs['verbose'] = False
                system_resonse = action2simulator[action](**kwargs)

                if not system_resonse or not system_resonse.get('response', ''):
                    status = False

                if 'asking_slots' in system_resonse and isinstance(system_resonse['asking_slots'], list):
                    system_asked_slot_keys.update(system_resonse['asking_slots'])

                system_utterance = post_process_system_response(system_resonse.get('response', ''))
            else:
                system_utterance = system_template_utterance

            extra_marks = ' '.join([x for x in ['[BOOKED]', '[EOD]'] if x in system_utterance])
            if '?' in system_utterance and '[EOD]' not in system_utterance:
                parts = system_utterance.split('?')
                if len(parts[0].split(' ')) >= 20:
                    system_utterance = parts[0] + '?'
                if '[BOOKED]' in system_utterance:
                    system_utterance += ' [BOOKED]'
            if 'If' in system_utterance and not system_utterance.startswith('If'):
                system_utterance = system_utterance.split('If')[0]
            if extra_marks:
                system_utterance += ' ' + extra_marks
            print(f'[{current_service}] *** system ***: {system_utterance}')

            turns.append({
                "turn_id": str(turn_no+1),
                "speaker": "SYSTEM",
                "service": current_service,
                "actions": [f"{current_service.capitalize()}-{system_service_status.capitalize()}"],
                "utterance": clean_utterance(system_utterance),
            })
            history.append(f'from local guide: {clean_utterance(system_utterance)}')

            update_dialog_status_whole(
                service=current_service,
                services=services,
                simulator=dialog_state_name_simulator,
                search_results=search_results,
                history=history,
                preference_gen=preference_gen,
                preference_gen_latest=preference_gen_latest,
                service_booking=service_booking,
                extra_slots_for_search=extra_slots_for_search
            )
            print('\n')

            if action == 'asking':
                asking_slots = system_resonse.get('asking_slots', [])
                turns[-1]['asking_slots'] = {current_service: asking_slots}
            elif action == 'chatting':
                turns[-1]['asking_slots'] = {}

            if extra_slots_for_search:
                api_config_copy = copy.deepcopy(api_config)
                api_config_copy['slot_values'].update({k: [v] for k, v in extra_slots_for_search.items()})
                search_result_size = len(woz_db.search(**api_config_copy))

            turns[-1]['reference'] = [{
                'service': current_service,
                'active_intent': f'find_{current_service}',
                'slot_values': {slot_key: [slot_val] for slot_key, slot_val in preference_gen_latest.items()}
            }]

            if len([x for x in ['booked', 'reference code', '[BOOKED]'] if x in system_utterance]) > 0:
                service2status[current_service] = 'booked'

                if (current_service in {'hotel', 'restaurant', 'train'}
                        and '?' in system_utterance and '[BOOKED]' in system_utterance
                        and len([x for x in ['booked', 'reference code'] if x in system_utterance]) == 0
                        and len([x for x in preference if 'book' in x]) > len([x for x in preference_gen_latest if 'book' in x])):
                    service2status[current_service] = 'booking'

            if (len([x for x in preference if 'book' in x]) == 0
                    and is_preference_satifised(current_service, preference, preference_gen_latest)
                    and service2status[current_service] == 'booking'):
                if current_service == 'attraction' and (len(search_results) > 1 or num_asking_turn > 0):
                    pass
                else:
                    service2status[current_service] = 'booked'

            if current_service == 'attraction' and not service_booking and is_recommended_turn:
                service2status[current_service] = 'booked'
                num_asking_turn = 0

            turn_no += 2

            if service2status[current_service] == 'booked' and service_idx < len(services) - 1:
                turn_success = True
                break

            if is_dialog_end(user_utterance, system_utterance):
                turn_success = True
                break

            set_dst_preference(
                turn_idx+1,
                current_service,
                preference,
                preference_dst,
                kwargs.get('asking_slots', []),
                service2status.get(current_service, '') == 'booking'
            )

        if not turn_success:
            return False

    if len(turns) <= 4:
        return False

    cache.write(json.dumps({
        'services': services,
        'turns': turns,
        'status': status,
        'preference': service2preference
    }) + '\n')
    return True


def run_batch(services_list, service2preference_list, output_file):
    import traceback
    cache = Cache(output_file)

    for services, service2preference in zip(services_list, service2preference_list):
        try:
            for _ in range(3):
                status = generate_dialog(services, service2preference, cache)
                if status:
                    break
        except Exception as e:
            print('\nException', flush=True)
            traceback.print_exc(file=sys.stdout)
            print('', flush=True)
