import datetime
import os.path
import sys
import traceback

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
        logging.info(f'random update [{update_slot_key}] from [{preference[update_slot_key]}] to [{update_slot_val}]')
        preference[update_slot_key] = update_slot_val

    logging.info(f'user preference (from template) = {json.dumps(service2preference)}')

    return service2preference


def prepare_preference(services):
    service2slots = {}
    for service in services:
        if service == 'taxi':
            continue
        service2slots[service] = random.choice(woz_db.service2db[service])

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
        logging.info(f'random update [{update_slot_key}] from [{preference[update_slot_key]}] to [{update_slot_val}]')
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
    #
    name_prob_scale = 1.
    if 'restaurant' in service2preference:
        minutes = int(720 * random.random())
        minutes = datetime.timedelta(minutes=minutes)
        service2preference['restaurant']['booktime'] = (datetime.datetime.strptime('09:00', '%H:%M') + minutes).strftime('%H:%M')
        service2preference['restaurant']['bookday'] = random.choice(config.bookdays)
        service2preference['restaurant']['bookpeople'] = '1' if random.random() < 0.1 else str(int(random.random() * 8) + 1)

        if 'type' in service2preference['restaurant']:
            del service2preference['restaurant']['type']

        # remove name (80%) to avoid directly asking information for a specific one from user agent
        if 'name' in service2preference['restaurant'] and random.random() > (0.2 * name_prob_scale):
            del service2preference['restaurant']['name']

    if 'hotel' in service2preference:
        service2preference['hotel']['bookday'] = random.choice(config.bookdays)
        service2preference['hotel']['bookstay'] = str(int(random.random() * 8) + 1)
        service2preference['hotel']['bookpeople'] = '1' if random.random() < 0.1 else str(int(random.random() * 8) + 1)
        # same with restaurant
        if 'name' in service2preference['hotel'] and random.random() > (0.2 * name_prob_scale):
            del service2preference['hotel']['name']

    if 'attraction' in service2preference:
        # same with restaurant
        if 'name' in service2preference['attraction'] and random.random() > (0.2 * name_prob_scale):
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

        # keep only one between `leaveat` and `arriveby`, taxi's time is freestyle
        key = 'leaveat' if random.random() > 0.5 else 'arriveby'
        preference[key] = random.choice(config.taxi_times)

        if 'leaveat' in preference and random.random() < 0.1:
            arrivebys = [x for x in config.taxi_times if x > preference['leaveat']]
            preference['arriveby'] = random.choice(arrivebys)

        if 'arriveby' in preference and random.random() < 0.1:
            leaveats = [x for x in config.taxi_times if x < preference['arriveby']]
            preference['leaveat'] = random.choice(leaveats)
        service2preference['taxi'] = preference

    logging.info(f'user preference = {json.dumps(service2preference)}')

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

    # remove booking slots with probality, the probablity is pre-configured
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
    logging.info(f'user preference (after delete) = {json.dumps(service2preference)}')

    return service2preference


def update_taxi_slots(preference, service2preference_gen_latest):
    up_serivices = [service for service in ['hotel', 'train', 'restaurant', 'attraction'] if service in service2preference_gen_latest]
    if not up_serivices:
        return

    up_service = random.choice(up_serivices)
    if 'train' == up_service:
        # train station will be the destination or arriveby for taxi
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
        # same with other services like, restaurant or hotel
        name = service2preference_gen_latest[up_service][f'{up_service}-name']
        preference['departure' if random.random() < 0.5 else 'destination'] = name
        if (up_service == 'restaurant'
                and name == preference['destination']
                and 'restaurant-booktime' in service2preference_gen_latest['restaurant']):
            preference['arriveby'] = service2preference_gen_latest['restaurant']['restaurant-booktime']
            if 'leaveat' in preference:
                del preference['leaveat']

    logging.info(f'[taxi] update taxi perference into {preference}')

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
    logging.info(f'system asking = {json.dumps(service2asking_slot_keys)}')
    return service2asking_slot_keys


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
        logging.info(f'api-config = {json.dumps(api_config)}, search results = {len(search_results)}')

        if len(search_results) > 0:
            logging.info(
                f'[{service}] update [{update_slot_key}] to from [{preference_dst[update_slot_key]}] '
                f'to [{update_slot_val}]， with search result = {len(search_results)}')
            preference[update_slot_key] = update_slot_val
            preference_dst[update_slot_key] = update_slot_val
            return {update_slot_key: update_slot_val}
    return {}

def is_dialog_end(user_utterance, system_utterance):
    good_bye_words_in_system = [
        x for x in ['re welcome', 'oodbye', 'no problem', 'bye', 'glad', '[eod]'] if x in system_utterance.lower()
    ]
    good_bye_words_in_user = [
        x for x in ['thank', '[eof]'] if x in user_utterance.lower()
    ]
    return len(good_bye_words_in_user) > 0 and len(good_bye_words_in_system) > 0

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


def random_choice_response_template(service, preference_gen_latest):
    dst_keys = {slot_key.split('-')[-1] for slot_key in preference_gen_latest if 'name' not in slot_key}

    matched_templates = []
    for template, template_keys in config.service2templates[service].items():
        template_keys_matched = [key for key in template_keys if key in dst_keys]
        if min(len(dst_keys), len(template_keys)) == len(template_keys_matched) or not template_keys:
            matched_templates.append(template)
    if not matched_templates:
        return ""
    return random.choice(matched_templates)


class GenerationContext:
    def __init__(self, services, service2preference):
        self.service2preference = update_preference(service2preference)

        # for agent
        self.service2asking_slot_keys = prepare_asking_slot_keys_from_preference(service2preference)
        # for dst
        # user profile dialog states
        self.service2preference_dst = defaultdict(dict)
        # dialog states that are generated by the agents
        self.service2preference_gen = defaultdict(dict)
        # the latest dialog states that are generated by the agents, most values are same with self.service2preference_gen
        self.service2preference_gen_latest = defaultdict(dict)

        # here we define the main user simulator and other user's assistency simulators to increase diversity
        self.user_simulator = GPTUserSimulator()
        self.user_simulator_empty_preference = GPTEmptyPreferenceUserSimulator()
        # rewrite user's utterance to increase diversity for user's utterances
        self.user_response_rewrite_simulator = GPTUserUtterenceRewriteSimulator()
        # user will update his/her preference if the agent can not find appropriate choice
        self.user_update_preference_simulator = GPTUserUpdatePreferenceSimulator()

        # simulator to verify the dialog state
        self.dialog_state_simulator = GPTDialogStateSimulator()
        # simulator to verify the names in dialog state which are always rewrite by GPT
        self.dialog_state_name_simulator = GPTDialogStateNameSimulator()

        # agent simulators for different dialog status
        self.action2simulator = {
            'asking': GPTSystemAskingResponseSimulator(),
            'chatting': GPTSystemChattingResponseSimulator(),
            'searching': GPTSystemSimulator(),
        }

        self.status = False
        self.turns = []
        self.turn_no = 1

        self.service2status = {}

        self.history = []
        self.user_history = []

        self.services = [x for x in services if x != 'taxi'] + [x for x in services if x == 'taxi']
        self.services_history = []

    def init_service(self, current_service):
        self.current_service = current_service
        self.service2preference_dst[current_service] = {}
        self.service2preference_gen[current_service] = defaultdict(set)
        self.service2preference_gen_latest[current_service] = {}
        self.service2status[current_service] = 'inform'

        self.dialog_status = {}

        self.preference = self.service2preference[current_service]
        self.preference_dst = self.service2preference_dst[current_service]
        self.preference_gen = self.service2preference_gen[current_service]
        self.preference_gen_latest = self.service2preference_gen_latest[current_service]
        self.system_asking_slot_keys = self.service2asking_slot_keys[current_service]
        self.system_asked_slot_keys = set()
        self.extra_slots_for_search = {}
        # whether the service needs booking
        self.service_booking = True
        if (current_service in ['train', 'restaurant', 'hotel']
                and (random.random() < 0.05 or len([x for x in self.preference if 'book' in x]) == 0)):
            self.service_booking = False
        if current_service == 'attraction' and random.random() < 0.1:
            self.service_booking = False

        self.num_asking_turn = 1 if random.random() < 0.5 else 0
        if current_service == 'attraction':
            self.num_asking_turn += 1 if random.random() < 0.8 else 0
        logging.info(f'[{current_service}] number asking turn = {self.num_asking_turn}, service booking = {self.service_booking}')

        if 'taxi' == current_service:
            update_taxi_slots(self.preference, self.service2preference_gen_latest)

        self.update_user_preference(0, [])

        self.num_update_preference_turn = 1 if random.random() < 0.8 else 2
        self.search_result_size = -1
        self.services_history.append(current_service)
        self.turn_success = False

    def get_user_asking_slots(self, search_result_size):
        if search_result_size == 1 and self.num_asking_turn > 0:
            self.num_asking_turn -= 1
            turn_user_asking_slot_keys = get_turn_user_asking_slot(self.current_service, self.preference_dst)
        else:
            turn_user_asking_slot_keys = []
        return turn_user_asking_slot_keys

    def get_system_asking_slots(self, search_results):
        # if the search result is greator than 3, the system agent will try to ask more preference from the user
        # it should be noted that the taxi's search is special, the search result is taxi's propertities
        if len(search_results) > 3 or self.current_service == 'taxi':
            asking_slots = [
                x for x in self.system_asking_slot_keys
                if x not in self.preference_gen_latest and 'book' not in x and x not in self.system_asked_slot_keys
            ]
            return asking_slots
        return []

    def update_dialog_states_after_user_response(self):
        # identify dialog states from dialog with specified prompts
        for _ in range(3):
            self.dialog_status = self.dialog_state_simulator(
                service=self.current_service,
                service_schemas={service: config.service2schema[service] for service in self.services},
                history=self.user_history,
                verbose=False
            )
            if self.dialog_status:
                break

        slots = {slot_key: slot_val.lower() for slot_key, slot_val in self.dialog_status.get('slots', {}).items()}
        logging.info(f'[{self.current_service}] dialog status: dst = {json.dumps(self.preference_dst, sort_keys=True)}')
        logging.info(f'[{self.current_service}] dialog status: predict = {json.dumps(slots, sort_keys=True)}')

        slots_update = {}
        for slot_key in list(slots.keys()):
            slot_val = slots[slot_key]
            slot_key = slot_key.split('-')[-1]
            if self.current_service not in config.replace_services or slot_key not in config.replace_keys:
                continue
            to_slot_key = config.replace_keys[slot_key]
            if fuzz.partial_ratio(slot_val, self.preference_dst.get(to_slot_key, '')) >= 80:
                del slots[f'{self.current_service}-{slot_key}']
                slots_update[f'{self.current_service}-{to_slot_key}'] = slot_val
                logging.info(f'[{self.current_service}] update preddict slot, from [{slot_key}] to [{to_slot_key}], value = {slot_val}')
        slots.update(slots_update)

        for slot_key, slot_val in slots.items():
            if (slot_key not in config.service2schema_keys[self.current_service]
                    or ('name' not in slot_key
                        and slot_key in config.service2schema_key2vals[self.current_service]
                        and slot_val.lower() not in config.service2schema_key2vals[self.current_service][slot_key])):
                logging.info(f'[{self.current_service}] remove preddict user slot [{slot_key}] = [{slot_val}], value is invalid.')
                continue

            if 'name' in slot_key and 'name' not in self.preference_dst:
                logging.info(f'[{self.current_service}] remove preddict user slot [{slot_key}] = [{slot_val}], value is not in user DST.')
                continue

            if 'name' in slot_key and 'name' in self.preference_dst and slot_val != self.preference_dst['name']:
                logging.info(
                    f'[{self.current_service}] update preddict user slot [{slot_key}], from [{slot_val}] to [{self.preference_dst["name"]}]')
                slot_val = self.preference_dst['name']

            if not self.service_booking and 'name' in slot_key and 'name' not in self.preference_dst:
                logging.info(f'[{self.current_service}] remove preddict user slot [{slot_key}] = [{self.preference_dst["name"]}], for no booking')
                continue

            self.preference_gen[slot_key].add(slot_val)
            self.preference_gen_latest[slot_key] = slot_val

        if self.current_service == 'hotel' and 'type' in self.preference_dst and 'hotel-type' not in self.preference_gen:
            self.preference_gen['hotel-type'].add(self.preference_dst['type'])
            self.preference_gen_latest['hotel-type'] = self.preference_dst['type']

        del_slot_keys = []
        for slot_key, slot_val in self.preference_gen_latest.items():
            slot_key = slot_key.split('-')[-1]

            if self.current_service in {'taxi', 'train'}:
                if 'trainid' not in slot_key and slot_key not in self.preference_dst and f'{self.current_service}-{slot_key}' in self.preference_gen_latest:
                    del_slot_keys.append(f'{self.current_service}-{slot_key}')
                    logging.info(f'[{self.current_service}] slot update, remove [{slot_key}]')

            checking_slots = {
                'type', 'area', 'stars',
                'internet', 'parking', 'pricerange',
                'departure', 'destination', 'leaveat',
                'bookday', 'bookstay', 'bookpeople', 'booktime'
            }
            if slot_key not in checking_slots:
                continue

            if slot_key not in self.preference_dst:
                del_slot_keys.append(f'{self.current_service}-{slot_key}')
                logging.info(f'[{self.current_service}] slot update, remove [{slot_key}].')

            elif self.preference_dst[slot_key].lower() != slot_val.lower():
                logging.info(
                    f'[{self.current_service}] slot update, update [{slot_key}] with pre-setting value [{slot_val}] to [({self.preference_dst[slot_key]})]')
                self.preference_gen_latest[f'{self.current_service}-{slot_key}'] = self.preference_dst[slot_key]

        for del_slot_key in del_slot_keys:
            if del_slot_key in self.preference_gen_latest:
                del self.preference_gen_latest[del_slot_key]

        if self.preference_dst.get('stars', '') == '0' and 'hotel-stars' not in self.preference_gen_latest:
            self.preference_gen_latest['hotel-stars'] = '0'
            logging.info(f'[{self.current_service}] slot insert [hotel-stars] = 0 to dst')

        logging.info(f'[{self.current_service}] dialog status (after user): slots   = {json.dumps(self.preference_gen_latest, sort_keys=True)}')

    def update_dialog_states_after_system_response(self, search_results):
        if self.current_service not in ['hotel', 'attraction', 'restaurant']:
            return
        names = [x['name'] for x in search_results]

        dialog_status = self.dialog_state_name_simulator(
            service=self.current_service,
            service_schemas={service: config.service2schema[service] for service in self.services},
            history=self.history,
            names=names, verbose=False
        )
        slot_key = f'{self.current_service}-name'
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
        self.extra_slots_for_search[f'{self.current_service}-name'] = max_matched_name

        if not self.service_booking and self.current_service in ['hotel', 'restaurant', 'attraction']:
            return
        if slot_key not in self.preference_gen_latest or self.preference_gen_latest[slot_key] != max_matched_name:
            self.preference_gen[slot_key].add(max_matched_name)
            self.preference_gen_latest[slot_key] = max_matched_name
            logging.info(f'[{self.current_service}] update [{self.current_service}-name] to [{max_matched_name}] after system response.')
        logging.info(f'[{self.current_service}] dialog status (after system): slots   = {json.dumps(self.preference_gen_latest, sort_keys=True)}')

    def update_service_status(self, user_utterance, system_utterance, system_action, system_asking_slots, system_recom_turn, api_config, search_results):
        if system_action == 'asking':
            self.turns[-1]['asking_slots'] = {self.current_service: system_asking_slots}
        elif system_action == 'chatting':
            self.turns[-1]['asking_slots'] = {}

        self.turns[-1]['reference'] = [{
            'service': self.current_service,
            'active_intent': f'find_{self.current_service}',
            'slot_values': {slot_key: [slot_val] for slot_key, slot_val in self.preference_gen_latest.items()}
        }]

        if len([x for x in ['booked', 'reference code', '[BOOKED]'] if x in system_utterance]) > 0:
            self.service2status[self.current_service] = 'booked'

            if (self.current_service in {'hotel', 'restaurant', 'train'}
                    and '?' in system_utterance and '[BOOKED]' in system_utterance
                    and len([x for x in ['booked', 'reference code'] if x in system_utterance]) == 0
                    and len([x for x in self.preference if 'book' in x]) > len(
                        [x for x in self.preference_gen_latest if 'book' in x])):
                self.service2status[self.current_service] = 'booking'

        if (len([x for x in self.preference if 'book' in x]) == 0
                and self.is_preference_satifised()
                and self.service2status[self.current_service] == 'booking'):
            if self.current_service == 'attraction' and (len(search_results) > 1 or self.num_asking_turn > 0):
                pass
            else:
                self.service2status[self.current_service] = 'booked'

        if self.current_service == 'attraction' and not self.service_booking and system_recom_turn:
            self.service2status[self.current_service] = 'booked'
            self.num_asking_turn = 0

        self.turn_no += 2

        if self.service2status[self.current_service] == 'booked' and len(self.services_history) < len(self.services):
            self.turn_success = True

        if is_dialog_end(user_utterance, system_utterance):
            self.turn_success = True

    def is_preference_satifised(self):
        for slot_key, slot_val in self.preference.items():
            if ('book' not in slot_key
                    and f'{self.current_service}-{slot_key}' not in self.preference_gen_latest):
                return False
        return True

    def is_recommend_need(self, user_utterance, search_results):
        if self.current_service == 'taxi':
            return False

        if len(search_results) <= 1:
            return False

        if '[RECOM]' in user_utterance and len(self.turns) > 2:
            return True

        if 'an you help with' in user_utterance:
            return True

        if self.is_preference_satifised() and len(search_results) > 1:
            return True

        return False

    def update_user_preference(self, turn, system_asking_slots):
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
                slot_key_drop_ratio = service2slot_key_drop_ratio[self.current_service]
                output_preference = {}
                slot_key_with_prob = []

                for slot_key, slot_val in preference.items():
                    if slot_key not in slot_key_drop_ratio:
                        continue
                    turn_prob = 1 - slot_key_drop_ratio[slot_key]
                    final_prob = 1 - service2slot_key_drop_ratio_base[self.current_service][slot_key]
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

        if system_asking_slots:
            # slots asked from system must be update to dst for a proper generation
            system_asking_slot_keys = []
            for asking_slot in system_asking_slots:
                parts = asking_slot.split('-')
                if len(parts) != 2 or parts[0] != self.current_service:
                    continue
                slot_key = parts[1]
                if slot_key not in self.preference or slot_key in self.preference_dst:
                    continue
                system_asking_slot_keys.append(slot_key)

            if system_asking_slot_keys:
                for slot_key in system_asking_slot_keys:
                    self.preference_dst[slot_key] = self.preference[slot_key]
                return

        avaliable_preference = {k: v for k, v in self.preference.items() if 'book' not in k and k not in self.preference_dst}
        avaliable_preference_book = {k: v for k, v in self.preference.items() if 'book' in k and k not in self.preference_dst}

        if avaliable_preference:
            avaliable_preference = random_choice(avaliable_preference)
            self.preference_dst.update(avaliable_preference)
            logging.info(f'[{self.current_service}] add new dst slots = {avaliable_preference}')

            if self.service2status.get(self.current_service, '') == 'booking':
                n = 1 if random.random() <= 0.5 else (2 if random.random() < 0.8 else 3)
                avaliable_preference_book = random_choice(avaliable_preference_book, n)
                self.preference_dst.update(avaliable_preference_book)

        elif avaliable_preference_book:
            avaliable_preference_book = random_choice(avaliable_preference_book)
            self.preference_dst.update(avaliable_preference_book)

        # 首轮有一定的空槽概率
        if turn == 0 and random.random() <= 0.1:
            self.preference_dst.clear()

    def generate_user_utterance(self, turn_idx, search_results, refuse_booking, user_asking_slot_keys):
        if not self.preference_dst:
            # conversion start
            user_utterance = self.user_simulator_empty_preference(service=self.current_service)
            user_utterance = random.choice(user_utterance)

        elif turn_idx != 0 and (not search_results or len(search_results) == 0):
            # update a preference to find other candidates
            update_slot_kv = {}
            if self.num_update_preference_turn > 0:
                update_slot_kv = rollback_preference(
                    service=self.current_service,
                    preference=self.preference,
                    preference_dst=self.preference_dst,
                    preference_gen=self.preference_gen,
                    preference_gen_latest=self.preference_gen_latest
                )
                logging.info(f'[{self.current_service}] update to new user preference = {json.dumps(update_slot_kv)}')
                self.num_update_preference_turn -= 1

            if not update_slot_kv:
                # end the service if no slot is updated
                self.service2status[self.current_service] = 'booked'
            else:
                self.service2status[self.current_service] = 'booking'

            # use update preference prompts to generate utterance
            user_utterance = self.user_update_preference_simulator(
                service=self.current_service,
                history=self.history,
                preference=self.preference_dst,
                preference_new=update_slot_kv
            )
        else:
            user_utterance = self.user_simulator(
                service=self.current_service,
                preference_dst=self.preference_dst,
                preference_all=self.preference,
                service_status=self.service2status.get(self.current_service, ''),
                servcie2preference_dst=self.service2preference_dst,
                service2preference_gen_latest=self.service2preference_gen_latest,
                refuse_booking=refuse_booking,
                history=self.history,
                pre_service='' if len(self.services_history) <= 1 else self.services_history[-2],
                user_asking_slot_keys=user_asking_slot_keys,
                search_result_size=len(search_results),
                verbose=False
            )

        user_utterance_list = self.user_response_rewrite_simulator(
            history=self.history, utterance=user_utterance, verbose=False
        )
        user_utterance_list.append(user_utterance)

        user_utterance_list = [x for x in user_utterance_list if 'heart' not in x]

        logging.info(f"[{self.current_service}] user: {user_utterance}")
        logging.info(f"[{self.current_service}] user: {json.dumps(user_utterance_list, indent=2)}")

        extra_marks = ' '.join([x for x in ['[EOF]', '[RECOM]'] if x in user_utterance])
        random.shuffle(user_utterance_list)
        user_utterance = user_utterance_list[0] + extra_marks
        logging.info(f"[{self.current_service}] *** user ***: {user_utterance}")

        self.turns.append({
            "turn_id": str(self.turn_no),
            "speaker": "USER",
            "actions": [f"{self.current_service.capitalize()}-{self.service2status.get(self.current_service, 'inform').capitalize()}"],
            "utterance": clean_utterance(user_utterance),
            'utterance_list': user_utterance_list
        })
        self.history.append(clean_utterance(user_utterance))
        self.user_history.append(clean_utterance(user_utterance))

        return user_utterance

    def generate_system_utterance(self, user_utterance, search_results, asking_slots, refuse_booking):

        kwargs = {
            'service': self.current_service,
            'history': self.history
        }
        # to save tokens for extensive search results, we limit the max search result to 10
        if len(search_results) > 10:
            kwargs['search_results'] = search_results[0:10]
        # please noted the result size is the original search result but not the truncated one
        search_result_size = len(search_results)

        # get system asking slots, like hotel-stars, hotel-parkingm, etc.

        # force the service status into 'booking' if all the preference is satisfied of the search result is unique
        if (self.service2status.get(self.current_service, '') != 'booked'
                and (len(search_results) == 1 or self.is_preference_satifised())):
            self.service2status[self.current_service] = 'booking'

        system_service_status = self.service2status.get(self.current_service, 'inform')
        system_template_utterance = ""

        system_recom_turn = False

        if (('[EOF]' in user_utterance or 'bye' in user_utterance)
                and self.service2status[self.current_service] == 'booked'):
            # END OF DIALOG STATUS with booked dialog status, update the response purpose into chatting
            system_action = 'chatting'

        elif asking_slots:
            # if there are slots that the system is going to ask, like hotel-type, hotel-stars
            # update the response propose into asking
            n = int(random.random() * len(asking_slots)) + 1
            kwargs['asking_slots'] = asking_slots[0:n]
            dialog_status = {
                'action': 'asking',
                'slots': {k: '' for k in asking_slots},
                'service': self.current_service,
            }
            logging.info(f'[{self.current_service}] system: action(update) = {dialog_status}')
            system_action = 'asking'

        else:
            # or else the response purpose is to complate the dialog task
            if self.is_recommend_need(user_utterance, search_results):
                system_service_status = 'recommend'

            if system_service_status == 'recommend' and '[RECOM]' in user_utterance:
                system_recom_turn = True
                number = len(search_results)
                search_results = [random.choice(search_results)]
                template = random_choice_response_template(self.current_service, self.preference_gen_latest)

                if self.current_service in {
                    'restaurant', 'hotel', 'attraction'
                } and system_service_status == 'recommend' and random.random() < 0.9 and template:
                    logging.info(f'[{self.current_service}] template = {template}')
                    params = copy.deepcopy(random.choice(search_results))
                    params['number'] = number

                    if self.current_service == 'hotel':
                        params['parking'] = 'parking' if params.get('parking') == 'yes' else 'no parking'
                        params['internet'] = 'free wifi' if params.get('internet') == 'yes' else 'no wifi'
                    if 'entrance fee' in params:
                        params['entrancefee'] = params['entrance fee']

                    system_template_utterance = template.format(**params)
                    if not self.service_booking and self.current_service in ['hotel', 'restaurant', 'attraction']:
                        pass
                    else:
                        self.preference_gen[f'{self.current_service}-name'].add(params['name'])
                        self.preference_gen_latest[f'{self.current_service}-name'] = params['name']
                        logging.info(f'[{self.current_service}] template answer, update name = [{params["name"]}]')
                    self.extra_slots_for_search[f'{self.current_service}-name'] = params['name']

            if '[eof]' in user_utterance.lower() and (
                    not self.service_booking or len([x for x in self.preference.keys() if 'book' in x]) == 0
            ) and self.current_service == 'train':
                system_service_status = 'booking'

            kwargs['service_status'] = system_service_status
            kwargs['search_results'] = search_results
            kwargs['search_condition'] = {self.current_service: self.preference_gen_latest}
            logging.info(f'[{self.current_service}] system: search size = {len(search_results)}')

            self.turns.append({
                "turn_id": f'{str(self.turn_no)}::follow_by_user_select_api',
                "speaker": "SYSTEM",
                "actions": [f"{self.current_service.capitalize()}-{system_service_status.capitalize()}"],
                "service": self.current_service,
                "service-action": 'search',
                "utterance": 'GenAPIConfig',
                'reference': [{
                    'service': self.current_service,
                    'active_intent': f'find_{self.current_service}',
                    'slot_values': {slot_key: [slot_val] for slot_key, slot_val in self.preference_gen_latest.items()}
                }]
            })
            self.turns.append({
                "turn_id": f"{self.turn_no}:follow_by_user_call_api",
                "speaker": "SYSTEM",
                "actions": [f"{self.current_service.capitalize()}-{system_service_status.capitalize()}"],
                "service": self.current_service,
                "service-action": 'search',
                "utterance": "DoAPICall",
                "reference": search_results,
            })
            system_action = 'searching'

        if (self.current_service in ['train', 'restaurant', 'hotel'] and search_result_size == 1
                and (len([x for x in self.preference if 'book' in x]) == 0 or refuse_booking)):
            kwargs['service_booking'] = False

        if not system_template_utterance:
            kwargs['verbose'] = False
            system_resonse = self.action2simulator[system_action](**kwargs)

            if not system_resonse or not system_resonse.get('response', ''):
                status = False

            if 'asking_slots' in system_resonse and isinstance(system_resonse['asking_slots'], list):
                self.system_asked_slot_keys.update(system_resonse['asking_slots'])

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
        logging.info(f'[{self.current_service}] *** system ***: {system_utterance}')

        self.turns.append({
            "turn_id": str(self.turn_no + 1),
            "speaker": "SYSTEM",
            "service": self.current_service,
            "actions": [f"{self.current_service.capitalize()}-{system_service_status.capitalize()}"],
            "utterance": clean_utterance(system_utterance),
        })
        self.history.append(f'from local guide: {clean_utterance(system_utterance)}')
        return system_utterance, system_action, system_recom_turn


def generate_dialog(services, service2preference, cache):
    gen_ctx = GenerationContext(services, service2preference)

    for service_idx, current_service in enumerate(services):
        gen_ctx.init_service(current_service)

        search_results = []
        search_result_size = -1

        for turn_idx in range(config.service2turn_num[current_service]):
            refuse_booking = not gen_ctx.service_booking and search_result_size == 1

            # get information slots like opening hour, location, for asking
            user_asking_slot_keys = gen_ctx.get_user_asking_slots(search_result_size)

            user_utterance = gen_ctx.generate_user_utterance(
                turn_idx=turn_idx,
                search_results=search_results,
                refuse_booking=refuse_booking,
                user_asking_slot_keys=user_asking_slot_keys
            )

            gen_ctx.update_dialog_states_after_user_response()

            # search the API by dialog states
            api_config = {
                'service': current_service,
                'active_intent': f'find_{current_service}',
                'slot_values': {slot_key: [slot_val] for slot_key, slot_val in gen_ctx.preference_gen_latest.items()}
            }
            api_config['slot_values'].update({k: [v] for k, v in gen_ctx.extra_slots_for_search.items()})
            search_results = woz_db.search(**api_config)
            logging.info(f'[{current_service}] system: search size = {len(search_results)}')

            system_asking_slots = gen_ctx.get_system_asking_slots(search_results)
            (
                system_utterance, system_action, system_recom_turn
            ) = gen_ctx.generate_system_utterance(
                user_utterance=user_utterance,
                search_results=search_results,
                asking_slots=system_asking_slots,
                refuse_booking=refuse_booking
            )

            gen_ctx.update_dialog_states_after_system_response(search_results)

            # update service status after each round
            gen_ctx.update_service_status(
                user_utterance=user_utterance,
                system_utterance=system_utterance,
                system_action=system_action,
                system_recom_turn=system_recom_turn,
                system_asking_slots=system_asking_slots,
                api_config=api_config,
                search_results=search_results
            )
            # there will be some slots like restaurant name or attraction name be specified, but not included into the dst (multiwoz config?)
            if gen_ctx.extra_slots_for_search:
                api_config_copy = copy.deepcopy(api_config)
                api_config_copy['slot_values'].update({k: [v] for k, v in gen_ctx.extra_slots_for_search.items()})
                search_result_size = len(woz_db.search(**api_config_copy))

            gen_ctx.update_user_preference(turn_idx+1, system_asking_slots)

            if gen_ctx.turn_success:
                logging.info(f'history = {json.dumps(gen_ctx.history, indent=2)}')
                logging.info(f"generation for |{gen_ctx.current_service:*^30}| successed!")
                break

        if not gen_ctx.turn_success:
            return False

    if len(gen_ctx.turns) <= 4:
        return False

    cache.write(json.dumps({
        'services': services,
        'turns': gen_ctx.turns,
        'status': gen_ctx.status,
        'preference': service2preference
    }) + '\n')
    return True


def run_batch(cache, services_list, service2preference_list):

    for services, service2preference in zip(services_list, service2preference_list):
        try:
            for _ in range(3):
                status = generate_dialog(services, service2preference, cache)
                if status:
                    break
        except Exception as e:
            logging.info('\nException')
            traceback.print_exc(file=sys.stdout)
            logging.info('')

def run_random(cache):
    services = prepare_services()
    service2preference = prepare_preference(services)

    try:
        for _ in range(3):
            status = generate_dialog(services, service2preference, cache)
            if status:
                break
    except Exception as e:
        logging.info('\nException')
        traceback.print_exc(file=sys.stdout)
        logging.info('')