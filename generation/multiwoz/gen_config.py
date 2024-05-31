import collections
import os, json
import random


__dir__ = os.path.split(os.path.realpath(__file__))[0]



services = [
    'attraction', 'restaurant', 'hotel', 'train', 'taxi'
]

service2turn_num = {
    'hotel': 12,
    'attraction': 8,
    'restaurant': 8,
    'train': 8,
    'taxi': 8
}

service2preference = {
    'attraction': ['area', 'type', 'name'],
    'restaurant': ['area', 'food', 'pricerange', 'name'],
    'hotel': ['area', 'internet', 'parking', 'pricerange', 'stars', 'type', 'name'],
    'train': ['departure', 'destination', 'day', 'leaveat', 'arriveby'],
}

service2asking_slot_keys = {
    'attraction': ['area', 'type', 'name', 'entrance fee', 'openhours'],
    'restaurant': ['area', 'food', 'pricerange', 'type', 'address', 'phone', 'postcode'],
    'hotel': ['area', 'internet', 'parking', 'pricerange', 'stars', 'type', 'address', 'phone', 'postcode'],
    'train': ['departure', 'destination', 'day', 'arriveby', 'leaveat', 'price', 'trainid', 'duration'],
    'taxi': ['departure', 'destination', 'arriveby', 'leaveat'],
}

service2schema = {}
service2schema_keys = json.load(open(f'{__dir__}/configs/schema_used_keys.json'))
service2schema_key2vals = collections.defaultdict(dict)

for obj in json.load(open(f'{__dir__}/configs/schema.json')):
    service = obj['service_name']
    if service not in ['attraction', 'restaurant', 'hotel', 'train', 'taxi']:
        continue
    else:
        service2schema[service] = []
        for slot in obj['slots']:
            if slot['name'] not in service2schema_keys[service]:
                continue
            service2schema[service].append(slot)
            if slot.get('is_categorical', False):
                vals = slot.get('possible_values')
                service2schema_key2vals[service][slot['name']] = set(vals)
for service in ['hotel', 'restaurant', 'attraction']:
    datas = json.load(open(f'{__dir__}/db_datas/{service}_db.json'))
    names = {data['name'] for data in datas}
    service2schema_key2vals[service][f'{service}-name'] = names

service_combinations = []
for service_combination, count in json.load(open(f'{__dir__}/configs/distribution_service.json')).items():
    for _ in range(int(count)):
        service_combinations.append(service_combination)


turn_service2slot_key_drop_ratio = []
for service2info in json.load(open(f'{__dir__}/configs/distribution.json')):
    service2slot_key_drop_ratio = collections.defaultdict(dict)
    for service, info in service2info.items():
        slots = info['slots']
        for slot_key, slot_prob in slots.items():
            if slot_key == 'type' and service == 'hotel':
                slot_prob *= 1.
            service2slot_key_drop_ratio[service][slot_key] = 1 - slot_prob
    turn_service2slot_key_drop_ratio.append(service2slot_key_drop_ratio)
n_tt = 5 # 五轮以后可以释放全部槽位
# print(json.dumps(service2slot_key_drop_ratio, indent=2))

taxi_times = json.load(open(f'{__dir__}/configs/taxi.json'))['leaveat']
taxi_departures = json.load(open(f'{__dir__}/configs/taxi.json'))['departure']
taxi_destinations = json.load(open(f'{__dir__}/configs/taxi.json'))['destination']

rollback_service2prefierence = json.load(open(f'{__dir__}/configs/preference.json'))

bookdays = [
  "monday",
  "tuesday",
  "wednesday",
  "thursday",
  "friday",
  "saturday",
  "sunday"
]

replace_keys = {
    'leaveat': 'arriveby',
    'arriveby': 'leaveat',
    'departure': 'destination',
    'destination': 'departure',
}
replace_services = {'taxi', 'train'}


first_round_slot_keys2ratio = {}
for data in open(f'{__dir__}/configs/distribution_slots.csv'):
    parts = data.strip().split('\t')
    _, service, slots, _, ratio = parts
    if service not in first_round_slot_keys2ratio:
        first_round_slot_keys2ratio[service] = collections.defaultdict(float)
    first_round_slot_keys2ratio[service][slots] = float(ratio)

def sample_by_probablity(service):
    val2ratio = first_round_slot_keys2ratio[service]
    vals = []
    bounds = []
    for val, ratio in val2ratio.items():
        vals.append(val)
        if not bounds:
            bounds.append(ratio)
        else:
            bounds.append(bounds[-1] + ratio)
    ratio = random.random() * bounds[-1]
    for i in range(len(bounds)):
        if ratio <= bounds[i]:
            return vals[i]
    return random.choice(vals)


service2templates = collections.defaultdict(dict)
keys = [
    '{type}', '{area}', '{pricerange}', '{internet}', '{parking}', '{stars}', '{food}'
]
for service in ['hotel', 'restaurant', 'attraction']:
    for template in open(f'{__dir__}/configs/templates/{service}.txt'):
        template = template.strip()
        if not template:
            continue
        matched_keys = [
            key.replace('{', '').replace('}', '') for key in keys if key in template
        ]
        service2templates[service][template] = matched_keys
