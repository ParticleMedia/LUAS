import copy
import datetime
import json
import os.path
from typing import Dict

__dir__ = os.path.split(os.path.realpath(__file__))[0]


class DataBase:
    def __init__(self):
        self.service2db = {}
        self.load()

        self.n2n = {
            "nando's": "nandos",
            "allen bell": "allenbell",
            "guest house's": "guest house",
            "erania" :"michaelhouse cafe"
        }

    def load(self):
        input_files = [
            f'{__dir__}/db_datas/attraction_db.json',
            f'{__dir__}/db_datas/hospital_db.json',
            f'{__dir__}/db_datas/hotel_db.json',
            f'{__dir__}/db_datas/police_db.json',
            f'{__dir__}/db_datas/restaurant_db.json',
            f'{__dir__}/db_datas/train_db.json',
            f'{__dir__}/db_datas/taxi_db.json'
        ]
        for input_file in input_files:
            service = input_file.split('/')[-1].replace('_db.json', '')
            items = json.load(open(input_file))
            if service == 'taxi':
                self.service2db[service] = items
                continue
            ex_items = []
            for item in items:
                if 'location' in item and isinstance(item['location'], list):
                    item['location'] = [round(x, 2) for x in item['location']]

                if 'name' in item and "'" in item['name']:
                    ex_item = copy.deepcopy(item)
                    ex_item['name'] = ex_item['name'].replace("'", '')
                    ex_items.append(ex_item)
            self.service2db[service] = items + ex_items
        for item in self.service2db['train']:
            item['trainid'] = item['trainid'].lower()

        self.service2keys = {
            'train': ['arriveby', 'day', 'departure', 'destination', 'trainid', 'leaveat'],
            'attraction': ['address', 'area', 'pricerange', 'type', 'name'],
            'hotel': ['area', 'internet', 'parking', 'name', 'pricerange', 'stars', 'type'],
            'restaurant': ['area', 'food', 'name', 'pricerange', 'type'],
        }

    def get_keys(self, service):
        if service == 'taxi':
            return list(self.service2db[service].keys())
        else:
            return self.service2db[service][0].keys()

    def search(self, service='', active_intent='', slot_values: Dict[str, str]=None, return_keys=[]):
        db = self.service2db.get(service, {})
        if not db:
            return []
        if service == 'taxi':
            return db

        elif 'find_' in active_intent:
            # find service
            match_idxs = [i for i in range(len(db))]
            valid_keys = set(db[0].keys()) if service not in self.service2keys else self.service2keys[service]

            for key, vals in slot_values.items():
                if 'book' in key:
                    # 预订类的Slot不输出
                    continue
                if key.startswith(f'{service}-'):
                    key = key[len(f'{service}-'):]
                if key not in valid_keys:
                    continue
                vals = set([val.lower() for val in vals])
                next_match_idxs = [idx for idx in match_idxs if db[idx].get(key, '') in vals]
                if not next_match_idxs and key in ('leaveat', 'arriveby') and vals:
                    val = list(vals)[0]
                    try:
                        val = [v for v in val.split(' ') if ":" in v][0]
                        s_time = (datetime.datetime.strptime(val, '%H:%M') - datetime.timedelta(minutes=60)).strftime('%H:%M')
                        e_time = (datetime.datetime.strptime(val, '%H:%M') + datetime.timedelta(minutes=180)).strftime('%H:%M')

                        if e_time >= s_time:
                            next_match_idxs = [
                                idx for idx in match_idxs if e_time >= db[idx].get(key, '') >= s_time
                            ]
                        else:
                            next_match_idxs = [
                                idx for idx in match_idxs if (
                                    '23:59' >= db[idx].get(key, '') >= s_time or e_time >= db[idx].get(key, '') >= '00:01'
                                )
                            ]
                        # print(s_time, e_time, next_match_idxs, flush=True)
                    except:
                        next_match_idxs = match_idxs
                match_idxs = next_match_idxs

            if not match_idxs and f'{service}-name' in slot_values and slot_values[f'{service}-name']:
                # 如果没有搜索结果，尝试使用名字做fuzzy match
                tgt_name = slot_values[f'{service}-name'][0].lower().split(',')[0]
                tgt_name = self.n2n.get(tgt_name, tgt_name)
                tgt_words = set(tgt_name.split(' '))
                for idx in range(len(db)):
                    src_name = db[idx]['name'].lower()
                    src_words = src_name.split(' ')
                    if tgt_name.startswith(src_name) or src_name.startswith(tgt_name):
                        match_idxs.append([idx, min(len(src_name), len(tgt_name))])
                    elif len([sw for sw in src_words if sw in tgt_words]) > 0:
                        match_idxs.append([idx, len(' '.join([sw for sw in src_words if sw in tgt_words]))])
                    else:
                        continue
                if match_idxs:
                    match_idxs = [sorted(match_idxs, key=lambda x:x[1], reverse=True)[0][0]]
                print(f'matched by {service}-name [{tgt_name}], match {len(match_idxs)} results')

            items = [db[idx] for idx in match_idxs]
            return items
        elif 'book_' in active_intent:
            return []


if __name__ == '__main__':
    config = {'service': 'attraction', 'active_intent': 'find_attraction', 'slot_values': {'attraction-type': ['college'], 'attraction-name': ['De Luca Cucina and Bar']}}

    db = DataBase()
    print(json.dumps(db.search(**config)))
