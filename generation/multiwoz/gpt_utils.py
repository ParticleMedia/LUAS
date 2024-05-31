import collections
import copy
import json
import os
import random
import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', )
from generation.multiwoz.gpt_base import GPTBase


try:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
except:
    rank = 0
    world_size = 1


GEN_API_CONFIG = 'GenAPIConfig'
DO_API_CALL = 'DoAPICall'

SPEAKER_USER = 'USER'
SPEAKER_SYSTEM = 'SYSTEM'

service2slot_ask = {
    'attraction': ['attraction-area', 'attraction-type'],
    'restaurant': ['restaurant-area', 'restaurant-food', 'restaurant-pricerange'],
    'hotel': ['hotel-area', 'hotel-pricerange', 'hotel-stars', 'hotel-type'],
    'train': ['train-departure', 'train-destination', 'train-day'],
    'taxi': ['taxi-departure', 'taxi-destination'],
}

service2prompt = {
    'attraction': ['a place to go', 'a trip', 'local attractions'],
    'restaurant': ['a place to eat', 'a restaurant'],
    'hotel': ['some places to stay', 'a hotel'],
    'train': ['a train '],
    'taxi': ['a taxi ']
}

def is_preference_will_meet(preference_all, preference_dst):
    for slot_key, slot_val in preference_all.items():
        if ('book' not in slot_key
                and f'{slot_key}' not in preference_dst):
            return False
    return True

def get_same_slot_key_from_history(service, slot_key, slot_val, servcie2preference_gen_latest):
    slot_val = slot_val.lower()
    for history_service, history_preference_gen_latest in servcie2preference_gen_latest.items():
        history_preference_gen = {k.split('-')[-1]: v for k, v in history_preference_gen_latest.items()}

        if service == history_service:
            continue
        for history_slot_key, history_slot_val in history_preference_gen.items():
            # 这里大概以 10% 的概率替换？
            if history_slot_key == slot_key and history_slot_val == slot_val:
                if 'area' in slot_key:
                    return random.choice([
                        f"same {history_slot_key} with {history_service}",
                        f"near the {history_service}",
                    ])
                else:
                    return f"same {history_slot_key} with {history_service}"

        if service == 'taxi':
            if 'name' in history_preference_gen and slot_val == history_preference_gen['name'].lower():
                if slot_key == 'departure':
                    return f"the departure is the {history_service} you just selected"
                elif slot_key == 'destination':
                    return f"the destination is the {history_service} you just selected"

            if (history_service == 'train'
                    and slot_key == 'destination'
                    and history_preference_gen.get('departure', '').lower() == slot_val):
                return f"the destination is the departure train station of your travel"

            if (history_service == 'train'
                    and slot_key == 'departure'
                    and history_preference_gen.get('destination', '').lower() == slot_val):
                return f"the departure is the destination train station of your travel"
    return ""


class GPTEmptyPreferenceUserSimulator(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4-1106-preview')

    def prompting(self, service, **kwargs):
        print(f"[{service}] user simulation empty preference, service = {service}", flush=True)

        service2prompt = {
            'attraction': ['local attractions'],
            'restaurant': ['a place to eat', 'a restaurant'],
            'hotel': ['some places to stay', 'a hotel'],
            'train': ['a train to take'],
            'taxi': ['a taxi ']
        }
        prompt = (
            f"You are requesting a help from a local guide to find {service2prompt[service]}.\n"
            "Now you are chatting online with the local guide. \n"
            "Please generate a response to start the conversion.\n"
            "Please output the response only.\n"
            "Please respond briefly, each response should be no more than 15 words."
            'Please output 10 different responses in Json format like ["response0", "response1", ...]\n'
        )
        return prompt

    def parsing(self, res, **kwargs):
        res = res.replace('```json', '').replace('```', '')
        res = json.loads(res)
        return res

class GPTUserSimulator(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4-1106-preview')

    def prompting(self,
                  service,
                  preference_dst,
                  preference_all,
                  service_status,
                  servcie2preference_dst,
                  service2preference_gen_latest,
                  refuse_booking,
                  user_asking_slot_keys,
                  history,
                  pre_service,
                  search_result_size,
                  **kwargs):

        print(f"[{service}] user simulation, status = {service_status}, preference = {json.dumps(preference_dst)}", flush=True)

        service_templates = []
        if service == 'train':
            template = service2prompt[service][0]
            service_templates.append(template)
        else:
            service_templates.append(random.choice(service2prompt[service]))
        service_prompt = service_templates[-1]
        if len(service_templates) > 1:
            service_prompt = ', '.join(service_templates[0:-1]) + ' and ' + service_prompt

        preference_same_with_history = {}
        if random.random() < 0.2 or service == 'taxi':
            # 历史相同槽位替换
            preference_same_with_history = {
                f'{service}-{slot_key}': get_same_slot_key_from_history(service, slot_key, slot_val, service2preference_gen_latest)
                for slot_key, slot_val in preference_dst.items()
            }
            preference_same_with_history = {k: v for k, v in preference_same_with_history.items() if v}
        if preference_same_with_history:
            print(f'[{service}] same slot compared with history = {preference_same_with_history}', flush=True)

        # 当前槽位
        preference_formated = {
            f"{service}'s {slot_key}": slot_val for slot_key, slot_val in preference_dst.items()
            if f'{service}-{slot_key}' not in preference_same_with_history
        }
        preference_formated.update(preference_same_with_history)

        position = 'here' if service == 'train' or random.random() < 0.5 else 'cambridge'
        prompt = (
            f"This is your first time in {position} and want to find {service_prompt}.\n"
            f"Now you are chatting with a local guide online. \n"
            f"And this is your preference:\n"
            f"{json.dumps(preference_formated)}\n"
            f"and the conversation history (may be empty or not relevant to the current preference):\n"
            f"{json.dumps(history, indent=2)}\n"
            f"Your responses should resemble an online chat as much as possible, and make them as brief as you can.\n"
            f"How would you initiate the inquiry or respond to the guide online?\n"
            f"Please do not provide any information that is not exist in your preference.\n"
        )

        if not preference_formated:
            prompt += (
                "Please output a very simple response based on the service if the preference is empty, like:\n"
                "- Please help me to find a hotel in Cambridge, if the service is a hotel.\n"
                "Please don't copy the example directly, generate a new one by yourself.\n"
            )

        if preference_same_with_history:
            prompt += (
                "Please use values inside the 'preference' as much as possible, rather than using specific names."
            )

        # GPT 4 似乎不太能区分清楚 taxi arrive by 和 leave at 的区别
        if 'taxi-arriveby' in preference_formated:
            prompt += (
                "For `taxi` service, please note the `taxi-arriveby` is the time that you arrived at the destination, not your departure time.\n"
            )

        if not service:
            prompt += (
                "And now, all your preferences are meet.\n"
                "Please thank for the help from the local guide and say goodbye to him and output the mark `[EOF]`"
            )

        elif user_asking_slot_keys:
            print(f"[{service}] user simulation, status = {service_status}, asking slots = {user_asking_slot_keys}")
            prompt += (
                'Here is some information that you want to get from the local guide:\n'
                f'{json.dumps(user_asking_slot_keys)}\n'
                'Please read the history carefully and ask the information that is in your list but has not been mentioned in the history.\n'
                "Please ask a question for the information only, don't respond with other thing.\n"
                "Please try not to mention names in your questions as much as possible.\n"
            )

        elif service_status == 'inform':
            if random.random() < 0.5:
                # 50% 的概率使用同义词
                prompt += (
                    f"Please randomly use synonyms or synonymous phrases to describe your intention, for example:\n"
                    f"- you can use `something to eat` or some food` instead of `restaurant`."
                )
            prompt += (
                f"Please provide all the information in your preferences to the guide, except the ones that have been informed in the history.\n"
                f"Please remember do not provide any information that is not exist in your preference.\n"
                f"If the local guide asks your preference, answer it directly and don't answer with other words.\n"
                # f"Please don't provide any extra information that is not listed in your preference.\n"
                f"Please don't repeat asking the same thing that is in the history.\n"
                f"Please don't repeat your old preference which you have informed in the history when you respond to the guide.\n"
                f"Please make sure the time in your response must be in the format `after, at or around %H:%M` in 24 hours.\n"
                f"Pay attention to the diversity of responses, and try not to reuse sentence patterns that have been used in history.\n"
            )
            preference_meet = is_preference_will_meet(preference_all, preference_dst) and 'name' not in preference_dst

            directly_recom_ratio = 0.1

            if preference_meet and random.random() < directly_recom_ratio:
                prompt += (
                    f"In this round, all your preferences will be informed to the guide.\n"
                    f"After stating your needs, Please request the guide to recommend a result for you, and output a special mark `[RECOM]` at the end.\n"
                )
            if pre_service:
                prompt += (
                    f"You may need to thank the guide for providing information about {pre_service} in you response.\n"
                )

        elif service_status == 'booking':
            if (service in ['train', 'restaurant', 'hotel'] and search_result_size == 1
                    and (len([x for x in preference_all if 'book' in x]) == 0 or refuse_booking)):
                prompt += (
                    "And now, all your preference is meet, but you don't need a reservation at the moment.\n"
                    "Please say thanks for the help from local guide politely and output the mark `[EOF]`"
                )
                print(f"[{service}] user simulation, do not need a booking")

            elif search_result_size > 1:
                prompt += (
                    "There are several choices meet your preference.\n"
                    "If the agent doesn't recommend you a selection, \n"
                    "please ask directly for a recommendation from the local agent, "
                        "and output a special mark `[RECOM]` if you are looking for a recommendation.\n"
                )
            else:
                prompt += (
                    f"Please ask for a booking from the local guide with your booking preference.\n"
                    f"Please don't use today or other relative days to describe the `bookday`.\n"
                    f"If no booking is needed, please end the conversion directly.\n"
                    f"If the guide asks you for the booking information, please avoid providing the booking information only.\n"
                    f"Please don't put other references that are non-relevant to your booking, like price range, area or others.\n"
                    f"Please try not repeat the booking information that you have already informed in the history.\n"
                )

            if service == 'train':
                prompt += (
                    "The recommendation provided by the guide is the best choice, even if the time difference on leave at or arriveby is large, it depends on the train schedule.\n"
                    "Please accept the recommendation, and don't ask for a better time."
                )

        elif service_status == 'booked':
            if (service in ['train', 'restaurant', 'hotel'] and search_result_size == 1
                    and (len([x for x in preference_all if 'book' in x]) == 0 or refuse_booking)):
                prompt += (
                    "And now, all your preferences are met, but you don't need a reservation at the moment.\n"
                    "Please say thanks for the help from the local guide politely and output the mark `[EOF]`"
                )
                print(f"[{service}] user simulation, do not need a booking")
            else:
                prompt += (
                    "And now, all your preferences are met.\n"
                    f"Please always answer with **No** if the guide asks you whether more information is needed.\n"
                    "Please thanks for the help from the local guide and output the mark `[EOF]`"
                )

        prompt += (
            f"Only output the newest utterance, don't output the conversation history.\n"
            # f"Please output 5 different latest utterances in JSON format like: ['utterance0', ... ]\n"
        )
        return prompt

    def parsing(self, res, **kwargs):
        res = str(res)
        if res == 'None':
            res = '[EOF]'
        res = res.replace('```json', '').replace('```', '')
        if 'Tom:' in res:
            return res.replace('Tom:', '').strip()
        # return json.loads(res.strip())
        return res.strip()


class GPTUserUpdatePreferenceSimulator(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4-1106-preview')

    def prompting(self,
                  service,
                  history,
                  preference,
                  preference_new,
                  **kwargs):

        print(f"[{service}] user preference update simulation, preference = {json.dumps(preference_new)}", flush=True)

        service_templates = []
        if service == 'train':
            template = service2prompt[service][0]
            service_templates.append(template)
        else:
            service_templates.append(random.choice(service2prompt[service]))
        service_prompt = service_templates[-1]
        if len(service_templates) > 1:
            service_prompt = ', '.join(service_templates[0:-1]) + ' and ' + service_prompt

        preference_formated = {f'{service}-{slot_key}': slot_val for slot_key, slot_val in preference.items()}
        preference_formated_new = {f'{service}-{slot_key}': slot_val for slot_key, slot_val in preference_new.items()}
        for k, v in preference_formated.items():
            if k not in preference_formated_new:
                preference_formated_new[k] = v

        if preference_new:
            prompt = (
                f"You are the first time to Cambridge and want to find {service_prompt}.\n"
                f"And now you are chatting with a local guide online. \n"
                f"Here is your old preference:\n"
                f"{json.dumps(preference_formated)}\n"
                f"and here is your new perference:"
                f"{json.dumps(preference_formated_new)}\n"
                f"and the conversation history:\n"
                f"{json.dumps(history, indent=2)}\n"
                "Please output your response to inform the local guide for your preference change.\n"
                f"Your responses should resemble an online chat as much as possible, and make them as brief as you can.\n"
                f"Don't tell the guide you change your mind, please inform him like:\n"
                "- how about, would you like or do you have and ect.\n"
                f"Only output the newest utterance, don't output the conversation history.\n"
            )
        else:
            prompt = (
                f"You are the first time to here and want to find {service_prompt}.\n"
                f"And now you are chatting with a local guide online. \n"
                f"Here is your old preference:\n"
                f"{json.dumps(preference_formated)}\n"
                "Based on your preference, the local guide can not find an appropriate candidate.\n"
                "And now you don't want to change your preference either.\n"
                "Please thank for the help from the local guide and say goodbye to him and output the mark `[EOF]`"
            )
        return prompt

    def parsing(self, res, **kwargs):
        res = str(res)
        if res == 'None':
            res = '[EOF]'
        res = res.replace('```json', '').replace('```', '')
        if 'Tom:' in res:
            return res.replace('Tom:', '').strip()
        if '!' in res and len(res.split('!')[0].split(' ')) <= 3:
            # 处理这种 case ：Hey! I'm looking for 'Rosas Bed and Breakfast'. Can you guide me?
            res = '!'.join(res.split('!')[1:]).strip()
        return res.strip()

class GPTUserUtterenceRewriteSimulator(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4-1106-preview')

    def prompting(self,
                  history,
                  utterance,
                  **kwargs):

        parts = utterance.split(',')
        if len(parts[0].split(' ')) <= 2:
            utterance = ','.join(parts[1:]).strip().capitalize()

        return (
            "Here is a user response for a online conversion with a local guide:\n"
            f"{utterance}\n"
            "Please rewrite the user's reply in different tone, and ensure that the content of the reply does not change.\n"
            "The overall length of the rewritten reply should be as close as possible to the length of the user's reply.\n"
            f"Please output 3 different rewrited responses in JSON format like: ['response0', ... ]\n"
        )
    def parsing(self, res, **kwargs):
        res = res.replace('```json', '').replace('```', '')
        obj = json.loads(res)
        out_resp = []
        for resp in obj:
            if isinstance(resp, dict) and 'response' in resp:
                resp = resp['response']
            if not isinstance(resp, str):
                continue
            if '!' in resp and len(resp.split('!')[0].split(' ')) <= 3:
                # 处理这种 case ：Hey! I'm looking for 'Rosas Bed and Breakfast'. Can you guide me?
                resp = '!'.join(resp.split('!')[1:]).strip()
            if resp:
                out_resp.append(resp)
        return out_resp

class GPTDialogStateSimulator(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4-1106-preview')

    def prompting(self, service, service_schemas, history, **kwargs):
        return (
            f"You are a local agent, and now chatting with the user online for `{service}.\n"
            f"Here is the conversion history:\n{json.dumps(history)}\n"
            f"Here are the service schemas that you might use for all services: \n{json.dumps(service_schemas)}\n"
            f"Please read the history and the service schemas carefully: \n"
            f"- first find best service matched for the last utterance, \n"
            f"- then find the slots of {service} from the conversion history based on the schema of {service}.\n"
            'Your response should be in JSON format: {"slots": {"slot key": "slot value"}, "service": ""}, \n'
            "The service you selected must be in the schema.\n"
            "The slots in your output must be in the schema of your predicted `service`,\n"
            "- the `slot key` must be mentioned in the schema\n"
            "- the `slot value` should be mentioned in the schema `possible_values` if the slot value is categorical "
            "or you need to extract its value exactly from the conversion history.\n"
        )


class GPTDialogStateNameSimulator(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4-1106-preview')

    def prompting(self, service, service_schemas, history, names, **kwargs):
        schema = None
        for __service, __schemas in service_schemas.items():
            if __service != service:
                continue
            for __schema in __schemas:
                if __schema['name'] == f'{service}-name':
                    schema = __schema

        return (
            f"You are a local agent, and now chatting with the user online for `{service}`.\n"
            f"Here is the conversion history between the user and others:\n{json.dumps(history)}\n"
            f"Here are the schemas that you might use: \n{json.dumps(schema)}\n"
            f"Here are all the names that might be appeared in the history:\n{json.dumps(names)}\n"
            f"Please read the history and the service schema carefully: \n"
            f"- then find the best matched {service}-name from the conversion history based on the schema and names.\n"
            'The name you output must follow the schema and exist in the names that are provided.\n'
            f'Your response should be in JSON format: {{"name": "***"}} or and {{"name": ""}} if no {service}-name were found in the history.\n'
        )

class GPTSystemAskingResponseSimulator(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4-1106-preview')

    def prompting(self, service, history, asking_slots, **kwargs):
        print(f'[{service}] asking simulator, asking slots = {asking_slots}', flush=True)

        return (
            f"You are a local agent for `{service}`, and are chatting with the user online.\n"
            "You are going to rhetorical question some search criteria to make the user request more clearly.\n"
            f"Here is the conversion history:\n{json.dumps(history)}\n"
            f"and the rhetorical slots that you will ask: \n{json.dumps(asking_slots)}\n"
            "Please read the history and rhetorical slots carefully.\n"
            "Then generate a brief rhetorical response to continue the conversion.\n"
            "- the response should resemble an online chat as much as possible, and make them as brief as possible.\n"
            "- please ask by the rhetorical slots directly, don't respond with other words, don't tell the user that you are narrowing down the option.\n "
            "- please try asking all the rhetorical slots that are provided in the rhetorical slots at once.\n"
            "- for the service `train`, no return ticket is preferred from the user, "
                "and all the users will be adults as a group when booking tickets, "
                "but you need still to ask how many people instead.\n"
            f"Pay attention to the diversity of responses, and try not to reuse sentence patterns that have been used in history.\n"
            'Please answer in a JSON format, {"response": "", "asking_slots": []}\n'
        )


def get_system_in_domain_prompt(service):
    if service == 'train':
        return (
            "Please remember that no return ticket is perfered for the user.\n"
        )
    return ""


def get_system_recommend_prompt(service, **kwargs):
    prompt = (
        "If you have not inform the user of the result, please first inform the user of the result:\n"
        "- the information should have the number of candidates, please don't use the exact number, use many, several, some or others instead.\n"
        "- and it is also necessary to ask if the user needs a recommendation.\n"
        "If you have already informed the user of the result, and the user what a recommendation, please do the follows:\n"
    )
    service2prompt = {
        'hotel': (
            "Please recommend one candidate with the hotel name and detailed information from the search results directly, don't repeat the user need.\n"
            "The detailed information needed contains hotel area, stars, internet and parking available.\n"
        ),
        'train': (
            "Please recommend one candidate with the train ID with detailed information from the search results directly, don't repeat the user need.\n"
            "The detailed information needed contains departure, destination, leave and arrival time for the train.\n"
        ),
        'restaurant': (
            "Please recommend one candidate with the restaurant name and detailed information from the search results directly, don't repeat the user need.\n"
            "The detailed information needed contains restaurant area, pricerange, food type.\n"
        ),
        'attraction': (
            "Please recommend one candidate with the attraction name from the search results directly, don't repeat the user need.\n"
            "Please don't provide the opening hours unless the user asks it.\n"
        ),
    }
    prompt += service2prompt.get(service, '')
    prompt += (
        "- please don't output the canddiates details.\n"
    )
    return prompt

def get_system_booking_prompt(service, **kwargs):
    prompt = ""
    if service == 'restaurant':
        prompt += (
            "If you have not introduce the candidate to the user, please:\n"
            "- Inform the user with the name, and ask the user whether he needs a booking.\n"
            "Or else if you are responding to a booking request, please make sure you know the following information:\n "
            "- the information must be known before booking a restaurant are book-day, book-time and book-people.\n"
            "- you can ask these three attributes all at once or step by step.\n"
            "When all the book information, which are book day, book hour and book people are provided by the user, please respond with a confirm:\n"
            "- please inform the user that the booking is successful.\n"
            "- please output the name in your response, "
            "and other information like bookday, booktime, bookpeople are not necessary to inform.\n"
            "- please add an 8 character' reference code with numbers and letters in your response.\n"
        )
    elif service == 'train':
        prompt += (
            "You are responding to a booking request, please make sure you know the following information:\n "
            "- the information needs the number of tickets.\n"
            "When all the booking information is provided by the user, please respond with a confirm:\n"
            "- please inform the user that the booking is successful.\n"
            "- please output the train ID in your response, "
            "and other information like bookday, leaveat, arriveby, destination, departure, price is not necessary to inform.\n"
            "- please add an 8 character' reference code with numbers and letters in your response.\n"
        )
    elif service == 'hotel':
        prompt += (
            "If you have not introduced the candidate to the user, please:\n"
            "- Inform the user with the hotel name, and ask the user whether he needs a booking.\n"
            "Or else if you are responding to a booking request, please make sure you kown the following information:\n "
            "- the information must be known before a hotel booking is book-day, book-stay, and book-people.\n"
            "- you can ask these three attributes one by one, or all at once.\n"
            "When all the book information, which are book-day, book-stay and book-people are provided by the user, please respond with a confirm:\n"
            "- please inform the user that the booking is successful.\n"
            "- please output the name in your response, "
            "and other information like book-day, book-stay and book-people, parking, internet, price range is not necessary to inform.\n"
            "- please add an 8 character' reference code with numbers and letters in your response.\n"
        )
    elif service == 'attraction':
        prompt += (
            "All of the attractions do not support booking, please inform the user with the name, address and contact number briefly and friendly.\n"
            "Please don't inform the open-hours, post-code, unless the user asks for it.\n"
            "Even if the attractions can not be booked, you still need to output the mark `[BOOKED]` as a standard output.\n"
        )
    elif service == 'taxi':
        prompt += (
            "Please select one color from the `taxi-colors` and type from `taxi-types` to response to the user's booking request.\n"
            "You also need to add a phone number with 8 digits as the contact number to inform the user.\n"
        )
    prompt += (
        "- please output a mark `[BOOKED]` at the end of the response.\n"
        "- if the user informed you he doesn't need a booking or reservation at this moment or booking later. "
        "Please reply with good politely and shortly, and also output the mark `[BOOKED]`.\n"
    )
    return prompt


def get_system_inform_prompt(service, **kwargs):
    prompt = (
        "If the user asking you a question, please answer the question directly, and don't output other words.\n"
        "Please don't output the candidate details, to make the answer brief.\n"
        "Please tell the user the number if there are many candidates, and ask the user's preference if it's necessary.\n"
        "Your counterquestion must be short, like 'Do you have any preference?' or 'What are you prefer?'.\n"
        "Please don't make the counterquestion too long.\n"
    )
    if service == 'restaurant':
        prompt += (
            "If the search result is unique, please inform the user with the restaurant name.\n"
        )
    elif service == 'train':
        prompt += (
            "If the search result is unique, please inform the user with the trainid.\n"
        )
    elif service == 'hotel':
        prompt += (
            "If the search result is unique, please inform the user with the hotel name.\n"
        )
    elif service == 'attraction':
        prompt += (
            "If the search result is unique, please inform the user with the attraction name.\n"
        )
    elif service == 'taxi':
        # taxi 不需要 inform，细节都在 asking 阶段
        pass

    return prompt


def get_system_answer_prompt(service, **kwargs):
    prompt = (
        "Please answer the user's question based on the search results directly, don't output other words.\n"
    )
    if service == 'attraction':
        pre_set_keys = ["area", "type", "entrancefee", "openhours", "address", "phone", "postcode", ]
    elif service == 'restaurant':
        pre_set_keys = ["pricerange", "area", "food", "address", "phone", "postcode",]
    elif service == 'hotel':
        pre_set_keys = ["pricerange", "type", "parking", "stars", "internet", "area", "address", "phone", "postcode"]
    else:
        pre_set_keys = []
    response_slots = kwargs.get('response_slots', {})
    response_slots.extend([f'{service} {x}' for x in pre_set_keys if random.random() <= 0.2])
    response_slots = list(set(response_slots))
    print(f'[{service}] system answer slots = {response_slots}')
    if pre_set_keys:
        prompt += f"Please summarize the {','.join(pre_set_keys)} from the search result to response to the user.\n"
    return prompt

class GPTSystemSimulator(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4-1106-preview')

    def prompting(self, service, service_status, history, search_condition, search_results, **kwargs):
        if service_status == 'recommend':
            # 推荐轮次，减token，并增加随机性
            random.shuffle(search_results)
            search_results = search_results[0:5]

        prompt = (
            f"You are a local agent, and now are chatting with the user online for {service}.\n"
            "Given the conversion history and search condition, please read the history and search condition carefully.\n"
            "Then generate a proper response to answer the user demands.\n"
            f"Here is the conversion history:\n{json.dumps(history)}\n"
            f"the search condition: {json.dumps(search_condition)}\n"
            f"the search results: {json.dumps(search_results)}\n"
            "Your response must resemble an online chat as much as possible, and make them as brief as possible.\n"
        )
        # 不同的service，会有一些不同的通用Prompt限制
        prompt += get_system_in_domain_prompt(service)

        print(f'[{service}] system simulater, '
              f'status = {service_status}, '
              f'search results size = {len(search_results)}', flush=True)

        status2func = {
            'inform': get_system_inform_prompt,
            'booking': get_system_booking_prompt,
            'recommend': get_system_recommend_prompt,
            'answer': get_system_answer_prompt,
        }
        skip_service_status = {
            'taxi_recommend',
        }
        # 补充 service 绑定的不同 status 对应的 prompt
        if f'{service}_{service_status}' in skip_service_status:
            pass
        else:
            prompt += status2func.get(service_status, get_system_inform_prompt)(service, **kwargs)

        prompt += (
            'Please answer in a JSON format {"response": ""}\n'
        )
        return prompt


class GPTSystemChattingResponseSimulator(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4-1106-preview')

    def prompting(self, service, history, **kwargs):
        print(f'[{service}] system simulater')
        return (
            f"You are a local agent for `{service}`, and are chatting with the user online.\n"
            "Give your a conversion history and please read the history carefully.\n"
            f"Here is the conversion history:\n{json.dumps(history)}\n"
            "Then generate a casual response to continue or end the conversion if is necessary.\n"
            "The casual response should be:"
            "- highly related to the conversion history, and briefly enough.\n"
            "- resemble an online chat as much as possible, and make them as brief as possible.\n"
            "- make the reply simple when you respond to the users' thanks.\n"
            "- all the words in your response, should be limited to 15 words.\n"
            "- please also add a mark `[EOD]` at the end of your response.\n"
            'Please answer in a JSON format, {"response": ""}\n'
        )
