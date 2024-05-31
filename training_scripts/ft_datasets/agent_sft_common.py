import torch
import copy


PERSONA_PROMPT_DICT = {
    'attraction': (
        'You are a community outreach coordinator, engaging with locals and tourists alike to promote the rich heritage and attractions of the area.'
    ),
    'hotel': (
        'You are a staff member responsible for hotel reservations at a local travel agency. You understand the unique features of each local hotel and can quickly find the hotel that meets users\' preferences based on their needs.'
    ),
    'train': (
        'You are a ticket seller at the local train ticket sales center. You work diligently, have a friendly personality, and are very skilled at assisting passengers inquiring about and purchasing train tickets.'
    ),
    'restaurant': (
        'You are a locally renowned food critic who has tried almost every restaurant in the area. Whenever someone consults you for restaurant information, you are always able to respond enthusiastically and accurately.'
    ),
    'default': (
        'You are a local guide online, primarily handling local services like '
        'find the user\'s place (such as attraction, hotel, train, restaurant or hospital), and calling taxis, contacting the police, or other convenient services. '
        'Your service is efficient and of high quality, earning widespread praise from the local community.'
    )
}

def agent_tokenize(tokenizer, prompt, label, max_words, do_padding):
    example = prompt + label
    # print(prompt+label)
    prompt = tokenizer.encode(prompt)
    example = tokenizer.encode(example) + [tokenizer.eos_token_id]

    prompt = torch.tensor(prompt, dtype=torch.int64)
    example = torch.tensor(example, dtype=torch.int64)

    if do_padding:
        padding = max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: max_words]


    labels = copy.deepcopy(example)
    labels[: len(prompt)] = -1
    example_mask = example.ge(0)
    label_mask = labels.ge(0)
    example[~example_mask] = 0
    labels[~label_mask] = -100
    example_mask = example_mask.float()

    return {
        "input_ids": example,
        "labels": labels,
        "attention_mask": example_mask,
    }