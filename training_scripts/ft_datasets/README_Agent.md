### Data Format

#### Decision Maker

**Do Search**

```
You are a staff member responsible for hotel reservations at a local travel agency. You understand the unique features of each local hotel and can quickly find the hotel that meets users' preferences based on their needs.
Given the conversion history, Your task is to help determine whether the next response can be directly replied to or not.
1. If you think it can be replied without searching, the action would be `chat`.
2. Otherwise the action would be `search`.
In addition, please also output the types of information that need pay attention to for the current action.
Here is the conversion history:
[
  "user: I need a hotel in the south with free wifi.",
  "you: I have a cheap guesthouse, 2 moderately priced guesthouses, and 1 expensive hotel - do you have a preference?"
]
the user lastest utterence: 
Can you tell me if the cheap one has a star rating?
The output should be in JSON format like {"action": xxx, "slots": {"service": [xxx, ...]}}
Please give your decision:
{"action": "search", "slots": {"hotel": ["area", "internet", "pricerange"]}}
```

**Casual**
```
You are a local 114 operator, primarily handling local services such as contacting the police, finding hospitals, calling taxis, and other convenient services. Your service is efficient and of high quality, earning widespread praise from the local community.
Given the conversion history, Your task is to help determine whether the next response can be directly replied to or not.
1. If you think it can be replied without searching, the action would be `chat`.
2. Otherwise the action would be `search`.
In addition, please also output the types of information that need pay attention to for the current action.
Here is the conversion history:
[
  "user: Yes, I need a place to stay that is expensive, and is a hotel please.",
  "you: I have 5 different hotels that meet your needs. Is there a certain area you prefer to stay in?",
  "user: Not really. Do all of them include free parking?",
  "you: Yes, all of these hotels include parking.",
  "user: Can you please book one for me? I would also like the star of the hotel, the postcode, and the phone number please.",
  "you: I would be happy to book a room for you. I need to know when you plan to arrive, and how many people will be in your party.",
  "user: My mistake, no booking needed. I just require the hotel's name, the star of the hotel, its postcode, and also its phone number.",
  "you: The Gonville Hotel is located at Gonville Place. The phone number is 01223366611 and the postcode is cb11ly. It is a 3 star hotel.",
  "user: Thank you. I also need a train from Broxbourne to Cambridge on Monday. I need to arrive in Cambridge by 16:30.",
  "you: I have a train that arrives in Cambridge at 13:32. All other trains arrive after 14:30. Would you like me to book that for you?",
  "user: What was the departure time?",
  "you: The departure time for the TR0354 train is 12:32, it then arrives at 13:32."
]
the user lastest utterence: 
Great. That's all I need, thank you.
The output should be in JSON format like {"action": xxx, "slots": {"service": [xxx, ...]}}
Please give your decision:
{"action": "chat", "slots": {}}
```

**Ask rhetorically**
```
You are a staff member responsible for hotel reservations at a local travel agency. You understand the unique features of each local hotel and can quickly find the hotel that meets users' preferences based on their needs.
Given the conversion history, Your task is to help determine whether the next response can be directly replied to or not.
1. If you think it can be replied without searching, the action would be `chat`.
2. Otherwise the action would be `search`.
In addition, please also output the types of information that need pay attention to for the current action.
Here is the conversion history:
[]
the user lastest utterence: 
I need a place to stay that has a 4 star rating.
The output should be in JSON format like {"action": xxx, "slots": {"service": [xxx, ...]}}
Please give your decision:
{"action": "chat", "slots": {"hotel": ["hotel-parking"]}}
```

#### Generation

**API Search**
```
You are a staff member responsible for hotel reservations at a local travel agency. You understand the unique features of each local hotel and can quickly find the hotel that meets users' preferences based on their needs.
Given the conversion history, your task is to generate the formatted API for searching.
Here is the conversion history:
[]
the user lastest utterence: 
I need a hotel in the south with free wifi.
and here is the slots you'd better to refer when generating the formatted API: 
{"hotel": ["area", "internet"]}
Please give your API:
{"hotel":{"area":"south","internet":"yes"}}
```

**Ask rhetorically**
```
You are a staff member responsible for hotel reservations at a local travel agency. You understand the unique features of each local hotel and can quickly find the hotel that meets users' preferences based on their needs.
Given the conversion history, your task is to generate the next response.
Generate an appropriate response; this response can be in one of the following two styles:
1. Interrogative, If you think that the user's needs have not been met, please ask for the necessary information to provide a more accurate understanding.
2. Direct response: If you believe the conversation is concluded, politely say goodbye; or other direct response based on the conversion history.
Here is the conversion history:
[]
and the user lastest utterence: 
I need a place to stay that has a 4 star rating.
and here is the slots you'd better to ask rhetorically when outputting your response: 
{"hotel": ["hotel-pricerange"]}
Please give your response:
Okay, I'll help you with that. Do you have a price range in mind?
```

**Casual**
```
You are a local 114 operator, primarily handling local services such as contacting the police, finding hospitals, calling taxis, and other convenient services. Your service is efficient and of high quality, earning widespread praise from the local community.
Given the conversion history, your task is to generate the next response.
Generate an appropriate response; this response can be in one of the following two styles:
1. Interrogative, If you think that the user's needs have not been met, please ask for the necessary information to provide a more accurate understanding.
2. Direct response: If you believe the conversation is concluded, politely say goodbye; or other direct response based on the conversion history.
Here is the conversion history:
[
  "user: I need a hotel in the south with free wifi.",
  "you: I have a cheap guesthouse, 2 moderately priced guesthouses, and 1 expensive hotel - do you have a preference?",
  "user: Can you tell me if the cheap one has a star rating?",
  "you: Yes, that would be Rosa's Bed and Breakfast. It has a 4-star rating and offers free wifi and parking. Would you like me to book you a room?",
  "user: Yes, please book a room for 5 people.",
  "you: What day and how long should I book this for?",
  "user: Please book it on Thursday, for 5 people, and 3 days.",
  "you: Your booking was successful. Reference numbr is Y113FEV7 . Can I assist you with anything else today?"
]
and the user lastest utterence: 
That will be all, thank you.
Please generate a proper response based on the context.
Please give your response:
You're welcome. Enjoy your time in Cambridge!
```

