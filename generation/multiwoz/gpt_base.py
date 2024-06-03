import copy
import os
import json
import openai
import tiktoken
import requests
import logging

logging.basicConfig(level=logging.WARNING,
                    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', )


class GPTBase:
    """
    model: gpt-4, gpt-4-1106-preview, gpt-4-1106-preview'
    """
    def __init__(self, model='gpt-4-1106-preview', retry_times=3, temperature=0.5):
        self.model = model
        self.temperature = temperature
        self.retry_times = retry_times

    def prompting(self, **kwargs):
        raise NotImplementedError

    def parsing(self, res, **kwargs):
        return json.loads(res.replace('```json\n', '').replace('```', ''))

    def post_processing(self, res, **kwargs):
        try:
            return self.parsing(res, **kwargs)
        except Exception as e:
            return None

    def get(self, **kwargs):
        prompt = self.prompting(**kwargs)
        client = openai.Client(
            api_key=os.environ['OPENAI_API_KEY'],
            base_url=os.environ['OPENAI_BASE_URL']
        )
        if kwargs.get('verbose', False):
            logging.info(f'prompt is, \n {prompt}')

        content = ''
        try:
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )

            choice = response.choices[0]
            content = choice.message.content

            if kwargs.get('verbose', False):
                logging.info(f'output is \"{json.dumps(content)}\"')

            output = self.post_processing(res=content, **kwargs)
            if kwargs.get('verbose', False):
                logging.info(f'output is \"{json.dumps(output)}\"')

            return output
        except Exception as e:
            logging.info(str(content))
            logging.info(str(e))
            pass
        return None

    def __call__(self, **kwargs):
        for i in range(self.retry_times):
            res = self.get(**kwargs)
            if res is not None:
                return res


class GPTTest(GPTBase):
    def __init__(self):
        super().__init__(model='gpt-4-1106-preview')

    def prompting(self, **kwargs):
        prompt = f"""
You are in Cambridge and want to find a local attractions.
Now you are chatting with a local guide online. 
Please generate a response to start the conversion.
Please output the response only.
Please respond briefly, each response should be no more than 15 words.
Please output 10 different responses in Json format like ["response0", "response1", ...]
        """
        return prompt

    def parsing(self, res, **kwargs):
        return res


if __name__ == '__main__':
    import os

    test = GPTTest()
    print(test())