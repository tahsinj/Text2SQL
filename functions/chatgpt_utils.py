import time
import openai
import os
import json
from openai import OpenAI

# ----------------------------------------------
def get_openai_response(prompt, user_input, temperature=0, max_tokens=50, delay=None):
    apikey = os.environ["OPENAI_API_KEY"]
    openai.api_key = apikey
    msgs = [{"role": "system", "content": prompt},
            {"role": "user", "content": user_input}]
    return get_openai_response_msg(msgs, temperature, max_tokens, delay)


def get_openai_response_msg(messages = [{"role": "system", "content": "You are a helpful assistant."}], temperature=0, max_tokens=50, delay=None, model="gpt-3.5-turbo"):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],  # This is the default and can be omitted
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    if delay is not None:
        # Sleep for the delay
        time.sleep(delay)

    message_content = response.choices[0].message.content
    message = {
        "content": message_content,
        "role": response.choices[0].message.role,
        "usage": {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }
    return message

def get_openai_function_call(messages = [{"role": "system", "content": "You are a helpful assistant."}],
                             functions=[],
                             function_call="auto",
                             temperature=0,
                             max_tokens=50,
                             delay=None,
                             model="gpt-3.5-turbo"):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],  # This is the default and can be omitted
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        functions=functions,
        function_call=function_call
    )
    if delay is not None:
        # Sleep for the delay
        time.sleep(delay)
    
    function_call_info = response.choices[0].message.function_call
    parsed_arguments = None

    if function_call_info and hasattr(function_call_info, 'arguments'):
        try:
            # Parse the arguments as a dictionary
            parsed_arguments = json.loads(function_call_info.arguments)
        except Exception as e:
            print(f"Error decoding function call arguments: {e}")
    
    # Construct the response message with the entities included
    message_content = response.choices[0].message.content
    message = {
        "content": message_content,
        "role": response.choices[0].message.role,
        "usage": {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        },
        "arguments": parsed_arguments
    }
    
    return message

def get_embeddings(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response.data[0].embedding