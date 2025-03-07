import time # added
from openai import OpenAI
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)

def create_message(prompt):
    messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    return messages
    
def get_response(client, prompt, args):
    """
    Obtain response from GPT
    """
    SLEEP_TIME = 10
    success = False
    cnt = 0
    messages = create_message(prompt)
    while not success:
        if cnt >= 50:
            rslt = "Error"
            break
        try:
            response = client.chat.completions.create(
                model=args.gpt_model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            rslt = response.choices[0].message.content
            success = True
        except RateLimitError as e:
            print(f"sleep {SLEEP_TIME} seconds for rate limit error")
            time.sleep(SLEEP_TIME)
        except APITimeoutError as e:
            print(f"sleep {SLEEP_TIME} seconds for time out error")
            time.sleep(SLEEP_TIME)
        except APIConnectionError as e:
            print(f"sleep {SLEEP_TIME} seconds for api connection error")
            time.sleep(SLEEP_TIME)
        except APIError as e:
            print(f"sleep {SLEEP_TIME} seconds for api error")
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(e)
            success = True
            rslt = "Error"
        cnt += 1
    return rslt

def get_response_multiprompts(client, prompts, args):
    """
    Obtain response from GPT
    """
    SLEEP_TIME = 10
    success = False
    cnt = 0
    roles = ["user", "assistant"]
    messages = []
    for i, prompt in enumerate(prompts):
        i = i % 2
        role = roles[i]
        this_message = {"role": role, "content": prompt}
        messages.append(this_message)
    while not success:
        if cnt >= 50:
            rslt = "Error"
            break
        try:
            response = client.chat.completions.create(
                model=args.gpt_model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            rslt = response.choices[0].message.content
            success = True
        except RateLimitError as e:
            print(f"sleep {SLEEP_TIME} seconds for rate limit error")
            time.sleep(SLEEP_TIME)
        except APITimeoutError as e:
            print(f"sleep {SLEEP_TIME} seconds for time out error")
            time.sleep(SLEEP_TIME)
        except APIConnectionError as e:
            print(f"sleep {SLEEP_TIME} seconds for api connection error")
            time.sleep(SLEEP_TIME)
        except APIError as e:
            print(f"sleep {SLEEP_TIME} seconds for api error")
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(e)
            success = True
            rslt = "Error"
        cnt += 1
    return rslt