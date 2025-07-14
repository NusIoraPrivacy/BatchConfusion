import time # added
from openai import OpenAI
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)
from google import genai

def create_message(prompt):
    messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    return messages
    
def get_response(client, prompt, args, gpt_model=None):
    """
    Obtain response from GPT
    """
    SLEEP_TIME = 10
    success = False
    cnt = 0
    messages = create_message(prompt)
    if gpt_model is None:
        gpt_model = args.gpt_model
    while not success:
        if cnt >= 50:
            rslt = "Error"
            break
        try:
            if "gpt" in gpt_model:
                response = client.chat.completions.create(
                    model=gpt_model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                rslt = response.choices[0].message.content
            elif "gemini" in gpt_model:
                response = client.models.generate_content(
                    model=gpt_model, contents=prompt,
                    # config=genai.types.GenerateContentConfig(
                    #         max_output_tokens=args.max_tokens,
                    #         temperature=args.temperature,
                    #     ),
                )
                rslt = response.text
                # print(rslt)
            else:
                raise ValueError("Invalid model name!")
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
        except genai.errors.APIError as e:
            print(f"sleep {SLEEP_TIME} seconds for api error")
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(e)
            success = True
            rslt = "Error"
        cnt += 1
    return rslt

def get_response_multiprompts(client, prompts, args, gpt_model=None):
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
    if gpt_model is None:
        gpt_model = args.gpt_model
    while not success:
        if cnt >= 50:
            rslt = "Error"
            break
        try:
            response = client.chat.completions.create(
                model=gpt_model,
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


def get_response_o3(client, prompt, args, gpt_model=None):
    """
    Obtain response from GPT
    """
    SLEEP_TIME = 10
    success = False
    cnt = 0
    messages = create_message(prompt)
    if gpt_model is None:
        gpt_model = args.gpt_model
    while not success:
        if cnt >= 50:
            rslt = "Error"
            break
        try:
            response = client.chat.completions.create(
                    model=gpt_model,
                    messages=messages,
                    reasoning="low",
                )
            rslt = response.choices[0].message.content
            cnt += 1
            success = True
        except RateLimitError as e:
            print(e)
            print(f"sleep {SLEEP_TIME} seconds for rate limit error")
            time.sleep(SLEEP_TIME)
        except APITimeoutError as e:
            print(e)
            print(f"sleep {SLEEP_TIME} seconds for time out error")
            time.sleep(SLEEP_TIME)
        except APIConnectionError as e:
            print(e)
            print(f"sleep {SLEEP_TIME} seconds for api connection error")
            time.sleep(SLEEP_TIME)
        except APIError as e:
            print(e)
            print(f"sleep {SLEEP_TIME} seconds for api error")
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(e)
            success = True
            rslt = "Error"
        cnt += 1
    return rslt