from openai import OpenAI
from google import genai
from configs.key import OpenAI_API_KEY, GEMINI_API_KEY
import os

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"


def create_client(model_name):
    if "gpt" in model_name:
        client = OpenAI(api_key=OpenAI_API_KEY)
    elif "gemini" in model_name:
        client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        raise ValueError("Invalid model name!")
    return client