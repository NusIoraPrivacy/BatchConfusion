from data_util.gpt_utils import get_response
from configs.data_configs import *
from configs.templates import attr_extract_template # test new prompt
from configs.key import _API_KEY
from configs.common_words import common_words

from openai import OpenAI
import json
import copy
from nltk.stem import PorterStemmer
import argparse
import os
from tqdm import tqdm
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

stemmer = PorterStemmer()
common_stems = {stemmer.stem(word) for word in common_words}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--gpt_model", type=str, default="ft:gpt-4o-mini-2024-07-18:personal::BBH2yexM") # "gpt-4o-mini-2024-07-18"
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    num = 2000 # set number of samples to generate attributes
    data_paths = ['financial_datasets/financial_datasets_combined.json',
                 'legal-qa-v1/legal_qa_v1.json',
                 'medical_o1_reasoning_SFT/medical_o1_sft.json']
    output_paths = [f'financial_datasets/financial_datasets_combined_attr_{num}.json',
                    f'legal-qa-v1/legal_qa_v1_attr_{num}.json',
                    f'medical_o1_reasoning_SFT/medical_o1_sft_attr_{num}.json']
    # data_path = 'medical_o1_reasoning_SFT/medical_o1_sft.json'
    # output_path = 'medical_o1_reasoning_SFT/medical_o1_sft_attr.json'
    for data_path, output_path in zip(data_paths, output_paths):
        with open(f'{root_path}/data/{data_path}') as fin: 
            # loading new data
            data = json.load(fin)
            data = data[:num] 
        client = OpenAI(api_key=_API_KEY)
        output_data = []
        with tqdm(total=len(data), unit='batch') as pbar:
            for sample in data:
                question = sample["Question"]
                prompt = attr_extract_template.format(question=question)
                # print(prompt)
                success = False
                n_try = 0
                while not success:
                    result = get_response(client, prompt, args)
                    try:
                        prefix = "Sensitive attributes:"
                        if result.startswith(prefix):
                            result = result.replace(prefix, '')
                        result = eval(result)
                        success = True
                    except Exception as e:
                        n_try += 1
                    if n_try >= 5:
                        break
                # print(result)
                if success:
                    output = copy.deepcopy(sample)
                    output["private attributes"] = result
                    
                    # filter out common words
                    filtered_result = []
                    for word in result:
                        if type(word) == int:
                            print(word)
                            filtered_result.append(word)
                        elif stemmer.stem(word) not in common_stems:
                            filtered_result.append(word)
                    
                    # Remove duplicates while preserving order
                    output["filtered private attributes"] = list(dict.fromkeys(filtered_result))
                    
                    output_data.append(output)
                    with open(f'{args.root_path}/data/{output_path}', 'w', encoding="utf-8") as fout: # output new data
                        json.dump(output_data, fout, indent=4, ensure_ascii=False)
                pbar.update(1)