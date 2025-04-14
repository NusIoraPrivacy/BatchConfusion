from data_util.gpt_utils import get_response
from configs.data_configs import *
from configs.templates import attr_generated_template_one_context
from configs.key import _API_KEY

from openai import OpenAI
import json
import copy
import argparse
import os
from tqdm import tqdm
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))


def combine_data(data_filtered_path, data_compressed_path, output_json_path = None,output_json = False):
    with open(f'{root_path}/data/{data_filtered_path}', 'r', encoding="utf-8") as fin:
        data_filtered = json.load(fin)
    with open(f'{root_path}/data/{data_compressed_path}', 'r', encoding="utf-8") as fin:
        data_compressed = json.load(fin)
    
    combined_json = []

    for entry2 in data_compressed:
        for entry1 in data_filtered:
            if entry2["question"] == entry1["Question"]:
                # Combine attributes from entry1 into entry2 and append to combined_json
                combined_entry = entry2.copy()
                combined_entry["private attributes"] = entry1.get("private attributes")
                combined_entry["filtered private attributes"] = entry1.get("filtered private attributes")
                combined_json.append(combined_entry)
                break
    try:
        if output_json:
            with open(f'{root_path}/data/{output_json_path}', 'w', encoding="utf-8") as fout:
                json.dump(combined_json, fout, indent=4, ensure_ascii=False)
    except:
        if output_json_path:
            print(f"Error writing to {output_json_path}")
    
    return combined_json # list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini-2024-07-18")
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_args()
    
    data_paths = [f'medical_o1_reasoning_SFT/medical_o1_sft_combined_compress_attr.json']
    
    output_paths = [f'medical_o1_reasoning_SFT/medical_o1_sft_fake_attr.json']

for data_path, output_path in zip(data_paths, output_paths):
    with open(f'{root_path}/data/{data_path}', 'r', encoding="utf-8") as fin:
        data = json.load(fin)
    
    client = OpenAI(api_key=_API_KEY)
    output_data = []
    
    with tqdm(total=len(data), unit='batch') as pbar:
        for sample in data:
            question = sample["question"]
            compression = sample["compression"]
            filtered_private_attributes = sample["filtered private attributes"]

            # Format the prompt using the refined template (two separate contexts)
            # compression=compression,
            prompt_que = attr_generated_template_one_context.format(question=question,filtered_private_attributes=filtered_private_attributes)
            prompt_cmpr = attr_generated_template_one_context.format(question=compression,filtered_private_attributes=filtered_private_attributes)
            
            success = False
            n_try = 0
            
            while not success:
                result_que = get_response(client, prompt_que, args)
                result_cmpr = get_response(client, prompt_cmpr, args)
                try:
                    # Expecting two separate lists in the output based on the new prompt
                    result_que = eval(result_que)
                    result_cmpr = eval(result_cmpr)
                    
                    if result_que and result_cmpr:
                        success = True
                    else:
                        raise ValueError("Incomplete lists received from GPT-4o")

                except Exception as e:
                    n_try += 1
                    if n_try >= 2:
                        print(f"Failed after 5 attempts: {e}")
                        break

            if success:
                output = copy.deepcopy(sample)
                # Store both generated lists in the output
                output["fake attributes for question"] = result_que
                output["fake attributes for compression"] = result_cmpr
                output_data.append(output)
                
                # Save the output data incrementally after each successful attempt
                with open(f'{root_path}/data/{output_path}', 'w', encoding="utf-8") as fout:
                    json.dump(output_data, fout, indent=4, ensure_ascii=False)


            pbar.update(1)