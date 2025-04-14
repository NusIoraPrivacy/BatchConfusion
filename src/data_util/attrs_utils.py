import json
import copy
import os
import argparse
from tqdm import tqdm
from nltk.stem import PorterStemmer
from openai import OpenAI
from data_util.gpt_utils import get_response, get_response_o3
from configs.data_configs import *
from configs.templates import attr_extract_template, attr_generated_template_one_context, attr_generated_template_multiple_rounds
from configs.key import _API_KEY
from configs.common_words import common_words

import time

# Global Variables
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))
stemmer = PorterStemmer()
common_stems = {stemmer.stem(word) for word in common_words}

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

def parse_args(model= "gpt-4o-mini-2024-07-18"): #"o3-mini-2025-01-31"): # "o1-mini-2024-09-12"):# 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=10000,
        help = "max new token for text generation")
    parser.add_argument("--gpt_model", type=str, default=model)
    args = parser.parse_args()

    return args

def rule_based_filter(input_path, output_path, key_name = "private attributes", num=2000):
    """Applies rule-based filtering to remove common words from attributes."""
    with open(f'{input_path}') as fin: 
        data = json.load(fin)
    output_data = []
    for sample in data:
        question = sample[key_name]
        output = copy.deepcopy(sample)
        filtered_result = []
        for word in question:
            if type(word) == int:
                print(word)
                filtered_result.append(word)
            elif stemmer.stem(word) not in common_stems:
                filtered_result.append(word)
        
        # Remove duplicates while preserving order
        output[f"filtered {key_name}"] = list(dict.fromkeys(filtered_result))
        output_data.append(output)
            
    with open(f'{output_path}', 'w', encoding="utf-8") as fout:
        json.dump(output_data, fout, indent=4, ensure_ascii=False)

def generate_attributes(input_path, output_path, key_name = 'Question', output_key_name = "private attributes", num=2000):
    """Uses GPT to generate attributes and filter attributes from given questions."""
    args = parse_args(model = "ft:gpt-4o-mini-2024-07-18:personal::BBH2yexM")
    with open(f'{input_path}') as fin: 
        data = json.load(fin)
    client = OpenAI(api_key=_API_KEY)
    output_data = []
    with tqdm(total=len(data), unit='batch') as pbar:
        for sample in data:
            question = sample[key_name]
            prompt = attr_extract_template.format(question=question)
            
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
                    print(question, result)
                    break
                
            if success:
                output = copy.deepcopy(sample)
                output[output_key_name] = result
                # filter out common words
                filtered_result = []
                for word in result:
                    # if not isinstance(word, str):
                    #     filtered_result.append(str(word))
                    if stemmer.stem(str(word)) not in common_stems:
                        filtered_result.append(str(word))
                
                # Remove duplicates while preserving order
                output[f"filtered {output_key_name}"] = list(dict.fromkeys(filtered_result))
                
                output_data.append(output)
                with open(f'{output_path}', 'w', encoding="utf-8") as fout: # output new data
                    json.dump(output_data, fout, indent=4, ensure_ascii=False)
            pbar.update(1)

def generate_fake_attributes(input_path, output_path, key_name_context = 'question', key_name_attrs = "filtered private attributes", output_key_name = "fake attributes"):
    """Generates fake attributes using GPT based on filtered attributes."""
    args = parse_args()
    with open(f'{input_path}', 'r', encoding="utf-8") as fin:
        data = json.load(fin)
        
    client = OpenAI(api_key=_API_KEY)
    output_data = []
    
    with tqdm(total=len(data), unit='batch') as pbar:
        for sample in data:
            question = sample[key_name_context]
            filtered_private_attributes = sample[key_name_attrs]

            # Format the prompt using the refined template (two separate contexts)
            prompt = attr_generated_template_one_context.format(question=question,filtered_private_attributes=filtered_private_attributes)
            
            success = False
            n_try = 0
            
            while not success:
                result = get_response(client, prompt, args)
                try:
                    result = eval(result)
                    if result:
                        success = True
                    else:
                        raise ValueError("Incomplete lists received from GPT-4o")

                except Exception as e:
                    n_try += 1
                    if n_try >= 5:
                        print(f"Failed after 5 attempts: {e}")
                        print(question, result)
                        output = copy.deepcopy(sample)
                        output_data.append(output)
                        break

            if success:
                output = copy.deepcopy(sample)
                output[output_key_name] = result
                output_data.append(output)
                
                # Save incrementally
                with open(f'{output_path}', 'w', encoding="utf-8") as fout:
                    json.dump(output_data, fout, indent=4, ensure_ascii=False)

            pbar.update(1)

def generate_fake_attributes_multi(input_path, output_path, 
                                   get_response, model="gpt-4o-mini-2024-07-18", 
                                   key_name_context='question', key_name_attrs='filtered private attributes', prev_fake_attrs ='fake attributes', output_key_name='fake attributes', num_rounds=3):
    """Generates multiple rounds of fake attributes and appends them while ensuring uniqueness."""
    args = parse_args(model)
    with open(input_path, 'r', encoding="utf-8") as fin:
        data = json.load(fin)
        # data = data[163:165]
    
    i = 0
    
    client = OpenAI(api_key=_API_KEY)
    output_data = []
    with tqdm(total=len(data) * num_rounds, unit='batch') as pbar:
        for sample in data:
            question = sample[key_name_context]
            filtered_private_attributes = sample[key_name_attrs]
            previous_fake_attributes = sample.get(output_key_name, [])

            # Convert previous fake attributes to a dictionary for fast lookup
            if previous_fake_attributes:
                previous_attr_dict = {row[0]: set(row[1:]) for row in previous_fake_attributes}
            else:
                previous_attr_dict = {attr.strip(): set() for attr in filtered_private_attributes}
                # print(previous_attr_dict)
                # print('-----------------------------------------------------')
            
            success = False
            
            for round_num in range(num_rounds):
                # print(f"Round {round_num + 1}")
                prompt = attr_generated_template_multiple_rounds.format(
                    question=question,
                    filtered_private_attributes=filtered_private_attributes,
                    previous_fake_attributes=previous_fake_attributes
                )

                success = False
                n_try = 0
                
                while not success:
                    result = get_response(client, prompt, args)
                    try:
                        result = eval(result)  # Convert string output into a Python list
                        # Ensure correct format and uniqueness
                        if result: # and all(len(row) == 6 and row[0] in previous_attr_dict for row in result):
                            # Check for uniqueness
                            for row in result:
                                original_attr = row[0]
                                new_fakes = set(row[1:])
                                # # Check for intersection
                                # if new_fakes & previous_attr_dict[original_attr]:  
                                #     raise ValueError("Generated fake attributes contain duplicates from previous rounds")
                                previous_attr_dict[original_attr].update(new_fakes)

                            success = True
                        else:
                            raise ValueError("Incorrect output format from GPT-4o")

                    except Exception as e:
                        n_try += 1
                        if n_try >= 5:
                            print(i)
                            print(f"Failed after 5 attempts: {e}")
                            print(question, result)
                            break

                # if success:
                #     # Append new attributes to the previous list
                #     for row in result:
                #         original_attr = row[0]
                #         existing_entry = next((entry for entry in previous_fake_attributes if entry[0] == original_attr), None)
                #         if existing_entry:
                #             existing_entry.extend(row[1:])  # Append new attributes
                #         else:
                #             previous_fake_attributes.append(row)  # If missing, add it

                #     # Update output structure
                #     output = copy.deepcopy(sample)
                #     output[output_key_name] = previous_fake_attributes
                #     output_data.append(output)

                #     # Save incrementally
                #     with open(output_path, 'w', encoding="utf-8") as fout:
                #         json.dump(output_data, fout, indent=4, ensure_ascii=False)

                # pbar.update(1)
                
                if success:
                    # Append new attributes to the previous list
                    for row in result:
                        original_attr = row[0]
                        existing_entry = next((entry for entry in previous_fake_attributes if entry[0] == original_attr), None)
                        if existing_entry:
                            existing_entry.extend(row[1:])  # Append new attributes
                        else:
                            previous_fake_attributes.append(row)  # If missing, add it

                pbar.update(1)
            
            i += 1
            
            # After all rounds for this sample, update and save
            output = copy.deepcopy(sample)
            output[output_key_name] = previous_fake_attributes
            output_data.append(output)

            # Save all samples once after processing every sample
            with open(output_path, 'w', encoding="utf-8") as fout:
                json.dump(output_data, fout, indent=4, ensure_ascii=False)




if __name__ == "__main__":
    # Generate attributes
    # key_name = 'compression', output_key_name = "private attributes compression"
    # key_name = 'question', output_key_name = "private attributes question"
    # generate_attributes(f'{root_path}/data/medical_o1_reasoning_SFT/compress_gpt_new990.json', f'{root_path}/data/medical_o1_reasoning_SFT/compress_gpt_new990_qattr_1.json', key_name = 'question', output_key_name = "private attributes question")
    # generate_attributes(f'{root_path}/data/medical_o1_reasoning_SFT/compress_gpt_new990_qattr_1.json', f'{root_path}/data/medical_o1_reasoning_SFT/compress_gpt_new990_qcattr_1.json', key_name = 'compression', output_key_name = "private attributes compression")
    
    # generate_attributes(f'{root_path}/data/legal-qa-v1/compress_gpt.json', f'{root_path}/data/legal-qa-v1/compress_gpt_qcattr.json', key_name = 'compression', output_key_name = "private attributes compression")
    
    # _____________________________________________________________________________________________________________
    # Generate fake attributes
    # key_name_context = 'compression', key_name_attr = 'filtered private attributes compression', output_key_name = 'fake attributes compression'
    # key_name_context = 'question', key_name_attr = 'filtered private attributes question', output_key_name = 'fake attributes question'

    # generate_fake_attributes(f'{root_path}/data/legal-qa-v1/compress_gpt_qcattr.json', f'{root_path}/data/legal-qa-v1/compress_gpt_fake_qattr.json', key_name_context = 'question', key_name_attrs = 'filtered private attributes', output_key_name = 'fake attributes question')
    # print("Done")
    # print("--------------------------------------------------------------------------------")
    # generate_fake_attributes(f'{root_path}/data/legal-qa-v1/compress_gpt_fake_qattr.json', f'{root_path}/data/legal-qa-v1/compress_gpt_fake_qcattr.json', key_name_context = 'compression', key_name_attrs = 'filtered private attributes compression', output_key_name = 'fake attributes compression')
    
    # _____________________________________________________________________________________________________________
    # Generate fake attributes with multiple rounds
    # medical_o1_reasoning_SFT
    # key_name_context = 'compression', key_name_attrs = 'filtered private attributes compression', prev_fake_attrs = 'fake attributes compression', output_key_name = 'fake attributes compression'
    # key_name_context = 'question', key_name_attrs = 'filtered private attributes question', prev_fake_attrs = 'fake attributes question', output_key_name = 'fake attributes question'
    
    # generate_fake_attributes_multi(f'{root_path}/data/medical_o1_reasoning_SFT/compress_fake_qattr_multi_4omini_2.json', f'{root_path}/data/medical_o1_reasoning_SFT/compress_fake_qattr_multi_4omini_4.json', key_name_context='question', key_name_attrs='filtered private attributes question', prev_fake_attrs ='fake attributes question', output_key_name='fake attributes question', num_rounds= 2, model = "gpt-4o-mini-2024-07-18", get_response = get_response)
    
    # generate_fake_attributes_multi(f'{root_path}/data/medical_o1_reasoning_SFT/compress_fake_cattr_multi_4omini_4.json', f'{root_path}/data/medical_o1_reasoning_SFT/compress_fake_cattr_multi_4omini_4_add.json', key_name_context='compression', key_name_attrs='filtered private attributes compression', prev_fake_attrs ='fake attributes compression', output_key_name='fake attributes compression', num_rounds= 2, model = "gpt-4o-mini-2024-07-18", get_response = get_response)
    
    
    # legal-qa-v1
    # key_name_context = 'question', key_name_attrs = 'filtered private attributes', prev_fake_attrs = 'fake attributes question', output_key_name = 'fake attributes question'
    # key_name_context='compression', key_name_attrs='filtered private attributes compression', prev_fake_attrs ='fake attributes compression', output_key_name='fake attributes compression', num_rounds= 1
    
    # generate_fake_attributes_multi(f'{root_path}/data/legal-qa-v1/compress_fake_qattr_multi_4omini_3_0.json', f'{root_path}/data/legal-qa-v1/compress_fake_qattr_multi_4omini_4.json', key_name_context='question', key_name_attrs='filtered private attributes', prev_fake_attrs ='fake attributes question', output_key_name='fake attributes question', num_rounds= 1, model = "gpt-4o-mini-2024-07-18", get_response = get_response)
    
    # generate_fake_attributes_multi(f'{root_path}/data/legal-qa-v1/compress_fake_cattr_multi_4omini_3_all.json', f'{root_path}/data/legal-qa-v1/compress_fake_cattr_multi_4omini_4.json', key_name_context='compression', key_name_attrs='filtered private attributes compression', prev_fake_attrs ='fake attributes compression', output_key_name='fake attributes compression', num_rounds= 1, model = "gpt-4o-mini-2024-07-18", get_response = get_response)
    
    # o3-------------------------------   ignore   -----------------------------------
    
    # generate_fake_attributes_multi(f'{root_path}/data/legal-qa-v1/compress_gpt_fake_qcattr.json', f'{root_path}/data/legal-qa-v1/compress_gpt_fake_qcattr_multi_q1_3o.json', key_name_context='question', key_name_attrs='filtered private attributes', prev_fake_attrs ='fake attributes question', output_key_name='fake attributes question', num_rounds= 1, model = "o3-mini-2025-01-31", get_response = get_response_o3)
    
    # with open(f'{root_path}/data/legal-qa-v1/compress_fake_cattr_multi_4omini_4.json', 'r', encoding="utf-8") as fin:
    #     data_1 = json.load(fin)
    #     print(len(data_1))
    
    # with open(f'{root_path}/data/legal-qa-v1/compress_fake_cattr_multi_4omini_3_1717.json', 'r', encoding="utf-8") as fin:
    #     data_2 = json.load(fin)
    #     print(len(data_2))
        
    
    # combined_data = data_1 + data_2
    # print(len(combined_data))
    
    # with open(f'{root_path}/data/legal-qa-v1/compress_fake_cattr_multi_4omini_3_all.json', 'w', encoding="utf-8") as fout:
    #     json.dump(combined_data, fout, indent=4, ensure_ascii=False)
    
    # with open(f'{root_path}/data/legal-qa-v1/compress_fake_cattr_multi_4omini_3_all.json', 'r', encoding="utf-8") as fin:
    #     data = json.load(fin)
    #     print(len(data))
    
    # with open(f'{root_path}/data/legal-qa-v1/compress_fake_qattr_multi_4omini_2.json', 'r', encoding="utf-8") as fin:
    #     data1 = json.load(fin)
    #     # print(len(data1))
    
    # with open(f'{root_path}/data/legal-qa-v1/compress_fake_qattr_multi_4omini_2_785.json', 'r', encoding="utf-8") as fin:
    #     data2 = json.load(fin)
    #     print(len(data2))
    
    # with open(f'{root_path}/data/legal-qa-v1/compress_fake_qattr_multi_4omini_2_867.json', 'r', encoding="utf-8") as fin:
    #     data3 = json.load(fin)
    #     print(len(data3))
    
    # with open(f'{root_path}/data/legal-qa-v1/compress_fake_qattr_multi_4omini_2_1734.json', 'r', encoding="utf-8") as fin:
    #     data4 = json.load(fin)
    #     print(len(data4))
    
    # # print(len(data1) + len(data2) + len(data3) + len(data4))
    
    # combined_data = data1 + data2 + data3 + data4
    # print(len(combined_data))
    
    # with open(f'{root_path}/data/legal-qa-v1/compress_fake_qattr_multi_4omini_2_all.json', 'w', encoding="utf-8") as fout:
    #     json.dump(combined_data, fout, indent=4, ensure_ascii=False)
    
    # with open(f'{root_path}/data/medical_o1_reasoning_SFT/compress_fake_cattr_multi_4omini_4.json', 'r', encoding="utf-8") as fin:
    #     data3 = json.load(fin)
    #     print(data3[163])