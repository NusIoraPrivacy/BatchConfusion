from configs.common_words import common_words

import json
import copy
from nltk.stem import PorterStemmer
import os
from tqdm import tqdm
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

stemmer = PorterStemmer()
common_stems = {stemmer.stem(word) for word in common_words}

if __name__ == "__main__":
    num = 2000 # set number of samples to generate attributes
    data_paths = [f'financial_datasets/financial_datasets_combined_attr_{num}.json',
                  f'legal-qa-v1/legal_qa_v1_attr_{num}.json',
                  f'medical_o1_reasoning_SFT/medical_o1_sft_attr_{num}.json']
                #   
    output_paths = [f'financial_datasets/financial_datasets_combined_attr_{num}_filteres.json',
                    f'legal-qa-v1/legal_qa_v1_attr_{num}_filteres.json',
                    f'medical_o1_reasoning_SFT/medical_o1_sft_attr_{num}_filteres.json']

    for data_path, output_path in zip(data_paths, output_paths):
        with open(f'{root_path}/data/{data_path}') as fin: 
            # loading new data
            data = json.load(fin)
            output_data = []
            for sample in data:
                question = sample["private attributes"]
                output = copy.deepcopy(sample)
                filtered_result = []
                for word in question:
                    if type(word) == int:
                        print(word)
                        filtered_result.append(word)
                    elif stemmer.stem(word) not in common_stems:
                        filtered_result.append(word)
                
                # Remove duplicates while preserving order
                output["filtered private attributes"] = list(dict.fromkeys(filtered_result))
                output_data.append(output)
                
        with open(f'{root_path}/data/{output_path}', 'w', encoding="utf-8") as fout:
            json.dump(output_data, fout, indent=4, ensure_ascii=False)
