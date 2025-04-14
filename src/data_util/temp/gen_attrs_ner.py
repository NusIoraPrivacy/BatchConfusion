import spacy
import copy
import argparse
import os
import json
from tqdm import tqdm
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

def extract_attributes(text):
    doc = nlp(text)
    attributes = []

    # Predefined categories (KNOW framework-like mapping)
    predefined_attributes = {"PERSON", "FAC", "LOC", "PRODUCT", "WORK_OF_ART", "LAW", 
                            "LANGUAGE", "ORG", "GPE", "DATE", "EVENT", "NORP", "TIME"}

    # Extract named entities that match predefined categories
    for ent in doc.ents:
        if ent.label_ in predefined_attributes:
            attributes.append(ent.text)

    return attributes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--extract_origin", type=bool, default=False,
        help = "whether to extract attributes from origin question")
    parser.add_argument("--extract_compress", type=bool, default=True,
        help = "whether to extract attributes from compressed question")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.extract_origin:
        with open(f'{args.root_path}/data/mmlu/raw_data.json') as fin:
            data = json.load(fin)

        output_data = []
        with tqdm(total=len(data), unit='batch') as pbar:
            for sample in data:
                question = sample["question"]
                attrs = extract_attributes(question)
                output = copy.deepcopy(sample)
                output["private attributes"] = attrs
                output_data.append(output)
                pbar.update(1)

        with open(f'{args.root_path}/data/mmlu/private_attrs_ner.json', 'w') as fout:
            json.dump(output_data, fout, indent=4)
    
    if args.extract_compress:
        with open(f'{args.root_path}/data/mmlu/compress_gpt.json') as fin:
            data = json.load(fin)

        output_data = []
        with tqdm(total=len(data), unit='batch') as pbar:
            for sample in data:
                origin_question, compress_question = sample["question"], sample["compression"]
                origin_attrs = extract_attributes(origin_question)
                compress_attrs = extract_attributes(compress_question)
                output = copy.deepcopy(sample)
                output["origin private attributes"] = origin_attrs
                output["compress private attributes"] = compress_attrs
                output_data.append(output)
                pbar.update(1)

        with open(f'{args.root_path}/data/mmlu/private_attrs_compress_ner.json', 'w') as fout:
            json.dump(output_data, fout, indent=4)