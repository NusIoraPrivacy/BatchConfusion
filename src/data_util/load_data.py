from datasets import load_dataset
import json
import os
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

def creat_mmlu_dict(data_subset, cat):
    out = []
    questions = data_subset["question"]
    choices = data_subset["choices"]
    answers = data_subset["answer"]
    sub_cats = data_subset["subject"]
    for question, choice, answer, sub_cat in zip(questions, choices, answers, sub_cats):
        this_dict = {
            "question": question,
            "choices": choice,
            "answer": answer,
            "category": cat,
            "sub-category": sub_cat,
        }
        out.append(this_dict)

    return out

def creat_mmlu_cat_data(cat_dict):
    filter_data = []
    for cat in cat_dict:
        sub_cats = cat_dict[cat]
        for sub_cat in sub_cats:
            data_subset = load_dataset("cais/mmlu", sub_cat, split="test")
            filter_data.extend(creat_mmlu_dict(data_subset, cat))
            data_subset = load_dataset("cais/mmlu", sub_cat, split="validation")
            filter_data.extend(creat_mmlu_dict(data_subset, cat))
    return filter_data

if __name__ == "__main__":
    # cat_dict = {"business": ["business_ethics", "marketing"],
    #         "legal": ["international_law", "jurisprudence", "professional_law"],
    #         "politics": ["us_foreign_policy", "high_school_government_and_politics"],
    #         "medicine": ["college_medicine", "clinical_knowledge", "nutrition", "professional_psychology", "high_school_psychology", "professional_medicine"],
    #         "religion": ["world_religions"]
    #         }
    # data = creat_mmlu_cat_data(cat_dict)
    # with open(f'{root_path}/data/mmlu/raw_data.json', 'w') as fout:
    #     json.dump(data, fout, indent=4)
    # print(data[:5])
    # print(len(data))
    
    
    cat_dict = {"business": ["business_ethics", "econometrics", "marketing", "high_school_macroeconomics", "high_school_microeconomics", "management", "marketing"]}
    # cat_dict = {"business": ["business_ethics"]}
    data = creat_mmlu_cat_data(cat_dict)
    with open(f'{root_path}/data/mmlu_fina/fina_raw_data.json', 'w') as fout:
        json.dump(data, fout, indent=4)
    print(data[:5])
    print(len(data))