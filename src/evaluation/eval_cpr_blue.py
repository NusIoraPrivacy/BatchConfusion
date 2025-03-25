import json
import os

from rouge import Rouge
from sacrebleu.metrics import BLEU

current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

if __name__ == "__main__":
    models = ["bart-large", "t5-large", "flan-t5-large", "gemma-2-9b-it-SimPO"]
    # models = ["gemma-2-9b-it-SimPO"]
    data_name = "medical_o1_reasoning_SFT"
    bleu_scorer = BLEU(effective_order=True)
    rouge_scorer = Rouge()
    for model in models:
        file_path = f'{root_path}/result/{data_name}/compression/{model}'
        file_list = os.listdir(file_path)
        for file_name in file_list:
            with open(f'{file_path}/{file_name}') as fin:
                data = json.load(fin)
            rougel_list = []
            blue_list = []
            for sample in data:
                org_cpr, pred_cpr = sample["compression"], sample["predicted compression"]
                try:
                    score = rouge_scorer.get_scores(hyps=pred_cpr, refs=org_cpr)
                    rougeL = score[0]["rouge-l"]["f"]
                except ValueError as e:
                    rougeL = 0
                try:
                    blue_score = bleu_scorer.sentence_score(hypothesis=pred_cpr, references=[org_cpr])
                    blue = blue_score.score/100
                except ValueError as e:
                    blue = 0
                rougel_list.append(rougeL)
                blue_list.append(blue)
            
            rougel = sum(rougel_list) / len(rougel_list)
            blue = sum(blue_list) / len(blue_list)
            print(f"Rouge for {model} {file_name}:", rougel)
            print(f"BLUE for {model} {file_name}:", blue)