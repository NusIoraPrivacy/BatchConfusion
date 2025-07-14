import json
import os
import argparse
import itertools

current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

data_names = ["legal-qa-v1", "medical_o1_reasoning_SFT", "mmlu_fina", "twitter"]
fake_keys = ['fake attributes question', 'fake attributes compression', "fake attributes text"]
priv_keys = ["filtered private attributes question", "filtered private attributes compression", "filtered private attributes", "filtered private attributes text"]
query_keys = ["question", "compression", "text"]


def get_unique_combs(comb_list):
    seen = set()
    new_comb_list = []
    for combination in comb_list:
        if tuple(combination) not in seen:
            seen.add(tuple(combination))
            new_comb_list.append(combination)
    return new_comb_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--data_name", type=str, default=data_names[0])
    parser.add_argument("--in_file_name", type=str, default="fake_cattr_random_Qwen2.5-0.5B-Instruct_10.json")
    parser.add_argument("--fake_key", type=str, default=fake_keys[0])
    parser.add_argument("--priv_key", type=str, default=priv_keys[2])
    parser.add_argument("--query_key", type=str, default=query_keys[0])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # for model in ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"]:
    for model in ["Qwen2.5-0.5B-Instruct"]:
        for ratio in [2, 5, 10]:
        # for ratio in [2]:
            args.in_file_name = f"fake_qattr_random_{model}_{ratio}.json"
            print("----------------------------------")
            print(f"### Counting for {args.in_file_name} ###")
            data_path = f'{args.root_path}/result/{args.data_name}/{args.in_file_name}'
            with open(data_path) as fin:
                data = json.load(fin)
            print(len(data))
            n_fakes = []
            n_attrs = []
            n_2way_attrs = []
            for sample in data:
                priv_attrs, fake_attrs, query = sample[args.priv_key], sample[args.fake_key], sample[args.query_key]
            
                n_fakes.append(len(fake_attrs))

                for attrs in zip(*fake_attrs):
                    attrs = set(list(attrs))
                    n_attrs.append(len(attrs))
                
                ### count 2-way combinations
                fake_attrs_2way = []
                for fake_attr_list in fake_attrs:
                    this_fakes = []
                    for subset in itertools.combinations(fake_attr_list, 2):
                        this_fakes.append(list(subset))
                    fake_attrs_2way.append(this_fakes)
                    # print(this_fakes)
                
                for attrs in zip(*fake_attrs_2way):
                    attrs = get_unique_combs(attrs)
                    # print(attrs)
                    n_2way_attrs.append(len(attrs))

            avg_n_fakes = sum(n_fakes)/len(n_fakes)
            avg_n_attrs = sum(n_attrs)/len(n_attrs)
            avg_n_2way = sum(n_2way_attrs)/len(n_2way_attrs)
            print(f"Average fake combinations: {avg_n_fakes}")
            print(f"Average fake 2-way combinations: {avg_n_2way}")
            print(f"Average unique attributes: {avg_n_attrs}")