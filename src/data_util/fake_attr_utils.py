import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List
import os
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

def analyze_attribute_counts(
    data_path: str,
    output_image_path: str,
    item_key: str = "filtered private attributes question"
):
    # Load and analyze data
    with open(data_path, "r") as f:
        data = json.load(f)
        
    attribute_lists = [item[item_key] for item in data if item_key in item]
    # print(attribute_lists[:5])
    attr_series = pd.Series([len(attrs) for attrs in attribute_lists])
    # print(attr_series[:5])
    summary = attr_series.describe(percentiles=[.25, .5, .75]).round(2)
    print("Summary Statistics of Attribute Count per Question:")
    print(summary)

    # Bar plot
    freq_df = attr_series.value_counts().sort_index().reset_index()
    freq_df.columns = ['Attribute Count', 'Number of Questions']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(freq_df['Attribute Count'], freq_df['Number of Questions'], color='skyblue')
    ax.bar_label(bars, padding=3, fontsize=9)

    ax.set_title("Distribution of Attribute Count per Question")
    ax.set_xlabel("Number of Attributes")
    ax.set_ylabel("Number of Questions")
    ax.set_xticks(freq_df['Attribute Count'])
    plt.xticks(rotation=90, fontsize=8)
    
    fig.tight_layout()
    fig.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()


def filter_attribute(
    input_path: str,
    output_path: str,
    item_key: str = "filtered private attributes question"
):
    # Load data from input file
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure data is a list of dictionaries
    if not isinstance(data, list):
        raise ValueError("Expected data to be a list of JSON objects")

    # Filter out entries where the specified item_key is empty
    filtered_data: List[dict] = [
        item for item in data
        if len(item[item_key]) > 0
    ]

    # Save the filtered data to output file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(f"Filtered data saved to: {output_path}")

# if __name__ == "__main__":
    # # legal
    # data_path = f'{root_path}/data/legal-qa-v1/compress_gpt_qcattr.json'
    # output_image_path = f'{root_path}/data/legal-qa-v1/attribute_frequencies_q.png'
    # item_key = "filtered private attributes"
    
    # # medical
    # data_path = f'{root_path}/data/medical_o1_reasoning_SFT/compress_gpt_qcattr_combined.json'
    # output_image_path = f'{root_path}/data/medical_o1_reasoning_SFT/attribute_frequencies_q.png'
    # item_key = "filtered private attributes question"

    # mmlu_fina
    # data_path = f'{root_path}/data/mmlu_fina/fina_raw_data_qattr.json'
    # output_image_path = f'{root_path}/data/mmlu_fina/attribute_frequencies.png'
    # item_key = "filtered private attributes question"
    # analyze_attribute_counts(data_path, output_image_path, item_key)
    # -----
    
    # input_path = f'{root_path}/data/mmlu_fina/fina_raw_data_qattr.json'
    # output_path = f'{root_path}/data/mmlu_fina/fina_raw_data_qattr_none_zero.json'
    # item_key = "filtered private attributes question"
    # filter_attribute(input_path, output_path, item_key)
    # output_image_path = f'{root_path}/data/mmlu_fina/attribute_frequencies_filtered.png'
    # analyze_attribute_counts(output_path, output_image_path, item_key)
    # -----
    
    # input_path = f'{root_path}/data/mmlu_fina/fina_qcattr.json'
    # output_path = f'{root_path}/data/mmlu_fina/fina_qcattr_none_zero.json'
    # output_image_path = f'{root_path}/data/mmlu_fina/com_attribute_frequencies_filtered.png'
    # zero_output_path = f'{root_path}/data/mmlu_fina/fina_qcattr_compression_zero.json'
    # item_key = "filtered private attributes compression"
    # filter_attribute(input_path, output_path, item_key)
    # analyze_attribute_counts(output_path, output_image_path, item_key)
    
    # # Filter out entries where the specified item_key is empty
    # with open(input_path, "r", encoding="utf-8") as f:
    #     data = json.load(f)
    # empty_indices = []
    # empty_data = []
    # for i, item in enumerate(data):
    #     if len(item[item_key]) == 0:
    #         empty_indices.append(i)
    #         empty_data.append(item)
    # print("Indices of empty attributes:", empty_indices)
    # # Save the empty entries into a new JSON file
    # with open(zero_output_path, "w", encoding="utf-8") as f:
    #     json.dump(empty_data, f, ensure_ascii=False, indent=2)
            # [33, 92, 106, 321, 366, 378, 763, 1000, 1094, 1302, 1362, 1444, 1483, 1508]
    # -----