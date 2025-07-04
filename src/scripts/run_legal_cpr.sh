#!/bin/bash

top_k_ratios=(0.1 0.2 0.3 0.4)
# top_k_ratios=(0.6)
sample_top_ks=(0 1)

for top_k in ${top_k_ratios[@]}
do
    for sample_top_k in ${sample_top_ks[@]}
    do
        echo "Generate fake combinations under top-k ratio $top_k"
        python -m train_local.generator_v2 --out_file_name "fake_cattr_random_$top_k.json" --top_k_ratio $top_k --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --data_name legal-qa-v1 --in_file_name compress_fake_cattr_multi_4omini_4.json --model_name meta-llama/Llama-3.2-1B --max_length 100
        echo "Prompt judgement attack under top-k ratio $top_k with sample top-k $sample_top_k"
        python -m attack.pjd_attack --in_file_name "fake_cattr_random_$top_k.json" --model_name "Qwen/Qwen2.5-1.5B-Instruct" --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --data_name legal-qa-v1 --sample_top_k $sample_top_k
    done
done