#!/bin/bash

top_k_ratios=(0.1 0.2 0.3 0.4)
# top_k_ratios=(0.4 0.1)
sample_top_ks=(0 1)

for top_k in ${top_k_ratios[@]}
do
    for sample_top_k in ${sample_top_ks[@]}
    do
        echo "Generate fake combinations under top-k ratio $top_k"
        python -m train_local.generator_v2 --out_file_name "fake_qattr_random_$top_k.json" --top_k_ratio $top_k --fake_key "fake attributes question" --priv_key "filtered private attributes question" --query_key question --data_name mmlu_fina --in_file_name fina_fake_qcattr_none_zero.json --model_name meta-llama/Llama-3.2-1B
        echo "Prompt judgement attack under top-k ratio $top_k with sample top-k $sample_top_k"
        python -m attack.pjd_attack --in_file_name "fake_qattr_random_$top_k.json" --model_name "Qwen/Qwen2.5-1.5B-Instruct" --fake_key "fake attributes question" --priv_key "filtered private attributes question" --query_key question --data_name mmlu_fina --sample_top_k $sample_top_k
    done
done