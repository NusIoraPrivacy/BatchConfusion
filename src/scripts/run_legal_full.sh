#!/bin/bash

models=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct")
sample_multipliers=(2 5 10 15 20)
sample_top_ks=(0)

for sample_mul in ${sample_multipliers[@]}
do
    for model in ${models[@]}
    do
        model_short_name="${model##*/}"
        echo "Generate fake combinations under multiplier $sample_mul using with model $model_short_name"
        python -m train_local.generator_v3 --out_file_name "fake_qattr_random_${model_short_name}_${sample_mul}.json" --sample_mul $sample_mul --fake_key "fake attributes question" --priv_key "filtered private attributes" --query_key question --data_name legal-qa-v1 --in_file_name compress_fake_qattr_multi_4omini_4.json --model_name $model --max_length 250 --batch_size 100
        # echo "Prompt judgement attack under top-k ratio $top_k with sample top-k $sample_top_k"
        # python -m attack.pjd_attack --in_file_name "fake_qattr_random_${model_short_name}_${sample_mul}.json" --model_name "Qwen/Qwen2.5-1.5B-Instruct" --fake_key "fake attributes question" --priv_key "filtered private attributes" --query_key question --data_name legal-qa-v1 --sample_top_k $sample_top_k
    done
done