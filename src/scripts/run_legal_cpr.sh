#!/bin/bash

models=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct")
sample_multipliers=(2 5 10)
sample_top_ks=(0)
dis_models=("FacebookAI/roberta-large" "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B")

# python -m train_local.generator_v2 --out_file_name "fake_cattr_random_1.json" --top_k_ratio 1 --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --data_name legal-qa-v1 --in_file_name compress_fake_cattr_multi_4omini_4.json

# for model in ${models[@]}
# do
#     echo "Prompt judgement attack using discrimination model $model"
#     python -m attack.pjd_attack --in_file_name "fake_cattr_random_1.json" --model_name $model --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --data_name legal-qa-v1 --epochs 10 --sample_top_k 0 --save_model
# done


for sample_mul in ${sample_multipliers[@]}
do
    
    for model in ${models[@]}
    do
        model_short_name="${model##*/}"
        echo "Generate fake combinations under multiplier $sample_mul using with model $model_short_name"
        python -m train_local.generator_v3 --out_file_name "fake_cattr_random_${model_short_name}_${sample_mul}.json" --sample_mul $sample_mul --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --data_name legal-qa-v1 --in_file_name compress_fake_cattr_multi_4omini_4.json --model_name $model --max_length 100
        for dis_model in ${dis_models[@]}
        do
            echo "Prompt judgement attack under multiplier $sample_mul using discrimination model $dis_model, generator model $model"
            if [ $dis_model = "meta-llama/Llama-3.1-8B" ]; then
                python -m attack.pjd_attack --in_file_name "fake_cattr_random_${model_short_name}_${sample_mul}.json" --model_name $dis_model --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --data_name legal-qa-v1 --sample_top_k 0 --epochs 20 --use_peft
            else
                python -m attack.pjd_attack --in_file_name "fake_cattr_random_${model_short_name}_${sample_mul}.json" --model_name $dis_model --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --data_name legal-qa-v1 --sample_top_k 0 --epochs 20
            fi
        done
    done
done