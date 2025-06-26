#!/bin/bash

sample_multipliers=(5 10)
sample_top_ks=(0)
models=("Qwen/Qwen2.5-0.5B-Instruct")
# dis_models=("FacebookAI/roberta-large" "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B")
dis_models=("meta-llama/Llama-3.1-8B")

# for top_k in ${top_k_ratios[@]}
# do
#     for sample_top_k in ${sample_top_ks[@]}
#     do
#         echo "Generate fake combinations under top-k ratio $top_k"
#         python -m train_local.generator_v2 --out_file_name "fake_cattr_random_$top_k.json" --top_k_ratio $top_k --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --data_name mmlu_fina --in_file_name fina_fake_qcattr_none_zero.json --model_name Qwen/Qwen2.5-1.5B-Instruct --max_length 100
#         for model in ${models[@]}
#         do
#             echo "Prompt judgement attack under top-k ratio $top_k with sample top-k $sample_top_k with model $model"
#             python -m attack.pjd_attack --in_file_name "fake_cattr_random_$top_k.json" --model_name $model --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --data_name mmlu_fina --sample_top_k $sample_top_k --epochs 10 --save_model
#         done
#     done
# done

for sample_mul in ${sample_multipliers[@]}
do
    
    for model in ${models[@]}
    do
        model_short_name="${model##*/}"
        # echo "Generate fake combinations under multiplier $sample_mul using with model $model_short_name"
        # python -m train_local.generator_v3 --out_file_name "fake_cattr_random_${model_short_name}_${sample_mul}.json" --sample_mul $sample_mul --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --data_name mmlu_fina --in_file_name fina_fake_qcattr_none_zero.json --model_name $model --max_length 100
        for sample_top_k in ${sample_top_ks[@]}
        do
            for dis_model in ${dis_models[@]}
            do
                echo "Prompt judgement attack under multiplier $sample_mul with sample top-k $sample_top_k with discrimination model $dis_model, generator model $model"
                python -m attack.pjd_attack --in_file_name "fake_cattr_random_${model_short_name}_${sample_mul}.json" --model_name $dis_model --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --data_name mmlu_fina --sample_top_k $sample_top_k --epochs 20 --train_batch_size 5 --test_batch_size 10 --use_peft
            done
        done
    done
done

models=("Qwen/Qwen2.5-1.5B-Instruct")
sample_multipliers=(2 5 10)

for sample_mul in ${sample_multipliers[@]}
do
    
    for model in ${models[@]}
    do
        model_short_name="${model##*/}"
        for sample_top_k in ${sample_top_ks[@]}
        do
            for dis_model in ${dis_models[@]}
            do
                echo "Prompt judgement attack under multiplier $sample_mul with sample top-k $sample_top_k with discrimination model $dis_model, generator model $model"
                python -m attack.pjd_attack --in_file_name "fake_cattr_random_${model_short_name}_${sample_mul}.json" --model_name $dis_model --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --data_name mmlu_fina --sample_top_k $sample_top_k --epochs 20 --train_batch_size 5 --test_batch_size 10 --use_peft
            done
        done
    done
done