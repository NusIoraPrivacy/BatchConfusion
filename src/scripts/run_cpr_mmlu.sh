#!/bin/bash

# models=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B") 
# models=("meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B")

# for model in ${models[@]}
# do
#     echo "Train compressor model $model"
#     python -m train_local.train_compress --model_name $model --data_name mmlu_fina --in_file_name fina_fake_qcattr_none_zero.json --query_key question --cpr_key compression --train_batch_size 2 --test_batch_size 4
# done

models=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B") 
# epochs=("epoch_1.json" "epoch_2.json" "epoch_3.json" "epoch_4.json" "epoch_5.json")
epochs=("epoch_3.json" "epoch_5.json")
gpt_models=("gpt-4o" "gpt-3.5-turbo")
for model in ${models[@]}
do
    model_short_name="${model##*/}"
    for epoch in ${epochs[@]}
    do
        for gpt_model in ${gpt_models[@]}
        do
            echo "Evaluate compressor model $model_short_name on epoch $epoch"
            python -m evaluation.mmlu_utility --data_name mmlu_fina --local_model $model_short_name --local_epoch $epoch --gpt_model $gpt_model --remote_cpr fina_fake_qattr_none_zero.json
        done
    done
done