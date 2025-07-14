#!/bin/bash

# models=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B") 

models=("Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") 

# for model in ${models[@]}
# do
#     echo "Train compressor model $model"
#     python -m train_local.train_compress --model_name $model --data_name medical_o1_reasoning_SFT --in_file_name compress_fake_cattr_multi_4omini_4.json --query_key question --cpr_key compression --train_batch_size 2 --test_batch_size 4
# done

# epochs=("epoch_1.json" "epoch_2.json" "epoch_3.json" "epoch_4.json" "epoch_5.json")
epochs=("epoch_3.json")
for model in ${models[@]}
do
    model_short_name="${model##*/}"
    for epoch in ${epochs[@]}
    do
        echo "Evaluate compressor model $model_short_name on epoch $epoch"
        python -m evaluation.utility_oa --data_name medical_o1_reasoning_SFT --local_model $model_short_name --local_epoch $epoch --query_online_model gpt-4o --remote_cpr compress_fake_cattr_multi_4omini_4.json --answer_key response --query_key question --cpr_key "predicted compression"
    done
done