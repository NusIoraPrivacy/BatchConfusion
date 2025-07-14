#!/bin/bash

# models=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B") 
# models=("meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.1-8B")

# for model in ${models[@]}
# do
#     echo "Train compressor model $model"
#     python -m train_local.train_compress --model_name $model --data_name mmlu_fina --in_file_name fina_fake_qcattr_none_zero.json --query_key question --cpr_key compression --train_batch_size 2 --test_batch_size 4
# done

models=("Llama-3.2-1B" "Llama-3.2-3B" "Llama-3.1-8B")
epochs=("epoch_1.json" "epoch_2.json" "epoch_3.json" "epoch_4.json" "epoch_5.json")
for model in ${models[@]}
do
    for epoch in ${epochs[@]}
    do
        echo "Evaluate compressor model $model on epoch $epoch"
        python -m evaluation.mmlu_utility --data_name mmlu_fina --local_model $model --local_epoch $epoch --gpt_model gpt-4o-mini --remote_cpr fina_fake_qattr_none_zero.json
    done
done