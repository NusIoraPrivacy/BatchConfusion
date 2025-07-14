#!/bin/bash

gpt_models=("gpt-4o-mini" "gpt-4o" "gpt-3.5-turbo")
# gpt_models=("gpt-4o-mini")

for gpt_model in ${gpt_models[@]}
do
    echo "Testing for model $gpt_model"
    python -m baselines.custext --remote_cpr "fina_fake_qattr_none_zero.json" --data_name mmlu_fina --data_size 400 --model_name openai-community/gpt2 --epsilon 1 --gpt_model $gpt_model
done
