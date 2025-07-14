#!/bin/bash

gpt_models=("gpt-4o-mini" "gpt-4o" "gpt-3.5-turbo")

for gpt_model in ${gpt_models[@]}
do
    echo "Testing for model $gpt_model"
    python -m baselines.paraphrase --remote_cpr "fina_fake_qattr_none_zero.json" --data_name mmlu_fina --data_size 400 --para_model eugenesiow/bart-paraphrase --epsilon 10 --gpt_model $gpt_model
done
