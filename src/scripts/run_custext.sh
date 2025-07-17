#!/bin/bash

gpt_models=("gpt-4o-mini" "gpt-4o" "gpt-3.5-turbo")
# epsilons=(1 10)
epsilons=(3 5)

for gpt_model in ${gpt_models[@]}
do
    for eps in ${epsilons[@]}
    do
        echo "Testing custex for model $gpt_model under epsilon $eps"
        python -m baselines.custext --remote_cpr "fina_fake_qattr_none_zero.json" --data_name mmlu_fina --data_size 400 --model_name openai-community/gpt2 --epsilon 1 --gpt_model $gpt_model --top_k 20 --epsilon $eps

        echo "Testing paraphrase for model $gpt_model under epsilon $eps"
        python -m baselines.paraphrase --remote_cpr "fina_fake_qattr_none_zero.json" --data_name mmlu_fina --data_size 400 --para_model eugenesiow/bart-paraphrase --epsilon $eps --gpt_model $gpt_model
    done
done