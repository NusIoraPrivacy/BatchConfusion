#!/bin/bash

dis_models=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct")
epsilons=(1 3 5)

for dis_model in ${dis_models[@]}
do
    for epsilon in ${epsilons[@]}
    do
        echo "Attribute inference attack for paraphrase method on discrimination model $dis_model, epsilon $epsilon"
        python -m attack.atr_attack_dp --in_file_name "gender_dataset.json" --model_name $dis_model  --epochs 20 --dp_method paraphrase --epsilon $epsilon
    done
done

for dis_model in ${dis_models[@]}
do
    for epsilon in ${epsilons[@]}
    do
        echo "Attribute inference attack for custext method on discrimination model $dis_model, epsilon $epsilon"
        python -m attack.atr_attack_dp --in_file_name "gender_dataset.json" --model_name $dis_model  --epochs 20 --dp_method custext --epsilon $epsilon
    done
done

for dis_model in ${dis_models[@]}
do
    echo "Attribute inference attack non private method on discrimination model $dis_model, epsilon $epsilon"
    python -m attack.atr_attack_dp --in_file_name "gender_dataset.json" --model_name $dis_model  --epochs 20 --dp_method paraphrase --epsilon $epsilon
done