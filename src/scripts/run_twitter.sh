#!/bin/bash

models=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct")
sample_multipliers=(5)

# for model in ${models[@]}
# do
#     model_short_name="${model##*/}"
#     for sample_mul in ${sample_multipliers[@]}
#     do
#         python -m train_local.generator_v3 --out_file_name "fake_cattr_random_${model_short_name}_${sample_mul}.json" --sample_mul $sample_mul --fake_key "fake attributes text" --priv_key "filtered private attributes text" --query_key text --data_name twitter --in_file_name gender_dataset_fake_qattr.json --model_name $model --max_length 100
#     done
# done

for model in ${models[@]}
do
    python -m attack.pjd_attack --in_file_name "fake_attr_random_0.5.json" --model_name $model --fake_key "fake attributes text" --priv_key "filtered private attributes text" --query_key text --data_name twitter --sample_top_k 0 --epochs 10 --save_model
done