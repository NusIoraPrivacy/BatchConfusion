#!/bin/bash

data_names=("legal-qa-v1" "medical_o1_reasoning_SFT" "mmlu_fina")
modes=("full" "compress")
# modes=("full")
# alpha_ratios=(2 5 10)
alpha_ratios=(10)
sample_alphas=(0.1 1 10)
gpt_models=("gpt-3.5-turbo" "gpt-4o-mini" "gpt-4o")
sample_sizes=(2500 5000 7500 10000 12500 15000)

for size in ${sample_sizes[@]}
do
    echo "sampling size $size under full query"
    python -m evaluation.cluster_template --file_name "fake_qattr_random_Qwen2.5-0.5B-Instruct_10.json" --data_name mix --sample_size $size --sent_transformer 'paraphrase-distilroberta-base-v1' --data_mode full --alpha 10
done

for size in ${sample_sizes[@]}
do
    echo "sampling size $size under compress query"
    python -m evaluation.cluster_template --file_name "fake_cattr_random_Qwen2.5-0.5B-Instruct_10.json" --data_name mix --sample_size $size --sent_transformer 'paraphrase-distilroberta-base-v1' --data_mode compress --alpha 10
done

for alpha in ${alpha_ratios[@]}
do
    for mode in ${modes[@]}
    do
        if [$mode == "full"]; then
            attr_str="qattr"
        else
            attr_str="cattr"
        fi
        for sample_alpha in ${sample_alphas[@]}
        do
            for gpt_model in ${gpt_models[@]}
            do
                echo "test query cost and time for mix dataset under $mode query, $alpha, sampling alpha $sample_alpha, gpt model $gpt_model"
                python -m evaluation.cluster_template --file_name "fake_${attr_str}_random_Qwen2.5-0.5B-Instruct_$alpha.json" --data_name mix --sample_size 5000 --sent_transformer 'paraphrase-distilroberta-base-v1'  --gpt_model $gpt_model --data_mode $mode --alpha $sample_alpha
            done
        done
    done
done

# for alpha in ${alpha_ratios[@]}
# do
#     echo "test query cost and time for legal-qa-v1 under full query, $alpha"
#     python -m evaluation.cluster_template --file_name "fake_qattr_random_Qwen2.5-0.5B-Instruct_$alpha.json" --data_name legal-qa-v1 --fake_key "fake attributes question" --priv_key "filtered private attributes" --query_key question --resp_key response
# done

# for alpha in ${alpha_ratios[@]}
# do
#     echo "test query cost and time for legal-qa-v1 under compression, $alpha"
#     python -m evaluation.cluster_template --file_name "fake_cattr_random_Qwen2.5-0.5B-Instruct_$alpha.json" --data_name legal-qa-v1 --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --resp_key response
# done

# for alpha in ${alpha_ratios[@]}
# do
#     echo "test query cost and time for medical_o1_reasoning_SFT under full query, $alpha"
#     python -m evaluation.cluster_template --file_name "fake_qattr_random_Qwen2.5-0.5B-Instruct_$alpha.json" --data_name medical_o1_reasoning_SFT --fake_key "fake attributes question" --priv_key "filtered private attributes question" --query_key question --resp_key response
# done

# for alpha in ${alpha_ratios[@]}
# do
#     echo "test query cost and time for medical_o1_reasoning_SFT under compression, $alpha"
#     python -m evaluation.cluster_template --file_name "fake_cattr_random_Qwen2.5-0.5B-Instruct_$alpha.json" --data_name medical_o1_reasoning_SFT --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --resp_key response
# done

# for alpha in ${alpha_ratios[@]}
# do
#     echo "test query cost and time for mmlu_fina under full query, $alpha"
#     python -m evaluation.cluster_template --file_name "fake_qattr_random_Qwen2.5-0.5B-Instruct_$alpha.json" --data_name mmlu_fina --fake_key "fake attributes question" --priv_key "filtered private attributes question" --query_key question --resp_key origin_pred
# done

# for alpha in ${alpha_ratios[@]}
# do
#     echo "test query cost and time for mmlu_fina under compression, $alpha"
#     python -m evaluation.cluster_template --file_name "fake_cattr_random_Qwen2.5-0.5B-Instruct_$alpha.json" --data_name mmlu_fina --fake_key "fake attributes compression" --priv_key "filtered private attributes compression" --query_key compression --resp_key compress_pred
# done