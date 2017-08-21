#!/usr/bin/env bash
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT=$(dirname ${SCRIPTS_DIR})
data_config=$1
if [ ! -e ${data_config} ]; then
    echo "No data config file specified; Exiting."
    exit 1
fi

model_config=$2
if [ ! -e ${model_config} ]; then
    echo "No data config file specified; Exiting."
    exit 1
fi

module=$3
if [ -z ${module} ]; then
    echo "No module mentioned for executing; Exiting."
    exit 1
fi


use_gpu=$4
if [ -z ${use_gpu} ]; then
    echo "Since there is no explicit mention of the GPU !!! The program will run by default"
    additional_args=${@:4}
else
    additional_args=${@:5}
fi

source ${data_config}
source ${model_config}
source ${PROJECT_ROOT}/config/lample-embeddings.conf
timestamp=`date +%Y-%m-%d-%H-%M-%S`
export OUT_DIR=${PROJECT_ROOT}/saved_models/${module}/${MODEL_NAME}/${timestamp}


if [[ "$text_train" != "" ]]; then
    additional_args=" --text_train=$text_train $additional_args "
fi
if [[ "$text_dev" != "" ]]; then
    additional_args=" --text_dev=$text_dev $additional_args "
fi
if [[ "$text_test" != "" ]]; then
    additional_args=" --text_test=$text_test $additional_args "
fi
if [[ "$text_in_files" != "" ]]; then
    additional_args=" --text_in_files=$text_in_files $additional_args "
fi

if [[ "$save_model" != "" ]]; then
    additional_args=" --save_model=$save_model $additional_args "
fi

if [[ "$epsilon" != "" ]]; then
    additional_args=" --epsilon=$epsilon $additional_args "
fi

if [[ "$max_steps" != "" ]]; then
    additional_args=" --max_steps $max_steps $additional_args "
fi

if [[ "$max_steps" != "" ]]; then
    additional_args=" --max_steps $max_steps $additional_args "
fi

if [[ "$embeddings" != "" ]]; then
    additional_args=" --embeddings $embeddings $additional_args "
fi





if [[ "$max_decrease_epochs" != "" ]]; then
    additional_args=" --max_decrease_epochs $max_decrease_epochs $additional_args "
fi

export CMD="python ${PROJECT_ROOT}/src/train/${module}/train.py \
--vocab_dir=$vocab_dir \
--optimizer=$optimizer \
--loss_type=$loss_type \
--lr=$lr \
--l2_weight=$l2_weight \
--word_dropout=$word_dropout \
--lstm_dropout=$lstm_dropout \
--final_dropout=$final_dropout \
--clip_norm=$clip_norm \
--token_dim=$token_dim \
--mention_dim=$mention_dim \
--entity_dim=$entity_dim \
--final_out_dim=$final_out_dim \
--lstm_dim=$lstm_dim \
--embed_dim=$embed_dim \
--text_epochs=$text_epochs \
--neg_samples=$neg_samples \
--random_seed=$random_seed \
--non_linearity $non_linearity \
--text_batch $text_batch \
--threshold $threshold \
$additional_args"


OUT_LOG="${OUT_DIR}"
echo "Writing to $OUT_LOG"

if [ -z ${use_gpu} ]; then
    CMD="${CMD} --logdir=$OUT_LOG"
else
    CMD="CUDA_VISIBLE_DEVICES=0 ${CMD} --logdir=$OUT_LOG"
fi

echo ${CMD}
mkdir -p ${OUT_LOG}
echo ${CMD} > ${OUT_LOG}/run.cmd

eval "${CMD} 2>&1 | tee ${OUT_LOG}/train.log"
