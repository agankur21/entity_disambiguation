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

source ${data_config}
source ${model_config}
source ${PROJECT_ROOT}/config/lample-embeddings.conf
additional_args=${@:4}
timestamp=`date +%Y-%m-%d-%H-%M-%S`
OUT_LOG=${PROJECT_ROOT}/hyperparams/tune-${module}-${MODEL_NAME}-${timestamp}

##########################

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

if [[ "$epsilon" != "" ]]; then
    additional_args=" --epsilon=$epsilon $additional_args "
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

##########################

echo "Writing to "${OUT_LOG}

num_gpus=64

lrs="0.0005 0.001 0.01"
l2s="0.0 0.00005 0.0005"
word_dropouts="0.85"
lstm_dropouts="0.85"
final_dropouts="1"
mention_dims="50 100 150"
entity_dims="50 100 150"
final_out_dims="50 100 150"
lstm_dims="1024"
embed_dims="50 100 150"
l_text_epochs="50"
text_batches="64 128"
l_neg_samples="10 20 40"
non_linearities="relu tanh"
clip_norms="10"

# array to hold all the commands we'll distribute
declare -a commands

for lr in ${lrs[@]}; do
    for l2 in ${l2s[@]}; do
        for word_dropout in ${word_dropouts[@]}; do
            for lstm_dropout in ${lstm_dropouts[@]}; do
                for final_dropout in ${final_dropouts[@]}; do
                    for mention_dim in ${mention_dims[@]}; do
                        for entity_dim in ${entity_dims[@]}; do
                            for final_out_dim in ${final_out_dims[@]}; do
                                for lstm_dim in ${lstm_dims[@]}; do
                                    for embed_dim in ${embed_dims[@]}; do
                                        for text_epochs in ${l_text_epochs[@]}; do
                                            for text_batch in ${text_batches[@]}; do
                                                for neg_samples in ${l_neg_samples[@]}; do
                                                    for non_linearity in ${non_linearities[@]}; do
                                                        for clip_norm in ${clip_norms[@]}; do
                                                            fname_append="lr_$lr-l2_$l2-wdrop_$word_dropout-ldrop_$lstm_dropout-fdrop_$final_dropout-tdim_$token_dim-mdim_$mention_dim-edim_$entity_dim-fdim_$final_out_dim-ldim_$lstm_dim-embeddim_$embed_dim-epoch_$text_epochs-batch_$text_batch-samples_$neg_samples-nonlin_$non_linearity-clipnorm_$clip_norm"
                                                            if [[ ! -e ${OUT_LOG}/train-${fname_append}.log ]]; then
                                                                commands+=("srun --gres=gpu:1 --partition=titanx-short python ${PROJECT_ROOT}/src/train/${module}/train.py \
                                                                    --vocab_dir=$vocab_dir \
                                                                    --optimizer=$optimizer \
                                                                    --loss_type=$loss_type \
                                                                    --lr=$lr \
                                                                    --l2_weight=$l2 \
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
                                                                    $additional_args &> $OUT_LOG/train-$fname_append.log")
                                                            fi
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
echo $commands
mkdir -p ${OUT_LOG}
# now distribute them to the gpus
num_jobs=${#commands[@]}
if [[ $num_jobs -lt $num_gpus ]]; then
    jobs_per_gpu=1
    num_gpus=$num_jobs
else
    jobs_per_gpu=$((num_jobs / num_gpus))
fi
echo "Distributing $num_jobs jobs to $num_gpus gpus ($jobs_per_gpu jobs/gpu)"

j=0
for (( gpuid=0; gpuid<num_gpus; gpuid++)); do
    for (( i=0; i<jobs_per_gpu; i++ )); do
        jobid=$((j * jobs_per_gpu + i))
        comm="${commands[$jobid]}"
        comm=${comm/XX/$gpuid}
        echo ${comm}
        eval ${comm}
    done &
    j=$((j + 1))
done
