#!/usr/bin/env bash
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT=$(dirname ${SCRIPTS_DIR})

data_config=$1
if [ ! -e ${data_config} ]; then
    echo "No data config file specified; Exiting."
    exit 1
fi
source ${data_config}


if [ ! -d "${vocab_dir}" ]; then
    mkdir -p ${vocab_dir}
fi

module=$2
if [ -z ${module} ]; then
    echo "No module mentioned for executing; Exiting."
    exit 1
fi


additional_args=${@:3}

export CMD="python ${PROJECT_ROOT}/src/preprocessing/${module}/tsv_to_tfrecords.py \
--text_in_files ${text_in_files} \
--out_dir ${vocab_dir}"

echo ${CMD}
eval "${CMD}"





