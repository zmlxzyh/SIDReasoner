#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-12340}"

# Preserve the original SIDReasoner launch defaults while allowing the A800
# server environment to override each transport setting.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
DATA_DIR="${DATA_DIR:-./data/EHR/mimic3_icd_name_path_0.1}"
DATASET_PREFIX="${DATASET_PREFIX:-mimic3_icd}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_dir/mimic3_icd_name_path_0.1_simple_sft_Qwen3-1.7B}"
WANDB_PROJECT="${WANDB_PROJECT:-SIDReasoner_EHR}"
RUN_NAME="${RUN_NAME:-mimic3_icd_name_path_0.1_simple_sft_Qwen3-1.7B}"

BATCH_SIZE="${BATCH_SIZE:-1024}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
CUTOFF_LEN="${CUTOFF_LEN:-1024}"
SAMPLE="${SAMPLE:--1}"
SEED="${SEED:-42}"

LOG_DIR="${LOG_DIR:-./logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.txt}"
mkdir -p "${LOG_DIR}" "$(dirname "${LOG_FILE}")" "$(dirname "${OUTPUT_DIR}")"

{
    echo "Starting EHR SimpleSFT"
    echo "model=${BASE_MODEL} data_dir=${DATA_DIR} prefix=${DATASET_PREFIX}"
    echo "gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} output=${OUTPUT_DIR}"

    torchrun \
        --nproc_per_node "${NPROC_PER_NODE}" \
        --master_port "${MASTER_PORT}" \
        sft_Qwen3_ehr.py \
        --base_model "${BASE_MODEL}" \
        --data_dir "${DATA_DIR}" \
        --dataset_prefix "${DATASET_PREFIX}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --micro_batch_size "${MICRO_BATCH_SIZE}" \
        --num_epochs "${NUM_EPOCHS}" \
        --learning_rate "${LEARNING_RATE}" \
        --cutoff_len "${CUTOFF_LEN}" \
        --sample "${SAMPLE}" \
        --seed "${SEED}" \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_run_name "${RUN_NAME}" \
        --train_from_scratch False \
        --mask_assistant True \
        --train_new_token_embeddings_only False
} > "${LOG_FILE}" 2>&1

echo "EHR SimpleSFT finished. Log: ${LOG_FILE}"
