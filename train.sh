#!/bin/bash
set -x
DATASET_NAME="agent_sft_act_dataset"

export PYTHONPATH=`pwd`
echo ${PYTHONPATH}

cd ./training_scripts

WANDB_PROJECT=""

MODEL_TYPE="7b"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"


DATASET_DIR="../generation/multiwoz/converters/woz.2.2.gen/"

LR=2e-5
BATCH_SIZE=4
EPOCH=1

TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.${DATASET_DIR}"

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  --master_port=1201 \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --model_name "${MODEL_NAME}" \
  --dataset "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --save_model \
  --pure_bf16 \
  --output_dir "./${DATASET_NAME}.${TAG}"/ \
  --lr ${LR} \
  --valid_batch_size ${BATCH_SIZE} \
  --train_batch_size ${BATCH_SIZE} \
  --micro_batch_size ${BATCH_SIZE} \
  --num_epochs ${EPOCH} \
  --evaluation_steps 200 \
  --check_point_steps 1000000 \
  --wandb_name ${TAG} \
  --wandb_project "${WANDB_PROJECT}"

python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path "./${DATASET_NAME}.${TAG}/epoch_000" \
    --consolidated_model_path "./${DATASET_NAME}.${TAG}/epoch_000.hf" \
    --HF_model_path_or_name "meta-llama/Llama-2-7b-hf"

PRE_TRAIN_MODEL="./${DATASET_NAME}.${TAG}/epoch_000.hf"


# training on real data

DATASET_DIR="../generation/multiwoz/converters/woz.2.2.real/"

# train 2.2 with pre-train 8k
TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.${DATASET_DIR}"

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  --master_port=1201 \
  ./llama_finetuning.py \
  --enable_fsdp  \
  --model_name "${PRE_TRAIN_MODEL}" \
  --dataset "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --save_model \
  --pure_bf16 \
  --output_dir "./${DATASET_NAME}.${TAG}.Real"/ \
  --lr ${LR} \
  --valid_batch_size ${BATCH_SIZE} \
  --train_batch_size ${BATCH_SIZE} \
  --micro_batch_size ${BATCH_SIZE} \
  --num_epochs ${EPOCH} \
  --evaluation_steps 200 \
  --check_point_steps 1000000 \
  --wandb_name ${TAG} \
  --wandb_project "${WANDB_PROJECT}"

python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path "./models/${DATASET_NAME}.${TAG}.Real/epoch_000" \
    --consolidated_model_path "./models/${DATASET_NAME}.${TAG}.Real/epoch_000.hf" \
    --HF_model_path_or_name "meta-llama/Llama-2-7b-hf"