#!/bin/bash

# Feel free to change this parameter
export CUDA_DEVICE_MAX_CONNECTIONS=1

# SET UP THE FOLLOWING LINES ACCORDING TO TARGET ENVIRONMENT
# GPUS_PER_NODE=____
# MASTER_ADDR=____
# MASTER_PORT=____
# NNODES=____
# NODE_RANK=____
# WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=./checkpoints
rm -rf $CHECKPOINT_PATH || true

VOCAB_FILE=./data/gpt2-vocab.json
MERGE_FILE=./data/gpt2-merges.txt
DATA_PATH=./data/my-gpt_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# DO NOT CHANGE PARAMETERS AFFECTING MODEL SIZE
# (num layers, hidden size etc)

# feel free to use optimizations like flash attention e.g.
# --use-flash-attn-triton \


GPT_ARGS="
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --moe-expert-parallel-size 1 \
    --num-experts 1 \
    --expert-interval 1 \
    --num-layers 64 \
    --hidden-size 4096 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --train-iters 500 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --no-gradient-accumulation-fusion
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 0
"

export PYTHONPATH="$(pwd)/Megatron-DeepSpeed:${PYTHONPATH}"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

