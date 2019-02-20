#!/usr/bin/env bash

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

DATA_DIR=/media/deep/t6/datasets/insightface

python -u train_combinedloss.py \
--data-dir $DATA_DIR \
--batch-size 128 \
--gpus 4,5,6,7 \
-j 24 \
--num-epochs 40 \
--mode hybrid \
--margin1 1.0 \
--margin2 0.3 \
--margin3 0.2 \
--warmup-epochs 1 \
--warmup-lr 0