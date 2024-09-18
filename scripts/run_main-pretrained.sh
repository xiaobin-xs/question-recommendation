#!/bin/bash

python main.py \
    --sentence-transformer-type 'pretrained' \
    --sentence-transformer-path 'sentence-transformers/paraphrase-MiniLM-L6-v2' \
    --seed 1234 \
    --max-history-len -1 \
    --batch-size 16 \
    --lstm-dropout 0.5 \
    --score-fn cosine \
    --fc-dropout 0.5 \
    --lr 0.001 \
    --epochs 20 \
    --margin-hinge 0.1 \
    --weight-bce 1.0