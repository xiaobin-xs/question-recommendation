#!/bin/bash

# python main.py \
#     --sentence-transformer-type finetuned \
#     --sentence-transformer-path sentence-transformer/embedding_model_tuned/ \
#     --seed 1234 \
#     --max-history-len -1 \
#     --batch-size 16 \
#     --score-fn cosine \
#     --lr 0.001 \
#     --epochs 20 \
#     --margin-hinge 0.1 \
#     --weight-bce 1.0

python main.py \
    --sentence-transformer-type finetuned \
    --sentence-transformer-path sentence-transformer/embedding_model_tuned/ \
    --seed 1111 \
    --max-history-len -1 \
    --candidate-scope batch \
    --batch-size 16 \
    --lstm-dropout 0.5 \
    --score-fn custom \
    --fc-dropout 0.7 \
    --lr 0.001 \
    --epochs 50 \
    --patience 10 \
    --margin-hinge 0.1 \
    --weight-bce 1.0