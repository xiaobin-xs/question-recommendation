# ConceptQA-Question-Recommendation

Example script: use default hyperparameters

```bash
python main.py \
    --sentence-transformer-type finetuned \
    --sentence-transformer-path sentence-transformer/embedding_model_tuned/ \
    --seed 1024 \
    --max-history-len -1 \
    --batch-size 16 \
    --lstm-dropout 0.5 \
    --score-fn cosine \
    --fc-dropout 0.5 \
    --lr 0.001 \
    --epochs 20 \
    --margin-hinge 0.2 \
    --weight-bce 1.0
```

Example script: sweep through hyperparameters

```bash
for dropout in "${dropout_values[@]}"
do
    for margin_hinge in "${margin_hinge_values[@]}"
    do
        for weight_bce in "${weight_bce_values[@]}"
        do
            echo "Running with dropout=$dropout, margin-hinge=$margin_hinge, weight-bce=$weight_bce"
            python main.py \
                --sentence-transformer-type finetuned \
                --sentence-transformer-path sentence-transformer/embedding_model_tuned/ \
                --seed 1024 \
                --max-history-len -1 \
                --batch-size 16 \
                --lstm-dropout $dropout \
                --score-fn cosine \
                --fc-dropout $dropout \
                --lr 0.001 \
                --epochs 30 \
                --margin-hinge $margin_hinge \
                --weight-bce $weight_bce
        done
    done
done
```