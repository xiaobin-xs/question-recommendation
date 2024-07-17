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
# Define the possible values for each hyperparameter
dropout_values=(0.3 0.5 0.7)
margin_hinge_values=(0.1 0.3 0.5)
weight_bce_values=(0 1.0 5.0)
score_fn_values=("custom" "cosine")

# Loop through each combination of hyperparameters
for dropout in "${dropout_values[@]}"
do
    for margin_hinge in "${margin_hinge_values[@]}"
    do
        for weight_bce in "${weight_bce_values[@]}"
        do
            for score_fn in "${score_fn_values[@]}"
            do
                echo "Running with dropout=$dropout, margin-hinge=$margin_hinge, weight-bce=$weight_bce"
                python main.py \
                    --comment "newFirstHitRate"\
                    --sentence-transformer-type finetuned \
                    --sentence-transformer-path sentence-transformer/embedding_model_tuned/ \
                    --seed 1111 \
                    --max-history-len -1 \
                    --candidate-scope batch \
                    --batch-size 16 \
                    --lstm-dropout $dropout \
                    --score-fn $score_fn \
                    --fc-dropout $dropout \
                    --lr 0.001 \
                    --epochs 100 \
                    --patience 20 \
                    --margin-hinge $margin_hinge \
                    --weight-bce $weight_bce
            done
        done
    done
done
```