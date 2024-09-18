#!/bin/bash

# Define the possible values for each hyperparameter
dropout_values=(0.5)
margin_hinge_values=(0.3)
weight_bce_values=(1.0 5.0)
weight_sim_values=(0.1 0.5)
score_fn_values=("custom")

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
                    --comment "testSimLoss"\
                    --sentence-transformer-type pretrained \
                    --sentence-transformer-path sentence-transformers/paraphrase-MiniLM-L6-v2 \
                    --seed 1234 \
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
                    --weight-bce $weight_bce \
                    --weight-sim 0.1
            done
        done
    done
done
