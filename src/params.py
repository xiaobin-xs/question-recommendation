import argparse


def print_args(args):
    print("Running with the following configuration:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-folder",
        type=str,
        default="data/",
        help="path to the data folder",
    )

    parser.add_argument(
        "--raw-json-file",
        type=str,
        default='copilot_prod_interactions_2023-08-01_2024-05-13.json',
        help="batch size",
    )

    parser.add_argument(
        "--sentence-transformer-type",
        type=str,
        default='finetuned',
        choices=['finetuned', 'pretrained'],
        help="path to the fine tuned Sentence Transformer model",
    )

    parser.add_argument(
        "--sentence-transformer-path",
        type=str,
        default='sentence-transformer/embedding_model_tuned/', # 'sentence-transformers/paraphrase-MiniLM-L6-v2'
        help="local path to the fine tuned Sentence Transformer model; or the name of the pre-trained model from the sentence-transformers library",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed",
    )

    parser.add_argument(
        "--max-history-len",
        type=int,
        default=-1,
        help="maximum length of past history history. -1 means all history is used"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="batch size",
    )

    parser.add_argument(
        "--lstm-dropout",
        type=float,
        default=0.1,
        help="dropout rate for the LSTM",
    )

    parser.add_argument(
        "--score-fn",
        type=str,
        default='cosine',
        choices=['cosine', 'custom'],
        help="number of epochs",
    )

    parser.add_argument(
        "--fc-dropout",
        type=float,
        default=0.1,
        help="dropout rate for the fully connected layer in the Score module, if score_fn is 'custom'",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="number of epochs",
    )

    parser.add_argument(
        "--margin-hinge",
        type=float,
        default=0.1,
        help="margin for the hinge loss",
    )

    parser.add_argument(
        "--weight-bce",
        type=float,
        default=1.0,
        help="weight for the binary cross entropy loss",
    )

    # TODO: Add more arguments here


    args = parser.parse_args()

    return args