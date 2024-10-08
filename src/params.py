import json
import os
import argparse
import random
import torch
import numpy as np

if torch.cuda.is_available():
    DEVICE = 'cuda'
# elif torch.backends.mps.is_available():
#     DEVICE = 'mps' # some error with mps...
else:
    DEVICE = 'cpu'

def print_args(args):
    print("Running with the following configuration:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="additional comment of the experiment, will add to the end of the exp name",
    )
    
    parser.add_argument(
        "--root-dir",
        type=str,
        default="./",
        help="root directory of the project",
    )

    parser.add_argument(
        "--data-folder",
        type=str,
        default="data/",
        help="path to the data folder",
    )

    parser.add_argument(
        "--preprocessed-data-filename",
        type=str,
        default='chat_preprocessed',
        help="filename for the preprocessed data",
    )

    parser.add_argument(
        "--raw-json-file",
        type=str,
        default='',
        help="json file containing the raw data, empty string for all json files in the data folder",
    )

    parser.add_argument(
        "--device",
        default=DEVICE,
        # choices=['cuda', 'cpu', 'mps'],
        help="device to run the model on",
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
        "--candidate-scope",
        type=str,
        default='batch',
        choices=['own', 'batch'],
        help="scope of candidates to consider during learning; 'own' means only consider the candidates for each query, 'batch' means consider all candidates in the batch",
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
        default='custom',
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
        "--patience",
        type=int,
        default=20,
        help="patience for early stopping",
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

    parser.add_argument(
        "--weight-sim",
        type=float,
        default=0.1,
        help="weight for the similarity loss",
    )

    parser.add_argument(
        "--ks",
        type=list,
        default=[1, 3, 5, 10, 30, 50],
        help="list of k values for recall@k",
    )

    # TODO: Add more arguments here


    args = parser.parse_args()

    return args


def parse_log_dir():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="name of the experiment",
    )

    args = parser.parse_args()

    return args

def save_args(args):
    # save args as a json file
    with open(os.path.join(args.log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args(log_dir):
    with open(os.path.join('experiments', log_dir, 'args.json'), 'r') as f:
        loaded_args = json.load(f)
    args = argparse.Namespace(**loaded_args)
    return args

# code from https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch/blob/master/utils.py#L65
def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False