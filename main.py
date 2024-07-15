import os, sys
import torch
from datetime import datetime

sys.path.append('src/')

from data import get_data_loaders
from logger import DualLogger
from params import parse_args, print_args
from train import train
from params import fix_random_seed_as

def main():
    args = parse_args()

    args.name = '-'.join([
        datetime.now().strftime('%Y_%m_%d-%H_%M_%S'), # TODO: add more args
        f'{args.sentence_transformer_type}',
        f'{args.score_fn}',
        f'{args.candidate_scope}',
        f'dropout_{args.fc_dropout}',
        f'margin_{args.margin_hinge}',
        f'weight_{args.weight_bce}',
        f'lr_{args.lr}',
        f'seed_{args.seed}',
        ])
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fix_random_seed_as(args.seed)

    # Create a logs directory and a subdirectory for this run
    log_dir = os.path.join('experiments', args.name)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir

    # Set up dual logging to a file in the new directory and terminal
    log_path = os.path.join(log_dir, "run.log")
    sys.stdout = DualLogger(log_path)
    sys.stderr = sys.stdout  # Redirect stderr to the same logger
    print_args(args)

    # Get data loaders
    train_loader, val_loader, test_loader, embed_size = get_data_loaders(args)
    args.embed_size = embed_size

    # train
    train(args, train_loader, val_loader, test_loader)

    # # evaluation
    # eval(args, test_loader)



if __name__ == '__main__':
    main()
    