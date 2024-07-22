import json
import os, sys
import torch
from datetime import datetime

sys.path.append('src/')

from data import get_data_loaders
from logger import DualLogger
from params import parse_args, print_args, save_args, fix_random_seed_as
from train import train

def main():
    args = parse_args()

    args.name = '-'.join([
        datetime.now().strftime('%Y_%m_%d-%H_%M_%S'), # TODO: add more args
        f'{args.sentence_transformer_type}',
        f'{args.score_fn}',
        f'{args.candidate_scope}',
        f'batch_{args.batch_size}',
        f'dropout_{args.fc_dropout}',
        f'margin_{args.margin_hinge}',
        f'weight_{args.weight_bce}',
        f'weight_sim_{args.weight_sim}',
        f'lr_{args.lr}',
        f'seed_{args.seed}',
        ]) + (f'_{args.comment}' if args.comment else '')
    
    fix_random_seed_as(args.seed)
    if 10 not in args.ks:
        args.ks.append(10)

    # Create a logs directory and a subdirectory for this run
    log_dir = os.path.join(args.root_dir, 'experiments', args.name)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir

    # Set up dual logging to a file in the new directory and terminal
    log_path = os.path.join(args.root_dir, log_dir, "run.log")
    sys.stdout = DualLogger(log_path)
    sys.stderr = sys.stdout  # Redirect stderr to the same logger
    print_args(args)

    # Get data loaders
    train_loader, val_loader, test_loader, embed_size = get_data_loaders(args)
    args.embed_size = embed_size
    
    # save args as a json file
    save_args(args)

    # train
    train(args, train_loader, val_loader, test_loader)



if __name__ == '__main__':
    main()
    