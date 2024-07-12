import os, sys
from datetime import datetime

from data import get_data_loaders
from logger import DualLogger
from params import parse_args, print_args

def main():
    args = parse_args()

    args.name = '-'.join([
        datetime.now().strftime("%Y_%m_%d-%H_%M_%S"), # TODO: add more args
        f"seed_{args.seed}"
        ])

    # Create a logs directory and a subdirectory for this run
    log_dir = os.path.join('experiments', args.name)
    os.makedirs(log_dir, exist_ok=True)

    # Set up dual logging to a file in the new directory and terminal
    log_path = os.path.join(log_dir, "run.log")
    sys.stdout = DualLogger(log_path)
    sys.stderr = sys.stdout  # Redirect stderr to the same logger
    print_args(args)

    train_loader, val_loader, test_loader = get_data_loaders(args)


if __name__ == '__main__':
    main()
    