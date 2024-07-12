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
        "--seed",
        type=int,
        default=1234,
        help="random seed",
    )

    parser.add_argument(
        "--max-history-len",
        default=None,
        help="maximum length of past history history. None means all history is used"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="batch size",
    )

    # TODO: Add more arguments here


    args = parser.parse_args()

    return args