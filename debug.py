import os
import argparse

from codes import config
from codes import dataset

def test_config():
    c = config.Config()
    print(c.__dict__)

def test_dataset():
    d = dataset.Dataset()
    print(d.image_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-C", action="store_true", help="Config test")
    parser.add_argument("--dataset", "-D", action="store_true", help="Dataset test")
    args = parser.parse_args()

    if args.config:
        test_config()

    if args.dataset:
        test_dataset()
    