import os
import argparse

from codes import config
from codes import dataset
from codes import model
from codes import solver

def test_config():
    c = config.Config()
    print(c.__dict__)

def test_dataset():
    d = dataset.FTDataset(batch_size=2)
    print(d.image_dataset)
    print(d.dataloader.__dict__)
    print(d._size)
    print(d._classes)

def test_model():
    m = model.FineTuneModel()
    print(m.get_model(2))

def test_solver():
    s = solver.Solver()
    print(s.__dict__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-C", action="store_true", help="Config test")
    parser.add_argument("--dataset", "-D", action="store_true", help="Dataset test")
    parser.add_argument("--model", "-M", action="store_true", help="Model test")
    parser.add_argument("--solver", "-S", action="store_true", help="Solver test")
    args = parser.parse_args()

    if args.config:
        test_config()

    if args.dataset:
        test_dataset()

    if args.model:
        test_model()

    if args.solver:
        test_solver()
    