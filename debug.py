import os
import argparse

from image_classification import config
from image_classification import dataset
from image_classification import model
from image_classification import solver

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
    _model = m.get_model(2)
    print(_model)
    print("=" * 80)
    print(_model.state_dict().keys())
    print("=" * 80)
    print("The number of total parameters: {}".format(m._num_total_params(_model)))
    print("The number of trainable parameters: {}".format(m._num_trainable_params(_model)))
    

def test_solver():
    s = solver.Solver()
    print(s.val_dataloader.__dict__)
    
def test_training():
    s = solver.Solver(num_epochs=2, gpu_number=7)
    s.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-C", action="store_true", help="Config test")
    parser.add_argument("--dataset", "-D", action="store_true", help="Dataset test")
    parser.add_argument("--model", "-M", action="store_true", help="Model test")
    parser.add_argument("--solver", "-S", action="store_true", help="Solver test")
    parser.add_argument("--traintest", "-T", action="store_true", help="Training test")
    args = parser.parse_args()

    if args.config:
        test_config()

    if args.dataset:
        test_dataset()

    if args.model:
        test_model()

    if args.solver:
        test_solver()
    
    if args.traintest:
        test_training()