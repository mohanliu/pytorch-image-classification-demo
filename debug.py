import os
import argparse
import logging

from image_classification.helpers import Formatter
from image_classification import config
from image_classification import dataset
from image_classification import model
from image_classification import solver

# Logging setup
if not os.path.exists(os.path.join(".", "logs")):
    os.makedirs(os.path.join(".", "logs"))

# create logger 
logger = logging.getLogger('image_classification')
logger.setLevel(logging.INFO)

# create file handler which logs even debug messages
fh = logging.FileHandler(os.path.join(".", "logs", "debug.log"))
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
fh.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
ch.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

def test_config():
    c = config.Config()
    print(c.__dict__)

def test_dataset():    
    dl = dataset.FTDataLoader(batch_size=2)
    logger.info(dl.dataloader.__dict__)
    logger.info(dl._size)
    logger.info(dl._classes)
    logger.info(dl.image_dataset.data_location)
    logger.info(dl.image_dataset[0])

def test_model():
    m = model.FineTuneModel()
    _model = m.get_model(2)
    logger.info(_model)
    logger.info("=" * 80)
    logger.info(_model.state_dict().keys())
    logger.info("=" * 80)
    logger.info("The number of total parameters: {}".format(m._num_total_params(_model)))
    logger.info("The number of trainable parameters: {}".format(m._num_trainable_params(_model)))

def test_solver():
    s = solver.Solver()
    logger.info(s.val_dataloader.__dict__)
    
def test_training():
    s = solver.Solver(num_epochs=5, gpu_number=[6], lr_scheduler={
            "__name__": "step_lr",
            "step_size": 1,
            "gamma": 0.1
        })
    s.train()
    
def test_evaluation():
    s = solver.Solver(gpu_number=7)
    s.evaluate(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-C", action="store_true", help="Config test")
    parser.add_argument("--dataset", "-D", action="store_true", help="Dataset test")
    parser.add_argument("--model", "-M", action="store_true", help="Model test")
    parser.add_argument("--solver", "-S", action="store_true", help="Solver test")
    parser.add_argument("--traintest", "-T", action="store_true", help="Training test")
    parser.add_argument("--evaluate", "-E", action="store_true", help="Evalutation Test")
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
        
    if args.evaluate:
        test_evaluation()