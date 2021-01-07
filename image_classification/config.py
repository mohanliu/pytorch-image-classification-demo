import os

class Config(object):
    def __init__(self, **kwargs):
        self._homedir = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        )
        
        # Training Data path
        self._datapath = os.path.join(
            self._homedir, 
            kwargs.get("datapath", "hymenoptera_data")
        )
        
        # Model backbone
        self._model_backbone = "resnet18"
        self._pretrain = True

        # Data Loader configs
        self._batch_size = kwargs.get("batch_size", 16)
        self._shuffle = kwargs.get("shuffle", True)
        self._num_worker = kwargs.get("num_worker", 0)

        # Optimization params
        self._num_epochs = kwargs.get("num_epochs", 25)
        self._learning_rate = kwargs.get("learning_rate", 0.001)
        self._momentum = kwargs.get("momentum", 0.9)
        self._lr_scheduler_dict = kwargs.get("lr_scheduler", {
            "__name__": "step_lr",
            "step_size": 7,
            "gamma": 0.1
        })
        
        # Output file
        self._snapshot_folder = os.path.join(
            self._homedir,
            kwargs.get("snapshot_folder", "snapshots")
        )
        self._results_folder = os.path.join(
            self._homedir,
            kwargs.get("result_folder", "results")
        )
