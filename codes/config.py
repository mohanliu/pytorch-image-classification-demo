import os

class Config(object):
    def __init__(self, **kwargs):
        self._homedir = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))
        )

        self._datapath = os.path.join(
            self._homedir, 
            kwargs.get("datapath", "hymenoptera_data")
        )

        self._model_backbone = "resnet18"
        self._pretrain = True

        # Data Loader configs
        self._batch_size = kwargs.get("batch_size", 4)
        self._shuffle = kwargs.get("shuffle", True)
        self._num_worker = kwargs.get("num_worker", 0)

        # Optimization params
        self._num_epoch = kwargs.get("num_epoch", 25)
