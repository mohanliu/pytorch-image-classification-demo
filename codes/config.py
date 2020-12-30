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