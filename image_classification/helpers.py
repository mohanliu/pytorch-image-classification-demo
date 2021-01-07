import os
import datetime
import pytz
import numpy as np
import logging
import json

class Formatter(logging.Formatter):
    """override logging.Formatter to use an aware datetime object"""
    def converter(self, timestamp):
        dt = datetime.datetime.fromtimestamp(timestamp)
        tzinfo = pytz.timezone("America/Chicago")
        return tzinfo.localize(dt)
        
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec='milliseconds')
            except TypeError:
                s = dt.isoformat()
        return s
    
def timeit(f):
    """Function decorator to time how long a function runs for
    Args:
        f: function to time
    Returns:
        result: output of function
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        eta = datetime.timedelta(seconds=(end - start))
        print("Elapsed time: {}".format(str(eta)))
        return result

    return wrapper
