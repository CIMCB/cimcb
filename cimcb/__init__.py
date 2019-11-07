from .__version__ import version as __version__

# To ignore TensorFlow Depreciation Warnings
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Use Theano
import os
os.environ["KERAS_BACKEND"] = "theano"
import keras
import keras.backend

from . import bootstrap
from . import cross_val
from . import model
from . import plot
from . import utils

__all__ = ["bootstrap", "cross_val", "model", "plot", "utils"]
