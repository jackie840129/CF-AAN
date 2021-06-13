# encoding: utf-8
from .MARS import MARS
from .DukeV import DukeV
from .dataset_loader import ImageDataset,VideoDataset

__factory = {
    'mars' : MARS,
    'dukev':DukeV
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
