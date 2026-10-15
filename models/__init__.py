from __future__ import absolute_import

from models.PPL import ProgressivePartialLearning
from models.ResNet50_PCB import ResNet50_PCB

__factory = {
    'PPL': ProgressivePartialLearning,
    'PCB': ResNet50_PCB,
}

def get_names():
    return __factory.keys()

def init_model(name,*args,**kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model :{}".format(name))
    return __factory[name](*args,**kwargs)