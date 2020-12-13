from keras.backend.load_backend import backend
from nmutant_model.network import EnsembleModel
from nmutant_model.model import Model
from keras.engine.sequential import Sequential
import keras
from keras import backend as K
from keras.backend import tensorflow_backend
import tensorflow as tf

class AdapterModel(EnsembleModel):
    def __init__(self,model:Sequential,input_shape) -> None:
        self.model = model
        super().__init__(layers=model.layers,input_shape=input_shape)

    def fprop(self, x):
        session = tensorflow_backend.get_session()
        with session.graph.as_default():
            outputs = [layer.output for layer in self.layers_list]
        
        states = dict(zip(self.get_layer_names(), outputs))
        return states