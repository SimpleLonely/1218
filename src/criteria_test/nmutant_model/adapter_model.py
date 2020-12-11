



from nmutant_model.network import EnsembleModel
from nmutant_model.model import Model
from keras.engine.sequential import Sequential


class AdapterModel(EnsembleModel):
    def __init__(self,model:Sequential,input_shape) -> None:
        self.model = model
        super().__init__(layers=model.layers,input_shape=input_shape)
        
    
   
    
