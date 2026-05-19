
from .MLP import MLP
from .ActivationFunctions import ActivationFunctions
from numpy import prod as np_prod

class MLPBuilder():
    def __init__(self,params):
        self.layers=[int(i) for i in params['layers'].split('~')]
        self.hidden_activation=params['hidden_activation']
        self.output_activation=params['output_activation']

    def __call__(self, obs_shape,act_shape, seed):
        #from numpy import prod as np_prod
        #from .ActivationFunctions import ActivationFunctions
        return MLP(        
            input_dim=np_prod(obs_shape),
            output_dim=np_prod(act_shape),
            hidden_dims=self.layers,
            hidden_actv=ActivationFunctions()[self.hidden_activation],
            output_actv=ActivationFunctions()[self.output_activation],
            seed=seed,
        )
