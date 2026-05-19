
from .core.EvojaxEnv import EvojaxEnv

from evojax.task.cartpole import CartPoleSwingUp

class CartPole():

    def __init__(self,params):
        self.episode_length=params['episode_length']
        self.hard=params['hard']

    def __call__(self):
        #from evojax.task.cartpole import CartPoleSwingUp
        #from .core.EvojaxEnv import EvojaxEnv
        return \
            EvojaxEnv(CartPoleSwingUp(max_steps=self.episode_length,harder=self.hard,test=False),render_rate=2), \
            EvojaxEnv(CartPoleSwingUp(max_steps=self.episode_length,harder=self.hard,test=True),render_rate=2), \
            EvojaxEnv(CartPoleSwingUp(max_steps=self.episode_length,harder=self.hard,test=True),render_rate=2)
