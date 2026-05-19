
#from .core.BraxEnv import BraxEnv
#from .core.brax_selector import brax_selector as selector

class BraxBuilder():
    def __init__(self,params):
        self.env_name = params['name']
        self.physics_backend= params['physics_backend']
        self.episode_length= params['episode_length']
        self.action_repeat= params['action_repeat']
        self.episodic= params['episodic']
        self.auto_reset= params['auto_reset']


    def __call__(self):
        from .core.BraxEnv import BraxEnv
        from .core.brax_selector import brax_selector as selector
        env_name= selector[self.env_name]
        return \
            BraxEnv(
                env_name= env_name, 
                max_steps= self.episode_length,
                backend= self.physics_backend,
                action_repeat= self.action_repeat,
                episodic= self.episodic,
                auto_reset= self.auto_reset,
                test=False,
            ),\
            BraxEnv(
                env_name= env_name, 
                max_steps= self.episode_length,
                backend= self.physics_backend,
                action_repeat= self.action_repeat,
                episodic= self.episodic,
                auto_reset= self.auto_reset,
                test=True,
            ),\
            BraxEnv(
                env_name= env_name, 
                max_steps= self.episode_length,
                backend= self.physics_backend,
                action_repeat= self.action_repeat,
                episodic= self.episodic,
                auto_reset= self.auto_reset,
                test=True,
            )
