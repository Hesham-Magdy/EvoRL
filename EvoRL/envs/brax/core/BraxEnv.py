
#import jax
#import jax.numpy as jnp
#from flax.struct import dataclass

#from evojax.task.base import VectorizedTask
#from evojax.task.base import TaskState
#from evojax import ObsNormalizer

#from brax.envs import create
#from brax.envs import State as BraxState
#from brax.io import html

from flax.struct import dataclass
from evojax.task.base import TaskState
from brax.envs import State as BraxState
import jax.numpy as jnp
@dataclass
class State(TaskState):
	state: BraxState
	obs: jnp.ndarray

from evojax.task.base import VectorizedTask
class BraxEnv(VectorizedTask):

	def reset(self, key): return self._reset(key)
	def step(self,state,action): return self._step(state, action)

	def __init__(
		self,
		env_name: str,
		max_steps: int,
		backend: str = 'spring',
		action_repeat: int = 1,
		episodic: bool = True,
		auto_reset: bool = True,
		test: bool = False,
		**kwargs,
	):
		import jax
		#from brax.envs import create
		self.env_name= env_name
		self.max_steps= max_steps
		#self.physics_backend= backend
		self.test= test
		#self.env= create(env_name=self.env_name, episode_length=self.max_steps, backend=self.physics_backend)
		self.env= self._create(
			env_name=env_name,
			backend=backend,
			is_episodic= episodic,
			episode_length= max_steps,
			action_repeat= 1,
			auto_reset= auto_reset,
			is_vectorized= False,
			**kwargs,
		)
		self.obs_shape = tuple([self.env.observation_size, ])
		self.act_shape = tuple([self.env.action_size, ])
		self._reset=jax.jit(jax.vmap(self._init_reset()))
		self._step=jax.jit(jax.vmap(self._init_step()))

	def _init_reset(self):
		ENV=self.env
		def _reset(key):
			state = ENV.reset(key)
			return State(
				state=state,
				obs=state.obs
			)
		return _reset

	def _init_step(self):
		ENV=self.env
		def _step(state, action):
			brax_state = ENV.step(state.state, action)
			return \
				State(
					state=brax_state,
					obs=brax_state.obs
					), brax_state.reward, brax_state.done
		return _step

	def _create(
		self,
		env_name,
		backend,
		is_episodic= False,
		episode_length= 1000, 
		action_repeat= 1,
		auto_reset= False,
		is_vectorized= False,
		n_envs= None,
		**kwargs,
	):
		#from brax.envs import create
		#return create(
		#	env_name= env_name, 
		#	backend= backend,
		#	episode_length= episode_length if is_episodic else None,
		#	action_repeat= action_repeat if is_episodic else 1,
		#	auto_reset= auto_reset,
		#	batch_size= n_envs if is_vectorized else None,
		#	**kwargs,
		#)
		from brax.envs import get_environment
		from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper, AutoResetWrapper
		env= get_environment(env_name=env_name, backend=backend, **kwargs)
		if is_episodic:
			env= EpisodeWrapper(env, episode_length, action_repeat)
		if is_vectorized:
			env= VmapWrapper(env, n_envs)
		if auto_reset:
			env= AutoResetWrapper(env)
		return env


	def run_episode(self,policy,policy_params,init_key=None,n_steps=None):
		import jax
		import jax.numpy as jnp
		from evojax import ObsNormalizer
		normalizer= ObsNormalizer(obs_shape=self.obs_shape)
		normalize= jax.jit(normalizer.normalize_obs)
		normalize_params= normalizer.get_init_params()
		step= jax.jit(self._init_step())
		get_action= jax.jit(policy.get_actions)		
		state= jax.jit(self._init_reset())(init_key or jax.random.PRNGKey(seed=0))
		policy_state= jax.jit(policy.reset)(state)
		policy_params= jnp.array([policy_params])
		states, actions, rewards = [state], [], []
		for _ in range(n_steps or self.max_steps):
			state = state.replace(obs=normalize(jnp.array([state.obs]), normalize_params))
			action, policy_state = get_action(state, policy_params, policy_state)
			state, reward, done= step(state, action[0])
			states+=[state]; actions+=[action]; rewards+=[reward]
		return states,actions,rewards

	def render_episode(self,states,file_name):
		from brax.io import html
		with open(file_name+'.html','w') as _file:
			_file.write(html.render(
				#self.env.sys.replace(dt=self.env.dt),
				self.env.sys.tree_replace({'opt.timestep': self.env.dt}),
				[state.state.pipeline_state for state in states],
				colab=True,
			))
