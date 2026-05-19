
import jax
import jax.numpy as jnp

class EvojaxEnv():

	def __init__(self,task,render_rate=1):
		self.task = task
		self.max_steps = task.max_steps
		self.obs_shape = task.obs_shape
		self.act_shape = task.act_shape
		self.test = task.test if hasattr(task,'test') else None
		self.multi_agent_training = task.multi_agent_training
		self.render_rate=render_rate
		self.set_log_keys()
	
	def reset(self, key): return self.task.reset(key)
	def step(self, state, action): return self.task.step(state,action)

	def run_episode(self,policy,policy_params,init_key=None,n_steps=None):
		#import jax
		#import jax.numpy as jnp
		step = jax.jit(self.task.step)
		get_action = jax.jit(policy.get_actions)
		state = jax.jit(self.task.reset)(jnp.array([(init_key or jax.random.PRNGKey(seed=0))]))
		policy_state = jax.jit(policy.reset)(state)
		policy_params = jnp.array([policy_params])
		states=[state]; actions=[]; rewards=[]; done=False
		for _ in range(n_steps or self.max_steps):
			if done: break
			action, policy_state = get_action(state, policy_params, policy_state)
			state, reward, done = step(state, action)
			states+=[state]; actions+=[action]; rewards+=[reward]
		return states,actions,rewards

	def render_episode(self,states,file_name):
		images = [ self.task.render(states[i],0) for i in range(len(states)) if i%self.render_rate==0]
		images[0].save(file_name+'.gif', save_all=True, append_images=images[1:], duration=40, loop=0)

	def set_log_keys(self):
		self.log_keys=('steps',)
		self.save_keys=('steps','key','state','obs')

	def log_states(self,states):
		return states.steps.sum(),
