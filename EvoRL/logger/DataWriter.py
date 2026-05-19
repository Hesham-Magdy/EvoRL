
class DataWriter():
	def __init__(self,directory,interval=1,iters=[],save_flag=True,save_population=False,save_rewards=False,save_env=False,env=None):
		self.interval= interval or 1
		self.iters= tuple(iters)
		self.directory= directory
		self.save_flag= save_flag
		self.save_population= save_population
		self.save_rewards= save_rewards
		self.save_env= save_env
		self._env_save_keys= env.save_keys \
			if env is not None and hasattr(env,'save_keys') and env.save_keys is not None \
			else []

	def _cond(self,i):
		return self.save_flag and ((i>0 and i%self.interval==0) or (i in self.iters))

	def write(self,iter_n,population,normalization,rewards,env_states):	
		if not self._cond(iter_n): return
		from numpy import save
		from dataclasses import asdict
		from os import makedirs
		_dir= self.directory+'/'+str(iter_n)
		makedirs(_dir, exist_ok=True)
		save(_dir+'/best.npy',population[rewards.argmax()] if len(population.shape)==2 else population)
		save(_dir+'/obs_norm.npy',normalization)
		if self.save_rewards: save(_dir+'/rewards.npy',rewards)
		if self.save_population: save(_dir+'/population.npy',population)
		if self.save_env:
			env_states=asdict(env_states)
			for i in self._env_save_keys: save(_dir+'/env_'+i+'.npy',env_states[i])
