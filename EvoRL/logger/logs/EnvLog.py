
#from collections import deque

class EnvLog():
	def __init__(self,env,interval=1,iters=[]):
		from collections import deque
		self.interval = interval or 1
		self.iters = tuple(iters)
		self._Logs = deque()
		self._log_env_fn, self._env_log_keys= None, []
		if hasattr(env,'log_keys') and env.log_keys is not None:
			self._env_log_keys= ('Iter',*env.log_keys,); self._log_env_fn= env.log_states

	def _cond(self,i):
		return (i>0 and i%self.interval==0) or (i in self.iters)

	def append(self,iter_n,states):
		if not self._cond(iter_n) or self._log_env_fn is None: return
		self._Logs.append((iter_n,*self._log_env_fn(states),))

	def __str__(self):
		if len(self._Logs)==0: return ""
		ret= ""
		for log in self._Logs:
			for k, v in zip(self._env_log_keys,log):
				ret+=('{0}:{1}, '.format(k,str(v).replace(',',' ')))
			ret+=('\n')
		return ret
