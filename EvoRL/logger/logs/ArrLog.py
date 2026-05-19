
#from collections import deque
#from numpy import array

class ArrLog():
	def __init__(self,interval=1,iters=[]):
		from collections import deque
		self.interval = interval or 1
		self.iters = tuple(iters)
		self._Logs = deque()

	def _cond(self,i):
		return (i>0 and i%self.interval==0) or (i in self.iters)

	def append(self,iter_n,arr):
		if not self._cond(iter_n): return
		self._Logs.append((iter_n,*self._handle_array(arr),))

	def _handle_array(self,arr):
		return (
			float(arr.max()),
			float(arr.mean()),
			float(arr.min()),
			float(arr.std()),
		)

	def get_array(self):
		from numpy import array
		return array(self._Logs)

	def __str__(self):
		if len(self._Logs)==0: return ""
		ret= ""
		for log in self._Logs:
			ret+= ('Iter:{0}, max:{1}, mean:{2}, min:{3}, std:{4}\n'.format(*log))
		return ret
