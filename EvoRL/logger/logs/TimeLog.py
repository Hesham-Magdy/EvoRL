
#from collections import deque

class TimeLog():
	def __init__(self,interval=1,iters=[]):
		from collections import deque
		self.interval = interval or 1
		self.iters = tuple(iters)
		self._Logs_train, self._Logs_test, self._Logs_logging, self._Logs_writing= \
			deque(), deque(), deque(), deque()
		self._total_ask, self._total_tell, \
		self._total_eval, \
		self._total_train, self._total_test, \
		self._total_logging, self._total_writing, \
		self._total_time \
			= ([0]*8)

	def _cond(self,i):
		return (i>0 and i%self.interval==0) or (i in self.iters)

	def append(self,iter_n,_eval,ask=0,tell=0,test=False,logging=False,writing=False):
		if not self._cond(iter_n): return
		_Logs = self._Logs_logging if logging else self._Logs_writing if writing \
			else self._Logs_test if test else self._Logs_train
		_Logs.append((iter_n,*self._handle_time(_eval,ask,tell),))

	def _handle_time(self,_eval,ask=0,tell=0):
		return (_eval,ask,tell,) if bool(ask) else (_eval,)

	def compute_time(self):
		for log in self._Logs_train:
			_eval,ask,tell= log[1:]
			self._total_eval+=_eval
			self._total_ask+=ask
			self._total_tell+=tell
		for log in self._Logs_test: self._total_test+=log[1]
		for log in self._Logs_logging: self._total_logging+=log[1]
		for log in self._Logs_writing: self._total_writing+=log[1]
		self._total_train= self._total_eval+self._total_ask+self._total_tell
		self._total_time= self._total_train+self._total_test+self._total_logging+self._total_writing
		return self._total_time

	def _prec(self,time,total):
		return '{0:.2f}s, {1:.2f}%'.format(time,(time/total)*100)

	def __str__(self):
		if not self._total_time: self.compute_time()
		ret= ""
		ret+= ('Total time, {}\n'.format(self._prec(self._total_time,self._total_time)))
		ret+= ('	,Train time, {}\n'.format(self._prec(self._total_train,self._total_time)))
		ret+= ('	,	,Evaluation time, {}\n'.format(self._prec(self._total_eval,self._total_train)))
		ret+= ('	,	,Ask time, {}\n'.format(self._prec(self._total_ask,self._total_train)))
		ret+= ('	,	,Tell time, {}\n'.format(self._prec(self._total_tell,self._total_train)))
		ret+= ('	,Test time, {}\n'.format(self._prec(self._total_test,self._total_time)))
		if self._total_logging: \
		ret+= ('	,Logging time, {}\n'.format(self._prec(self._total_logging,self._total_time)))
		if self._total_writing: \
		ret+= ('	,Writing time, {}\n'.format(self._prec(self._total_writing,self._total_time)))
		for log in self._Logs_train:
			ret+= ('Iter:{0}, train:{1:.2f}s, eval:{2:.0f}%, ask:{3:.0f}%, tell:{4:.0f}%\n'
				.format(log[0],sum(log[1:]),*[i/sum(log[1:])*100 for i in log[1:]]))
		for log in self._Logs_test:
			ret+= ('Iter:{0}, test:{1:.2f}s\n'.format(log[0],log[1]))
		for log in self._Logs_logging:
			if log[1]<0.001: continue
			ret+= ('Iter:{0}, logging:{1:.2f}s\n'.format(log[0],log[1]))
		for log in self._Logs_writing:
			if log[1]<0.001: continue
			ret+= ('Iter:{0}, writing:{1:.2f}s\n'.format(log[0],log[1]))
		return ret
