
#from .logs import ArrLog, TimeLog, EnvLog
#from .DataWriter import DataWriter
#from .Visualizer import Visualizer

#from jax import local_devices
#from yaml import dump
#from numpy import save
#from os import makedirs
from time import perf_counter as get_time

class Logger():
	def __init__(self,params,directory):
		from jax import local_devices
		from os import makedirs
		self._params_summary= "{0}:{1}, {2}:{3}, {4}:{5}, iterations:{6}".format(
			params['env']['name'], params['env']['episode_length'],
			params['algorithm']['name'], params['algorithm']['population'],
			params['network']['name'], params['network']['layers'],
			params['iterations'],
			)
		self._env_name= params['env']['name']
		self._algorithm_name= params['algorithm']['name']
		self._device_info= str(local_devices())[1:-1]
		self._init_params= params
		makedirs(directory, exist_ok=True)
		self._id= self.get_id(params['iterations'],directory)
		self.directory = directory + '/' + self._id
		self.last_iter= params['iterations']-1
		params= params['log']
		self.log_interval= params['log_interval']
		self.log_save= params['log_save']
		self.save_train= params['save_train']
		self.save_test= params['save_test']
		self.save_train_interval= params['save_train_interval']
		self.save_population= params['save_population']
		self.save_rewards= params['save_rewards']
		self.save_env= params['save_env']
		self.visualize_final_env= params['visualize_final_env']
		self.visualize_train_envs= params['visualize_train_envs']
		self.visualize_test_envs= params['visualize_test_envs']

	def __call__(self,algorithm,network,env_train,env_test,env_sim):
		from .logs import ArrLog, TimeLog, EnvLog
		from .DataWriter import DataWriter
		from .ChartDrawer import ChartDrawer
		from .Visualizer import Visualizer
		from time import perf_counter as get_time
		self.trains= ArrLog(interval=self.log_interval,iters=[0,self.last_iter])
		self.tests= ArrLog(interval=1,iters=[self.last_iter])
		self.times= TimeLog(interval=1,iters=[0,self.last_iter])
		self.env_trains= EnvLog(env=env_train,interval=self.log_interval,iters=[0,self.last_iter])
		self.env_tests= EnvLog(env=env_test,interval=1,iters=[self.last_iter])
		self.train_writer= DataWriter(directory= self.directory+'/data/train',
			interval=self.save_train_interval, iters=[self.last_iter],
			save_flag=self.save_train, save_population=self.save_population, save_rewards=self.save_rewards, 
			save_env=self.save_env, env=env_train)
		self.test_writer= DataWriter(directory= self.directory+'/data/test',
			interval=1, iters=[self.last_iter],
			save_flag=self.save_test, save_population=False, save_rewards=self.save_rewards, 
			save_env=self.save_env, env=env_test)
		self.chart_drawer= ChartDrawer()
		self.visualizer= Visualizer(env_sim,network,
			self.visualize_final_env,self.visualize_train_envs,self.visualize_test_envs)
		print(self._device_info)
		print(self._id+', '+self._params_summary)
		return self

	def log(self,iter_n,population,rewards,env_states,test=False):
		#from time import perf_counter as get_time
		start_time = get_time()
		if not test:
			self.trains.append(iter_n,rewards)
			self.env_trains.append(iter_n,env_states)
		else:
			self.tests.append(iter_n,rewards)
			self.env_tests.append(iter_n,env_states)
		self.times.append(iter_n,get_time()-start_time,logging=True)
		start_time = get_time()
		writer= self.train_writer if not test else self.test_writer
		writer.write(iter_n,population,rewards,env_states)
		self.times.append(iter_n,get_time()-start_time,writing=True)

	def log_time(self,iter_n,_eval=0,ask=0,tell=0,test=False):
		if not test:
			self.times.append(iter_n,_eval,ask,tell,test=test)
		else:
			self.times.append(iter_n,_eval,test=test)

	def end(self,best_solution,best_score):
		from yaml import dump
		from numpy import save
		from os import makedirs
		self._params_summary= \
			'{0}, {1}, reward:{2:.2f}, time:{3:.0f}s'.format(
				self._id,
				self._params_summary,
				best_score,
				self.times.compute_time(),
			)
		files= {
			'device_info.csv': self._device_info,
			'params.yaml': '#{0}\n{1}'.format(self._id, dump([self._init_params],indent=2,sort_keys=False)),
			'train.csv': str(self.trains),
			'test.csv': str(self.tests),
			'env_train.csv': str(self.env_trains),
			'env_test.csv': str(self.env_tests),
			'time.csv': str(self.times),
			'result.csv': self._params_summary,
			}.items()
		print(self._params_summary)
		if self.log_save:
			makedirs(self.directory, exist_ok=True)
			for file_name, _str in files:
				if not _str: continue
				with open(self.directory+'/'+file_name,'w') as f: f.write(_str)
			save(self.directory+'/best.npy', best_solution)
			save(self.directory+'/train.npy', self.trains.get_array())
			save(self.directory+'/test.npy', self.tests.get_array())
			self.chart_drawer(self.trains.get_array(), self.directory+'/train.png', title="{} {}".format(self._env_name, self._algorithm_name))
			self.chart_drawer(self.tests.get_array(), self.directory+'/test.png', title="{} {}".format(self._env_name, self._algorithm_name))
			self.visualizer(self.directory)
		else:
			print('best_solution',*[i for i in best_solution],sep='\n',end='\n\n')
			for key,val in files:
				if not val: continue
				print(key.split('.')[0],val,sep='\n',end='\n\n')

	@staticmethod
	def get_id(K,D):
		from os import listdir
		from datetime import datetime
		from random import choice as rand_choice
		n=datetime.now(); d,t=str(n).split(' ')
		n=''.join(d.split('-')+t.split('.')[0].split(':'))
		n=''.join([n[6],n[7],n[4],n[5],n[2],n[3],*list(n[8:])])
		#return n
		I=[[str(i)] for i in range(10)]
		for i in range(26): I[(i+K)%10]+=[chr(97+i),chr(65+i)]
		n=''.join([rand_choice(I[int(i)]) for i in n])
		#return n
		if n in listdir(D): print('warning>>DUPLICTE NAME FOUND'); return get_id(K,D)
		return n+str(K);
		d=0; nt=[]
		for i in n:
			if ord(i)>=97: d=97
			elif ord(i)>=65: d=65
			else: nt+=i; continue
			nt+=str((((ord(i)-d)%26)+K)%10)
		return ''.join(nt);
