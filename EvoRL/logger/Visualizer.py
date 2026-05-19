
class Visualizer():
	def __init__(self,env,policy,visualize_final=True,visualize_train=True,visualize_test=True):
		self.env= env
		self.policy= policy
		self.visualize_final= visualize_final
		self.visualize_train= visualize_train
		self.visualize_test= visualize_test

	def __call__(self,_dir,n_steps=None):
		from os import listdir
		from os.path import isdir, isfile
		from numpy import load as np_load
		for n_dir in (
				([_dir] \
					if self.visualize_final and isdir(_dir) else []) +\
				([_dir+'/data/train/'+i for i in listdir(_dir+'/data/train')] \
					if self.visualize_train and isdir(_dir+'/data/train') else []) +\
				([_dir+'/data/test/'+i for i in listdir(_dir+'/data/test')] \
					if self.visualize_test and isdir(_dir+'/data/test') else []) \
					):
			print('rendering: {}'.format(n_dir))
			rewards,steps=self.visualize_env(
		    	output_file= n_dir+'/episode',
		    	policy_params= np_load(n_dir+'/best.npy') if isfile(n_dir+'/best.npy') else None,
		        norms_params= np_load(n_dir+'/obs_norm.npy') if isfile(n_dir+'/obs_norm.npy') else None,
				env_keys= np_load(n_dir+'/env_init_keys.npy') if isfile(n_dir+'/env_init_keys.npy') else None,
		        steps= n_steps,
		        )
			with open(n_dir+'/episode.txt','w') as f:
				f.write('reward:{}\nsteps:{}'.format(rewards,steps))


	def visualize_env(self,policy_params=None,norms_params=None,env_keys=None,steps=None,output_file='.'):
		if not hasattr(self.env,'run_episode') \
		or not hasattr(self.env,'render_episode') \
		or policy_params is None:
			return
		episode,actions,rewards,steps=self.env.run_episode(self.policy,policy_params,norms_params,env_keys,steps)
		self.env.render_episode(episode,output_file)
		return sum(rewards), steps
