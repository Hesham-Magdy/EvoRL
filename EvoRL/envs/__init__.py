
class Envs():
	def __getitem__(self,key):
		match key:
			case 'CartPole': from .evojax import CartPole as env
			case 'Ant': from .brax import BraxBuilder as env
			case 'Humanoid': from .brax import BraxBuilder as env
			case 'HumanoidStandup': from .brax import BraxBuilder as env
			case 'HalfCheetah': from .brax import BraxBuilder as env
			case 'InvertedPendulum': from .brax import BraxBuilder as env
			case 'InvertedDoublePendulum': from .brax import BraxBuilder as env
			case 'Pusher': from .brax import BraxBuilder as env
			case 'Reacher': from .brax import BraxBuilder as env
			case 'Hopper': from .brax import BraxBuilder as env
			case 'Walker2D': from .brax import BraxBuilder as env
			case 'Swimmer': from .brax import BraxBuilder as env
		if 'env' not in locals(): raise KeyError(key)
		return env

envs = Envs()
