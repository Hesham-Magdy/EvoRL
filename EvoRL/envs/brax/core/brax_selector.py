
class Selector():
	def __getitem__(self,key):
		match key:
			case 'Ant': return 'ant'
			case 'Humanoid': return 'humanoid'
			case 'HumanoidStandup': return 'humanoidstandup'
			case 'HalfCheetah': return 'halfcheetah'
			case 'InvertedPendulum': return 'inverted_pendulum'
			case 'InvertedDoublePendulum': return 'inverted_double_pendulum'
			case 'Pusher': return 'pusher'
			case 'Reacher': return 'reacher'
			case 'Swimmer': return 'swimmer'
			case 'Walker2D': return 'walker2d'
			case 'Hopper': return 'hopper'
			case 'Fast': return 'fast'
		raise KeyError(key)

brax_selector = Selector()
