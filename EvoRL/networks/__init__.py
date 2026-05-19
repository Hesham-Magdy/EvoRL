
class Networks():
	def __getitem__(self,key):
		match key:
			case 'MLP': from .MLPBuilder import MLPBuilder as network
		if 'network' not in locals(): raise KeyError(key)
		return network

networks = Networks()
