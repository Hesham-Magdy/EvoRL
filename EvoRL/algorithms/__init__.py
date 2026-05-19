
class Algorithms():
	def __getitem__(self,key):
		match key:
			case alg if alg in [
				'PGPE',
				'DE',
				'ARS',
				'ASEBO',
				'CMA_ES',
				'CR_FM_NES',
				'DES',
				'ESMC',
				'Full_iAMaLGaM',
				'GESMR_GA',
				'GLD',
				'GuidedES',
				'Indep_iAMaLGaM',
				'LM_MA_ES',
				'MA_ES',
				'MR15_GA',
				'OpenES',
				'PBT',
				'PersistentES',
				'PSO',
				'RandomSearch',
				'RmES',
				'SAMR_GA',
				'Sep_CMA_ES',
				'SimAnneal',
				'SimpleES',
				'SimpleGA',
				'SNES',
				'xNES',
				'LGA',
				'LES',
				'NoiseReuseES',
				'IPOP_CMA_ES',
				'BIPOP_CMA_ES',
			]: from .evosax import EvosaxBuilder as algorithm
		if 'algorithm' not in locals(): raise KeyError(key)
		return algorithm

algorithms = Algorithms()
