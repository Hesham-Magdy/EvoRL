
#https://flax.readthedocs.io/en/latest/api_reference/flax.linen/activation_functions.html

from jax.numpy import ndarray as jnp_array
from typing import Callable

from flax.linen import activation as actv

class ActivationFunctions:
	def __getitem__(self, key:str) -> Callable[[jnp_array], jnp_array]|None:
		if key is None: return None
		if key==\
			'PReLU': return actv.PReLU() #negative_slope_init=0.01
				#Parametric Rectified Linear Unit (PReLU) activation function
				#PReLU is a Flax layer and not a simple activation function, so it needs to be initialized before being called

		return {
			'tanh': actv.tanh,
				#Compute hyperbolic tangent element-wise
			'hard_tanh': actv.hard_tanh,
				#Hard tanh activation function

			'sigmoid': actv.sigmoid,
				#Sigmoid activation function
			'swish': actv.swish,
			'silu': actv.silu,
				#SiLU (aka swish) activation function
				#swish and silu are both aliases for the same function
			'log_sigmoid': actv.log_sigmoid,
				#Log-sigmoid activation function

			'relu': actv.relu,
				#Rectified linear unit activation function
				#https://openreview.net/forum?id=urrcVI-_jRm
			'leaky_relu': actv.leaky_relu, #negative_slope=0.01
				#Leaky rectified linear unit activation function

			'relu6': actv.relu6,
				#Rectified Linear Unit 6 activation function
			'hard_sigmoid': actv.hard_sigmoid,
				#Hard Sigmoid activation function
			'hard_silu': actv.hard_silu,
			'hard_swish': actv.hard_swish,
				#Hard SiLU (swish) activation function
				#Both hard_silu and hard_swish are aliases for the same function

			'elu': actv.elu, #alpha=1.0
				#Exponential linear unit activation function
			'celu': actv.celu, #alpha=1.0
				#Continuously-differentiable exponential linear unit activation
				#https://arxiv.org/abs/1704.07483
			'selu': actv.selu,
				#Scaled exponential linear unit activation
				#https://arxiv.org/abs/1706.02515

			'gelu': actv.gelu, #approximate=True
				#Gaussian error linear unit activation function
				#https://arxiv.org/abs/1606.08415
			'glu': actv.glu, #axis=-1
				#Gated linear unit activation function

			'standardize': actv.standardize, #axis=-1, epsilon=1e-05
				#Normalizes an array by subtracting mean and dividing by sqrt(variance)
			'softmax': actv.softmax, #axis=-1
				#Softmax function, rescales elements to the range [0,1] such that the elements along axis sum to 1.
			'log_softmax': actv.log_softmax, #axis=-1
				#Log-Softmax function, rescales elements to the range [-inf, 0]
			'softplus': actv.softplus,
				#Softplus activation function
			'soft_sign': actv.soft_sign,
				#Soft-sign activation function

			'logsumexp': actv.logsumexp, #axis=None, b=None
				#Log-sum-exp reduction, JAX implementation of scipy.special.logsumexp
			#'one_hot': actv.one_hot, #num_classes
			#	#One-hot encodes the given indices
			#	#Each index in the input is encoded as a vector of zeros of length num_classes with the element at index set to one, Indices outside the range [0, num_classes] will be encoded as zeros
		}[key]
