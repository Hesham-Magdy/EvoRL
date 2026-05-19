
from jax.numpy import ndarray as jnp_array
from flax.core import FrozenDict
from flax.struct import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

class MLP():

    class MLPFlaxNetwork(nn.Module):
        hidden_dims: list[int]
        final_dim: int
        hidden_activation: Callable[[jnp_array], jnp_array]|None
        final_activation: Callable[[jnp_array], jnp_array]|None
        #use_bias: bool = True
        #layer_norm: bool = False
        #kernel_init: Callable = jax.nn.initializers.lecun_uniform()

        @nn.compact
        def __call__(self, x:jnp_array) -> jnp_array:
            for hidden_size in self.hidden_dims:
                x= self.hidden_activation(
                    nn.Dense(
                        hidden_size,
                        #use_bias=self.use_bias,
                        #kernel_init=self.kernel_init,
                        #name=f'hidden_{i}',
                    )(x)
                )
                #if self.layer_norm: x=nn.LayerNorm()(x)
            x= nn.Dense(
                self.final_dim,
                #use_bias=self.use_bias,
                #kernel_init=self.kernel_init,
                #name=f'hidden_{i}',
            )(x)
            if self.final_activation!=None:
                x= self.final_activation(x)
                #if self.layer_norm: x=nn.LayerNorm()(x)
            return x

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        hidden_actv: Callable[[jnp_array], jnp_array]|None,
        output_actv: Callable[[jnp_array], jnp_array]|None,
        seed: int = 0,
    ):
        model= self.MLPFlaxNetwork(
            hidden_dims=hidden_dims,
            hidden_activation=hidden_actv,
            final_dim=output_dim,
            final_activation=output_actv,
        )
        self.seed= seed
        dummy_obs= jnp.ones([1, input_dim])
        params= model.init(jax.random.PRNGKey(self.seed), dummy_obs)
        self.num_params, format_params_fn = self._get_params_format_fn(params)
        self._format_params_fn= jax.vmap(format_params_fn)
        self._forward_fn= jax.vmap(model.apply)

    def reset(self, states):
        @dataclass
        class NetworkState:
            keys: jnp_array
        keys = jax.random.split(jax.random.PRNGKey(self.seed), states.obs.shape[0])
        return NetworkState(keys=keys)

    def get_actions(
        self, 
        t_states,
        params,
        p_states,
    ):
        params= self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states

    @staticmethod
    def _get_params_format_fn(init_params: FrozenDict) -> tuple[int, Callable[[jnp_array], FrozenDict]]:
        """Return a function that formats the parameters into a correct format."""
        flat, tree = jax.tree_util.tree_flatten(init_params)
        params_sizes = np.cumsum([np.prod(p.shape) for p in flat])
        def params_format_fn(params:jnp_array) -> FrozenDict:
            params = jax.tree_util.tree_map(
                lambda x, y: x.reshape(y.shape),
                jnp.split(params, params_sizes, axis=-1)[:-1],
                flat)
            return jax.tree_util.tree_unflatten(tree, params)
        return params_sizes[-1], params_format_fn
