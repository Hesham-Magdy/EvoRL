
#from evosax import Strategies
#from evosax.utils.evojax_wrapper import Evosax2JAX_Wrapper

class EvosaxBuilder():

    def __init__(self,params):
        self.name=params['name'].split('-')[-1]
        self.population=self._check_popsize(params['population'],self.name)
        self.config_params, self.algorithm_params ,self.optimizer_params= \
            self._handle_params(params)
        self.config_params['maximize']= True
       
    def __call__(self, num_params, seed):
        from evosax import Strategies
        from evosax.utils.evojax_wrapper import Evosax2JAX_Wrapper
        #print(self.name)
        #print(self.population)
        #print('es_config')
        #print(self.config_params)
        #print('es_params')
        #print(self.algorithm_params)
        #print('opt_params')
        #print(self.optimizer_params)
        return Evosax2JAX_Wrapper(
            evosax_strategy= Strategies[self.name],
            pop_size= self.population,
            param_size= num_params,
            es_config= self.config_params,
            es_params= self.algorithm_params,
            opt_params= self.optimizer_params,
            seed= seed,
        )

    def _check_popsize(_,population,algorithm_name):
        odds= ['ESMC',]
        evens= ['ARS','ASEBO','CR_FM_NES','GuidedES','OpenES','PersistentES','PGPE','NoiseReuseES']
        if (algorithm_name in evens and population%2!=0) \
        or (algorithm_name in odds and population%2==0):
            population+=1
        return population

    def _handle_params(_,params):
        optimizer_params_keys= [
            'opt_name',
            'lrate_init', 'lrate_decay', 'lrate_limit',
        ]
        config_params_keys= [
            'fitness_trafo', 'w_decay',
            'mean_decay',
            'elite_ratio',
            'subspace_dims',
            'memory_size',
            #'temperature',
            #'sigma_init', 'sigma_decay', 'sigma_limit',
            #'sigma_meta',
            'sigma_ratio',
        ]
        algorithm_params_keys= [
            'init_min', 'init_max', 'clip_min', 'clip_max',
            'temperature',
            'sigma_init', 'sigma_decay', 'sigma_limit',
            'sigma_meta',
            #'sigma_ratio',
            'sigma_lrate', 'sigma_max_change',
            'sigma_best_limit',
            'lrate_mean',
            'c_prime', 'use_adasam',
            'lrate_sigma', 'lrate_sigma_init', 'lrate_B', 'rho', #
            'lrate_move_sigma', 'lrate_stag_sigma', 'lrate_conv_sigma', #
            'radius_max', 'radius_min', 'radius_decay',
            'cross_over_rate',
            'diff_w', 'num_diff_vectors', 'mutate_best_vector',
            'grad_decay',
            'alpha', 'beta',
            'T', 'K',
            'noise_scale', 'truncation_selection',
            'inertia_coeff', 'cognitive_coeff', 'social_coeff',
            'range_min', 'range_max',
            'temp_init', 'temp_limit', 'temp_decay', 'boltzmann_const',
            'c_sigma', 'd_sigma', 'c_m', 'c_s', 'q_star', 't_uncorr',
            'mu_eff', 'c_1', 'c1' 'c_mu', 'c_c', 'mu_w', 'chi_n', 'chi_N', 'h_inv', 'c_cov', 'alpha_dist', #
            'eta_avs_inc', 'eta_avs_dec', 'nis_max_gens', 'delta_ams', 'theta_sdr', 'c_mult_init',
            'eta_sigma', 'eta_shift', #
        ]
        config_params, algorithm_params ,optimizer_params= {}, {}, {}
        for param in params:
            if param in ['name', 'population']: continue
            selected_params= \
                config_params if param in config_params_keys else \
                algorithm_params if param in algorithm_params_keys else \
                optimizer_params if param in optimizer_params_keys else \
                None
            if selected_params is None: continue #raise KeyError(param)
            if param in ['clip_min', 'clip_max'] and params[param]==None: continue
            selected_params[param]=params[param]
        return config_params, algorithm_params, optimizer_params

