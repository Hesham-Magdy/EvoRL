
#from evojax.sim_mgr import SimManager
#from evojax.obs_norm import ObsNormalizer
#from tqdm import trange
from time import perf_counter as get_time

class Trainer(object):

    def __init__(self, algorithm, network, env_train, env_test, seed, params, logger):
        from evojax.sim_mgr import SimManager
        from evojax.obs_norm import ObsNormalizer
        #from time import perf_counter as get_time
        self.test_interval = params['test_interval']
        n_tests = params['n_tests']
        normalization = params['normalization']
        self.algorithm = algorithm
        self.env = env_train
        self.evaluator = SimManager(
            pop_size = algorithm.pop_size,
            policy_net = network,
            train_vec_task = env_train,
            valid_vec_task = env_test,
            n_evaluations = n_tests,
            obs_normalizer=ObsNormalizer(
                obs_shape=env_train.obs_shape,
                dummy=not normalization,
            ),
            seed=seed,
            n_repeats = 1,
            test_n_repeats = 1,
            use_for_loop = False,
        )
        self.logger = logger
        self.timer = get_time()

    def __call__(self,iters):
        from tqdm import trange
        best, max_score= None, -float('inf')
        for i in trange(iters, ascii=True, postfix='\n'):
            params, score= self._train(i)
            if score >= max_score: best, max_score= params, score
            if (i>0 and i%self.test_interval==0) or i==iters-1:
                score= self._test(i,best)
                if score >= max_score: max_score= score
        return best, max_score

    def _train(self,i):
        self.tick()
        population = self.algorithm.ask(); \
                    time_ask=self.tick()
        rewards, env_states = \
            self.evaluator.eval_params(params=population, test=False); \
                    time_eval=self.tick()
        self.algorithm.tell(rewards); \
                    time_tell=self.tick()
        self.logger.log(
            iter_n= i,
            population= population,
            rewards= rewards,
            env_states= env_states,
            test= False,
        )
        self.logger.log_time(
            iter_n= i,
            _eval= time_eval,
            ask= time_ask,
            tell= time_tell,
            test= False,
        )
        return population[rewards.argmax()], rewards.max()

    def _test(self,i,test_params=None):
        self.tick()
        test_params= test_params if test_params is not None else self.algorithm.best_params
        rewards, env_states = self.evaluator.eval_params(params=test_params, test=True)
        self.logger.log(
            iter_n= i,
            population= test_params,
            rewards= rewards,
            env_states= env_states,
            test= True,
        )
        self.logger.log_time(
            iter_n= i,
            _eval= self.tick(),
            test= True,
            )
        return rewards.max()

    def tick(self):
        #from time import perf_counter as get_time
        now = get_time()
        time_consumed = now - self.timer
        self.timer = now
        return time_consumed
