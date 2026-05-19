
def main(params):
    from algorithms import algorithms
    from networks import networks
    from envs import envs
    from trainer import Trainer
    from logger import Logger
    logger = Logger(params,params['out_dir'])
    train_env, test_env, sim_env = \
        envs[params['env']['name']](params['env']) \
            ()
    network = \
        networks[params['network']['name']](params['network']) \
            (train_env.obs_shape,train_env.act_shape,params['seed'])
    algorithm = \
        algorithms[params['algorithm']['name']](params['algorithm']) \
            (network.num_params,params['seed'])
    logger(algorithm,network,train_env,test_env,sim_env)
    trainer = Trainer(algorithm,network,train_env,test_env,params['seed'],params['train'],logger)
    solution, score= trainer(params['iterations'])
    logger.end(solution, score)

if __name__=="__main__":
    from defaults import fill
    from yaml import safe_load as load
    from sys import argv
    import traceback
    params= argv[1] if len(argv)>1 else './input.yaml'
    if params.endswith(('.yaml', '.yml')):
        with open(params) as params:
            params= params.read()
    for param in load(params):
        try: main(fill(param))
        except Exception : print(param, traceback.format_exc(), sep='\n')
