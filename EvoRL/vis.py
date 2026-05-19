
from networks import networks
from envs import envs
from logger.Visualizer import Visualizer

def main(params,_dir,steps):
    params['env']['episodic'] = False
    params['env']['auto_reset']= False
    train_env, test_env, sim_env = \
        envs[params['env']['name']](params['env']) \
            ()
    network = \
        networks[params['network']['name']](params['network']) \
            (train_env.obs_shape,train_env.act_shape,params['seed'])
    visualizer= Visualizer(sim_env,network,True,False,False)
    visualizer(_dir, steps)


if __name__=="__main__":
    from defaults import fill
    from yaml import safe_load as load
    from sys import argv
    import traceback
    params= argv[1] if len(argv)>1 else './input.yaml'
    _dir= argv[2] if len(argv)>2 else '.'
    steps= argv[3] if len(argv)>3 else None
    if params.endswith(('.yaml', '.yml')):
        with open(params) as params:
            params= params.read()
    for param in load(params):
        try: main(fill(param),_dir,steps)
        except Exception : print(param, traceback.format_exc(), sep='\n')