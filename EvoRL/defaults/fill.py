
def fill(params):
    from yaml import safe_load as load
    from os.path import split
    #path= '/'.join(__file__.split('/')[:-1])
    path, _ = split(__file__)
    files = [
        path+ '/default.yaml',
        path+ '/algorithms/'+params['algorithm']['name']+'.yaml',
        path+ '/envs/'+params['env']['name']+'.yaml',
        path+ '/networks/'+params['network']['name']+'.yaml',
    ]
    n_params= {}
    for file in files:
        with open(file) as file:
            n_params.update(load(file)[0])
    for i in [
        'env',
        'algorithm',
        'network',
        'train',
        'log',
        'iterations',
        'seed',
        'out_dir',
    ]:
        if i in params: 
            if isinstance(n_params[i], dict):
                n_params[i].update(params[i])
            else: 
                n_params[i]=params[i]            
    return n_params
