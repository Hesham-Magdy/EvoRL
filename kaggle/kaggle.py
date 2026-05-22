input_str_list= \
[
"""\
- iterations: 1000
  algorithm:
    name: PGPE
    population: 1024
    elite_ratio: 1.0
    sigma_lrate: 0.07
    sigma_max_change: 0.2
    sigma_init: 0.04
    sigma_decay: 0.999
    sigma_limit: 0.01
    opt_name: adam
    lrate_init: 0.01
    lrate_decay: 0.999
    lrate_limit: 0.001
    mean_decay: 0.0
    w_decay: 0.0
    fitness_trafo: centered_rank
    init_min: 0.0
    init_max: 0.0
    clip_min: null
    clip_max: null
  env:
    name: Humanoid
    physics_backend: spring
    episode_length: 1000
    action_repeat: 1
    episodic: true
    auto_reset: true
  network:
    name: MLP
    layers: 64~64~64~64
    activation_fn: tanh
  train:
    test_interval: 25
    n_tests: 128
    normalization: true
  log:
    log_interval: 1
    log_save: true
    save_test: true
    save_train: false
    save_train_interval: 50
    save_population: false
    save_rewards: false
    save_env: false
    visualize_final_env: true
    visualize_test_envs: false
    visualize_train_envs: false
  seed: 12

""",
]

dataset= \
    '-EvoRL-'

path= \
    '/kaggle/input/'+ dataset

main_commands= [
   ['python3', path+'/main.py', input_str, ] for input_str in input_str_list
   #['perun', 'monitor', path+'/main.py', input_str, ] for input_str in input_str_list
]

init_commands= [
    #['python3', '-m', 'pip', 'install', '--upgrade', 'pip', ],
    #['python3', '-m', 'pip', 'install', '--upgrade', 'perun[nvidia]', ],
    #['python3', '-m', 'pip', 'install', '--upgrade', '-r', path+'/.init/requirements.txt', ],
    #['python3', path+'/.init/init.py', path+'/.init/requirements.txt', ],
    #['python3', '-m', 'pip', 'install', '--upgrade', 'perun[nvidia, rocm]', ],
    #['python3', '-m', 'pip', 'install', '--upgrade', 'perun', ],
    ['python3', '-m', 'pip', 'install', '--upgrade', 'numpy==1.25.2' ],
    ['python3', '-m', 'pip', 'install', '--upgrade', 'cma==4.0.0' ],
    ['python3', '-m', 'pip', 'install', '--upgrade', 'flask-cors==5.0.1' ],
    ['python3', '-m', 'pip', 'install', '--upgrade', 'glfw==2.9.0' ],
    ['python3', '-m', 'pip', 'install', '--upgrade', 'mujoco==3.2.2' ],
    ['python3', '-m', 'pip', 'install', '--upgrade', 'mujoco-mjx==3.2.2' ],
    ['python3', '-m', 'pip', 'install', '--upgrade', 'PyOpenGL==3.1.9' ],
    ['python3', '-m', 'pip', 'install', '--upgrade', 'trimesh==4.6.8' ],
    ['python3', '-m', 'pip', 'install', '--upgrade', 'brax==0.10.4' ],
    ['python3', '-m', 'pip', 'install', '--upgrade', 'evojax==0.2.17' ],
    ['python3', '-m', 'pip', 'install', '--upgrade', 'evosax==0.1.6' ],
]

main_out, main_err = 'out.txt', 'stderr'
init_out, init_err = 'init.txt', 'init.txt'

TASKS= {
    (init_out, init_err): [
        *init_commands,
    ],
    ('./requirements.txt', './requirements.txt'): [
        ['echo', '-n', '#Date: ', ], ['date', '+%d/%m/%Y', ],
        ['echo', '-n', '#OS: ', ], ['lsb_release', '-d', '-s', ],
        ['echo', '-n', '#', ], ['python3', '--version', ],
        ['python3', '-m', 'pip', 'list', '--format', 'freeze', ],        
    ],
    (main_out, main_err): [
        *main_commands,
    ],
}

#Kaggle API 1.6.17
if __file__!="/kaggle/src/script.py": raise SystemExit(1)
if __name__=="__main__":
    from subprocess import run, DEVNULL
    from contextlib import nullcontext
    from os import listdir
    from os.path import isdir
    from shutil import make_archive, rmtree
    for command, out, err in [
        (command, *files) \
            for files, commands in TASKS.items() \
                for command in commands
    ]:
        out, err=[
            {
                'stdout': nullcontext(None),
                'stderr': nullcontext(None),
                None: nullcontext(DEVNULL),
            }.get(i) or open(i, 'a') \
                for i in (out, err)
        ]
        with out as out, err as err:
            run(command, stdout=out, stderr=err)
        for dir_name in listdir('.'):
            if not isdir(dir_name): continue
            archive_name, count = dir_name, 1
            while archive_name+'.zip' in listdir('.'):
                archive_name=dir_name+'_'+str(count); count+=1
            make_archive(archive_name, 'zip', dir_name)
            rmtree(dir_name)
