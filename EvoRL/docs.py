
def _generate(
	algorithm_name, 
	env_name,
	network_name = "MLP",
):
	from yaml import safe_load as load
	from yaml import dump
	from os.path import split
	path, _ = split(__file__)
	files = [
		path+ '/defaults/default_simple.yaml',
		path+ '/defaults/algorithms/'+algorithm_name+'.yaml',
		path+ '/defaults/envs/'+env_name+'.yaml',
		path+ '/defaults/networks/'+network_name+'.yaml',
	]
	ret= {}
	for file in files:
		with open(file) as file:
			ret.update(load(file)[0])
	print(dump(
			[ret],
			indent=2,
			sort_keys=False,
	))

def _list(key):
	from os import listdir
	from os.path import split, isfile
	key = {
		'algorithms': 'algorithms',
		'networks': 'networks',
		'envs': 'envs',
		'activation_fn': 'activation_fn',
	}[key]
	for i in [
		i[:-5] for i in listdir(
			"{}/defaults/{}".format(split(__file__)[0], key)
		) if i[-5:]=='.yaml'
	]: print(i)

if __name__=="__main__":
	from sys import argv
	if argv[1]=='generate':
		_generate(
			algorithm_name= argv[2], 
			env_name= argv[3],
		)
	if argv[1]=='list':
		_list(argv[2])
