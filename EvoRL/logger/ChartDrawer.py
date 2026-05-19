
#from matplotlib.pyplot import subplots, close

class ChartDrawer():
	def __init__(self):
		self.fig, self.plt = None, None

	def __call__(self, arr, file_name, title=None):
		from matplotlib.pyplot import subplots, close
		self.fig, self.plt = subplots()
		n_iters, max_arr, mean_arr, min_arr, std_arr= [arr[:,i] for i in range(5)]
		self.plt.set_title(title or 'Scores'); self.plt.set_xlabel('iterations'); self.plt.set_ylabel('reward')
		self.plt.plot(n_iters, max_arr, color='b', label='max')
		self.plt.plot(n_iters, min_arr, color='r', label='min')
		self.plt.plot(n_iters, mean_arr, color='g', label='mean')
		self.plt.errorbar(n_iters, mean_arr, yerr=std_arr)
		self.plt.legend()
		self.fig.savefig(file_name); close(self.fig)
