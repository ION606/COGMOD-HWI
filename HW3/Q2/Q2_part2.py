import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial


def sim_ddm(v=1.0, a=1.0, beta=0.5, tau=0.3, sigma=1.0, dt=0.001, max_steps=3000):
	X = beta * a  # start
	t = 0.0
	for _ in range(max_steps):
		dW = np.random.normal(0, np.sqrt(dt))
		dX = v * dt + sigma * dW
		X += dX
		t += dt
		if X >= a:
			return t + tau, 1  # upper bound hit
		elif X <= 0:
			return t + tau, 0  # lower bound hit
	return max_steps * dt + tau, None  # timeout (which I ignored)


def sim_param(param_name, param_value, n_trials=200000):
	default_params = {'v': 1.0, 'a': 1.0,
                   'beta': 0.5, 'tau': 0.3, 'sigma': 1.0}
	params = default_params.copy()
	params[param_name] = param_value
	upper_rts, lower_rts = [], []
	for _ in range(n_trials):
		rt, choice = sim_ddm(**params)
		if choice == 1:
			upper_rts.append(rt)
		elif choice == 0:
			lower_rts.append(rt)
	return (upper_rts, lower_rts)  # Return all RTs


# deepseek-r1 wrote this to help parallelize my code (because for loops aren't cool when they're frying my laptop)
def parallel_sim_param(param_name, param_values, n_trials):
	worker = partial(sim_param,
                  param_name, n_trials=n_trials)
	with Pool(processes=cpu_count()) as pool:
		results = pool.map(worker, param_values)
	return results


parameters = {
	'v': np.linspace(0.5, 1.5, 25),
	'a': np.linspace(0.5, 2.0, 25),
	'beta': np.linspace(0.3, 0.7, 25),
	'tau': np.linspace(0.1, 0.5, 25),
}

fig, axes = plt.subplots(4, 2, figsize=(15, 20))  # should this be (15, 15)?
axes = axes.flatten()

for i, (param, values) in enumerate(parameters.items()):
	results = parallel_sim_param(param, values, n_trials=200000)

	# no bootstrapping
	means_upper, means_lower = [], []
	stdev_upper, stdev_lower = [], []

	for upper_rts, lower_rts in results:
		mu_upper = np.mean(upper_rts) if upper_rts else np.nan
		mu_lower = np.mean(lower_rts) if lower_rts else np.nan
		std_upper = np.std(upper_rts) if upper_rts else np.nan
		std_lower = np.std(lower_rts) if lower_rts else np.nan

		means_upper.append(mu_upper)
		means_lower.append(mu_lower)
		stdev_upper.append(std_upper)
		stdev_lower.append(std_lower)

	# means
	ax_mean = axes[2 * i]
	ax_mean.plot(values, means_upper, 'o-', label='Upper Boundary Mean RT')
	ax_mean.plot(values, means_lower, 's-', label='Lower Boundary Mean RT')
	ax_mean.plot(values, np.subtract(means_upper, means_lower),
	             'd-', label='Difference', color='red')
	ax_mean.set_xlabel(param)
	ax_mean.set_ylabel('Response Time (s)')
	ax_mean.set_title(f'Effect of {param} on RT Means')
	ax_mean.legend()
	ax_mean.grid(True)

	# STDDEV
	ax_std = axes[2 * i + 1]
	ax_std.plot(values, stdev_upper, 'o-', label='Upper Boundary Std RT')
	ax_std.plot(values, stdev_lower, 's-', label='Lower Boundary Std RT')
	ax_std.set_xlabel(param)
	ax_std.set_ylabel('Standard Deviation (s)')
	ax_std.set_title(f'Effect of {param} on RT Std Devs')
	ax_std.legend()
	ax_std.grid(True)

	plt.tight_layout()
	plt.savefig('part2.png')

	# DEBUGGING
	print(f"\nVARYING {param.upper()}:\n")
	print(f"Means (Upper): {np.round(means_upper, 5)}")
	print(f"Means (Lower): {np.round(means_lower, 5)}")
	print(f"Std (Upper): {np.round(stdev_upper, 5)}")
	print(f"Std (Lower): {np.round(stdev_lower, 5)}")
