from param_and_farm_setup import *
from binary_functionals import BinaryRevenueFunctional, BinaryCostFunctional


numpy.set_printoptions(threshold=numpy.nan)

def optimize_continuous_farm(user_params):
	"""Optimize a continuous turbine farm given a mix of default and user-supplied parameters.

	Args:
		user_params: A dictionary of user-supplied and default parameters

	Returns:
		output_dict: A dict with profit, runtime, total friction, and optimal turbine number
		potential_turbine_locs: A list of x,y potential turbine locations, determined by stoachastically placing
								4 * N turbines with probability increasing in optimal turbine density.
								(From the paper, this means algorithm 13 with z = 4)
	"""
	c_farm, c_solver, c_control, c_problem = get_turbines_and_farm(user_params, continuous=True)
	model_turbine = user_params['model_turbine_params']
	cost_per_turbine = model_turbine['cost_coeff'] * (1 / model_turbine['friction'])

	revenue_functional = BinaryRevenueFunctional(c_problem, user_params)
	cost_functional = cost_per_turbine * CostFunctional(c_problem)
	cont_functional = revenue_functional - cost_functional

	c_revenue_params = {
		'max_smeared_friction': model_turbine['max_smeared_friction'],
		'friction_per_turbine': model_turbine['friction'],
		'as': None,
		'save_checkpts': False,
		'load_checkpts': False,
		'cost': cost_per_turbine,
		'iters': user_params['iternum'],
		'tolerance': user_params['tolerance']
	}

	output_dict = optimize_turbine_farm(c_farm, c_solver, c_control, cont_functional,
										c_revenue_params, 'cont')

	x_locs, y_locs = c_farm.placediscrete(4*round(output_dict['cont']['num_turbines']),
										  model_turbine['max_smeared_friction'],
										  model_turbine['min_dist'])
	potential_turbine_locs = zip(x_locs, y_locs)

	return output_dict, potential_turbine_locs


def evaluate_non_continuous_farm(user_params, potential_turbine_locs=None, algo_name=None, optimize_disc=False,
								 optimize_mipdeco=False, full_two_step=False, init_two_step=False):
	"""Evaluate initial profit of a discrete turbine farm with turbine locations chosen from the optimal
	   turbine density field as returned by the continuous algorithm.

	Args:
		user_params: A dictionary containing default and user-specified params for TSTO optimization
		potential_turbine_locs: A list of x,y tuples denoting potential turbine locations. This list
								was constructed using the optimal continuous turbine density field
								and a stochastic algorithm in the ContFarm.place_discrete method
		algo_name: A stirng name determining the key of the output dict
		optimize_disc: A boolean denoting whether or not the intiial two_step turbine locations will
				  be optimized as a second-step discrete optimization problem
		optimize_mipdeco: A boolean denoting whether or not the initial two_step turbine locations
							 will be optimized as a mipdeco problem
		full_two_step: A boolean denoting whether or to the full_two_step algorithm will be run
		init_two_step: A boolean denoting whether or not the init_two_step algorithm will be run

	Returns:
		output_dict: A dict with profit, runtime, total friction, and optimal turbine number for the
			     initial_two_step discrete turbine farm (optimize=False) or the intial and
			     fully optimized two_step dicrete farm (optimize=True)
	"""
	# Turbine, farm, and optimization hyperparameter setup.
	cost_per_turbine = user_params['model_turbine_params']['cost_coeff']
	iternum, tolerance = user_params['iternum'], user_params['tolerance']
	f_farm, f_solver, f_controls, f_problem = get_turbines_and_farm(user_params, locs=potential_turbine_locs)
	output_dict = {}

	# Define the profit functional and reduced functional.
	profit_functional = BinaryRevenueFunctional(f_problem, user_params) - (cost_per_turbine * CostFunctional(f_problem))
	rf_params = ReducedFunctional.default_parameters()
	rf = ReducedFunctional(profit_functional, f_controls, f_solver, rf_params)

	# Display initial params for mipdeco, discrete, two_step, initial_two_step algorithms
	if init_two_step:
		name = 'mipdeco_eval_as_discrete' if algo_name else 'init_two_step'

		two_step_dict = {
			name: {
				'profit': -rf(f_farm.control_array),
				'runtime': 'N/A',
				'total_friction': 0,
				'num_turbines': f_farm.number_of_turbines
			}
		}
		output_dict.update(two_step_dict)

	if optimize_disc:
		disc_functional = BinaryRevenueFunctional(f_problem, user_params)
		output_dict.update(optimize_turbine_farm(
			f_farm, f_solver, f_controls, disc_functional,
			{'cost': cost_per_turbine, 'iters': iternum, 'tolerance': tolerance},
			'discrete'))
	elif full_two_step:
		f_farm, f_solver, f_control, f_problem = get_turbines_and_farm(user_params, locs=potential_turbine_locs)
		disc_functional = BinaryRevenueFunctional(f_problem, user_params)
		output_dict.update(optimize_turbine_farm(
			f_farm, f_solver, f_controls, disc_functional, 
			{'cost': cost_per_turbine, 'iters': iternum, 'tolerance': tolerance}, 
			'full_two_step'))
	elif optimize_mipdeco:
		b_farm, b_solver, b_control, b_problem = get_turbines_and_farm(user_params, locs=potential_turbine_locs,
																	   mipdeco=True)
		revenue_functional = BinaryRevenueFunctional(f_problem, user_params)
		cost_functional = cost_per_turbine * BinaryCostFunctional(f_problem)
		b_functional = revenue_functional - cost_functional
		output_dict.update(optimize_turbine_farm(
			b_farm, b_solver, b_control, b_functional,
			{'cost': cost_per_turbine, 'iters': iternum, 'tolerance': tolerance},
			'mipdeco'))

		# now take the extracted binary variables and examine construct and evaluate a discrete
		# turbine farm with the optimal binaries/locs
		binary_locs = output_dict['mipdeco']['total_friction']
		output_dict.update(evaluate_non_continuous_farm(
			user_params, binary_locs, algo_name='mipdeco_eval_as_discrete', init_two_step=True))
	return output_dict
