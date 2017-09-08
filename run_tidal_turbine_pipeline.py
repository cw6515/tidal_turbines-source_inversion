import click
from opentidalfarm_extensions.helper_functions.main_optimization_functions import *


def print_optimization_info(output_dict):
	"""Print relevant optimization results given a dictionary of output"""
	print output_dict
	for algo_name, value_dict in output_dict.items():
		print 'Total profit from the {} algorithm is {} GBP'.format(algo_name, value_dict['profit'])
		print 'The runtime for the {} algorithm was {} seconds'.format(algo_name, value_dict['runtime'])

		if algo_name == 'cont':
			fric, turbines = value_dict['total_friction'], value_dict['num_turbines']
			print 'The cont algo has {} total friction and n={} optimal turbines'.format(fric, turbines)
		else:
			print 'This optimization routine was undertaken with {} turbines'.format(value_dict['num_turbines'])


@click.command()
@click.option('--turbine_num', default=30, help='The number of turbines to optimize in a discrete farm')
@click.option('--mesh_name', default='small_mesh', help='Choose one of four predefined meshes in {large_mesh, '
														'alt_large_mesh, middle_mesh, small_mesh}')
@click.option('--discrete/--no-discrete', default=False, help='Runs the discrete algorithm')
@click.option('--init_two_step/--no-init_two_step', default=False, help='Get the profit from thje init_two steoalgorithm')
@click.option('--full_two_step/--no-full_two_step', default=False, help='Runs the two-step algorithm')
@click.option('--mipdeco/--no-mipdeco', default=False, help='Runs the mipdeco algorithm')
@click.option('--turbine_friction', default=.927, help='User-specified friction of individual turbines')
@click.option('--minimum_distance', default=40, help='User-specified minimum distance between turbines')
@click.option('--efficiency', default=.5, help='Efficiency of turbine energy extraction')
@click.option('--lcoe', default=107.89, help='Levelized cost of Energy')
@click.option('--discount_rate', default=.1, help='Efficiency of turbine energy extraction')
@click.option('--income_per_unit', default=330.51, help='Income per MWh in 2016 GBP')
@click.option('--timesteps', default=5, help='Number of years over which the farm is optimized')
@click.option('--velocity', default=2.0, help='Number of years over which the farm is optimized')
@click.option('--blade_radius', default=8.35, help='Number of years over which the farm is optimized')
def run_tidal_turbine_pipeline(turbine_num, mesh_name, discrete, init_two_step, full_two_step, mipdeco,
							   turbine_friction, minimum_distance, efficiency, lcoe, discount_rate,
							   income_per_unit, timesteps, velocity, blade_radius):
	"""Run the tidal turbine pipeline with a choice of algorithms and default and/or user-specified parameters
		Args:
			turbine_num: number of discrete turbines to optimize for
			mesh_name: Name of the mesh folder which contains the mesh.xml you want to use
			discrete: A boolean determining whether the discrete algorithm will be run
			init_two_step: A boolean determining whether the initial two-step profit will be calculated
			full_two_step: A boolean determining whether the two-step algorithm will be run
			mipdeco: A boolean determining whether the mipdeco algorithm will be run
			turbine_friction: The friction per discrete turbine. Defaults to ModelBinaryTurbine value
			minimum_distance: The minimum distance between two turbines. Default value as above.
			efficiency: A value between 0 and 1 representing efficiency of power extraction
			lcoe: The levelized cost of energy extraction. See paper for default justification
			discount_rate: The time-discounting rate of money. See paper as above
			income_per_unit: Income per unit of MWh of energy. See paper as above
			timesteps: Number of time periods (years) for the optimization algorithms to optimize over
			velocity: Constant velocity of water entering the western boundary of the farm
			blade_radius: Radius of the turbine blade
	"""
	model_turbine = BinaryModelTurbine(blade_radius, efficiency, velocity, discount_rate, timesteps, lcoe,
									   minimum_distance)
	param_dict = get_algorithm_params(mesh_name, model_turbine)

	user_params = {
				   'turbine_friction': turbine_friction,
				   'diameter': 2 * blade_radius,
				   'minimum_distance': minimum_distance,
				   'turbine_num': turbine_num,
				   'efficiency': efficiency,
				   'LCOE': lcoe,
				   'discount_rate': discount_rate,
				   'I': income_per_unit,
				   'timesteps': timesteps
				   }

	# update user_params to include all user-supplied and default parameters
	update_user_params(user_params, param_dict)

	# the continuous algorithm is run by default
	output_dict = {}
	new_dict, discrete_locs = optimize_continuous_farm(user_params)
	output_dict.update(new_dict)

	# run the discrete algorithm
	if discrete:
		discrete_dict = evaluate_non_continuous_farm(user_params, potential_turbine_locs=None, algo_name='discrete',
													 optimize_disc=True)
		output_dict.update(discrete_dict)

	# evaluate the continuous algorithm after converting a continuous density field to discrete turbines
	if init_two_step:
		init_two_step_dict = evaluate_non_continuous_farm(user_params, discrete_locs, init_two_step=True)
		output_dict.update(init_two_step_dict)

	# optimize the initial placement of the turbines placed in `init_two_step`
	if full_two_step:
		full_two_step_dict = evaluate_non_continuous_farm(user_params, discrete_locs, algo_name='full_two_step',
														  full_two_step=True)
		output_dict.update(full_two_step_dict)

	# run the mipdeco algorithm with the continuous optimal density field as input
	if mipdeco:
		mipdeco_dict = evaluate_non_continuous_farm(user_params, discrete_locs, algo_name='mipdeco',
													optimize_mipdeco=True)
		output_dict.update(mipdeco_dict)	

	# display the results for all chosen optimization algorithms
	print_optimization_info(output_dict)


# runs the code with the specified --args
if __name__ == '__main__':
	run_tidal_turbine_pipeline()
