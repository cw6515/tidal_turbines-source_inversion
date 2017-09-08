import time as timer
import os
from farm_and_turbine_infra_helpers import *
from BinaryReducedFunctional import BinaryReducedFunctional
numpy.set_printoptions(threshold=numpy.nan)


CONSTANTS = {'viscosity': Constant(2.0), 'depth': Constant(50), 'friction': Constant(0.0025), 'efficiency': .5,
			'LCOE': 330.51, 'timesteps': 5, 'discount_rate': .1, 'cost_coeff': 2971450.19256, 'iternum': 25,
			'tolerance': 2.5e-6}

# adjust the farm area if so desired -- [x_start, x_end, y_start, y_end]
MESH_DICT = {'small_mesh': [200, 600, 200, 400],
			 'middle_mesh': [800, 1600, 800, 1600],
			 'large_mesh': [1100, 1900, 1200, 1800],
			 'alt_large_mesh': [1100, 1900, 1200, 1800]}

def update_user_params(user_params, param_dict, constants=CONSTANTS):
	"""Combine user-supplied params and default params into a single dict"""
	for key, value in constants.items():
		if not user_params.get(key):
			user_params[key] = value
	for key, value in param_dict.items():
		if not user_params.get(key):
			user_params[key] = value
	return user_params


def get_farm_area(mesh_name, mesh_dict=MESH_DICT):
	"""Get the length of the farm domain (in m) along the x and y dimensions"""
	return mesh_dict[mesh_name]


def get_domain(mesh_name):
	"""Gets the domain of a mesh xml file created with GMSH

	Args:
		mesh_name: the file name of the imported mesh object -- inspect will look up its path
				   and load the object accordingly
		mesh_dict: A dictionary with `mesh_name` keys and [x_start, x_end, y_start, y_end] list values which delineate
				   the farm subdomain

	Returns:
		domain: A dolfin mesh object
		farm_area: A list delineating the farm subdomain of the format [x_start, x_end, y_start, y_end]
	"""
	if mesh_name not in ['alt_large_mesh', 'large_mesh', 'middle_mesh', 'small_mesh']:
		raise NotImplementedError('There is currently no support for user-defined meshes. If desired, use and define a '
								  'gmsh mesh object and convert it to xml format by running the code in any of the '
								  'mesh folders Makefile. Currently supported meshnames are given in the click.help '
								  'module for the mesh parameter')
	farm_area = get_farm_area(mesh_name)
	mesh_path = '{}/mesh.xml'.format(os.path.abspath(mesh_name))
	domain = FileDomain(mesh_path)

	return domain, farm_area


def get_boundary_conditions(velocity=2, sinusoidal=False):
	"""Get the boundary conditions based on algorithm and desired flow of velocity

	Args:
		velocity: Desired constant velocity of flow into the western side ofn the box domain
		sinusoidal: Whether the velocity of flow is sinusoidal or not

	Returns:
		bcs: Boundary conditions for the specified algorithm
	"""
	if sinusoidal:
		raise NotImplementedError('Sinusoidal boundary conditions are not currently supported')
		# TODO(christian): implement sinusoidal support here

	bcs = BoundaryConditionSet()
	bcs.add_bc("u", Constant((velocity, 0)), facet_id=1)
	bcs.add_bc("eta", Constant(0), facet_id=2)
	bcs.add_bc("u", facet_id=3, bctype="free_slip")

	return bcs


def get_model_turbine_params(model_turbine):
	"""Get turbine parameters from an idealized model turbine. See BinaryModelTurbine for all params"""
	params = {'cost_coeff': model_turbine.discounted_cost_coefficient,
			  'bump_height': model_turbine.bump_height,
			  'diameter': model_turbine.blade_diameter,
			  'min_dist': model_turbine.minimum_distance,
			  'max_smeared_friction': model_turbine.maximum_smeared_friction,
			  'friction': model_turbine.friction
			  }

	return params


def get_sw_problem_params(boundary_conditions, domain, viscosity=Constant(2.0), depth=Constant(50),
						  friction=Constant(0.0025)):
	"""Get basic params for a tidal turbine optimization problem

	Args:
		boundary_conditions: A set of boundary conditions defined by get_boundary_conditions()
		domain: A FileDomain object with a path to the GMSH mesh which has been converted to dolfin
		viscosity: kinematic viscosity coefficient used in the shallow water equations
		depth: Resting water depth as used in the shallow water equations
		friction: Constant background friction

	Returns:
		prob_params: A class with appropriate problem parameters for solving a tidal stream turbine optimization problem
	"""
	prob_params = SteadySWProblem.default_parameters()
	prob_params.domain = domain
	prob_params.bcs = boundary_conditions
	prob_params.viscosity = viscosity
	prob_params.depth = depth
	prob_params.friction = friction

	return prob_params


def get_algorithm_params(mesh_name, model_turbine, constants_dict=CONSTANTS):
	"""Get all default params and a domain for a given mesh name

	Args:
		mesh_name: file name (not path) of the imported mesh object
		model_turbine: An object of the ModelTurbine or BinaryModelTurbine class, instantiated with chosen
					   parameters for turbine blade radius, minimum turbine distance, efficiency of energy
					   extraction, etc...
		constants_dict: A dict of default constants including viscosity, resting depth, background friction

	Returns:
		param_dict: A dictionary of problem parameters
	"""
	domain, farm_area = get_domain(mesh_name)
	bcs = get_boundary_conditions()
	problem_params = get_sw_problem_params(bcs, domain, constants_dict['viscosity'],
										 constants_dict['depth'], constants_dict['friction'])
	model_turbine_params = get_model_turbine_params(model_turbine)
	param_dict = {'model_turbine_params': model_turbine_params,
				  'domain': domain,
				  'bcs': bcs,
				  'prob_params': problem_params,
				  'farm_area': farm_area
	}
	return param_dict


def mark_farm_subdomain(farm_area, domain, farm):
	"""Mark the farm subdomain region to distinguish it from non-farm outer domain"""
	farm_domain = FarmDomain(farm_area[0], farm_area[1], farm_area[2], farm_area[3])
	domains = MeshFunction("size_t", domain.mesh, domain.mesh.topology().dim())
	domains.set_all(0)
	farm_domain.mark(domains, 1)
	site_dx = Measure("dx")(subdomain_data=domains)
	farm.site_dx = site_dx(1)
	plot(domains, interactive=True)

	# TODO(christian): combine subdomains so algorithm 13 can check based on farm.site_dx
	if isinstance(farm, ContFarm) or isinstance(farm, BinaryFarm):
		farm.subDomain = farm_area

	return


def get_basic_solver(prob_params, continuous):
	"""Get a basic coupled solver from fully defined problem params"""
	sol_params = CoupledSWSolver.default_parameters()
	sol_params.dump_period = 1

	if continuous:
		sol_params.cache_forward_state = False

	coupled_solver = CoupledSWSolver(prob_params, sol_params)

	return coupled_solver


def get_reduced_functional(raw_functional, prob_control, prob_solver, rf_params, algo_name):
	"""Get a ReducedFunctional or BinaryReducedFunctional depending on algorithm"""
	if algo_name == 'mipdeco':
		return BinaryReducedFunctional(raw_functional, prob_control, prob_solver, rf_params)

	else:
		return ReducedFunctional(raw_functional, prob_control, prob_solver, rf_params)


def get_turbine_layout(num_turbines):
	"""Get a [n, m] list representing the n x m regular or stagged turbine layout"""
	factor_list = []
	num = 2.0
	while num <= sqrt(num_turbines):
		factor = num_turbines / num
		if round(factor, 5) == factor:
			factor_list.append([factor, num])
		num += 1

	return factor_list[-1]


def place_turbines_in_farm(regular_layout, n_turbines, farm, locs):
	"""Place n turbines in a discrete farm using a regular, staggered, or user-determined layout

	Args:
		regular_layout: A boolean denoting if a regular layout is to be usaed
		n_turbines: The number of turbines to be placed
		farm: A discrete farm
		locs: A list of x,y tuple locations for binary turbines to be placed upon
	"""
	if locs:
		count = 0
		for loc in locs:
			if count < n_turbines:
				farm.add_turbine(loc)
			count += 1
		return

	if regular_layout:
		numx, numy = get_turbine_layout(n_turbines)
		farm.add_regular_turbine_layout(num_x=numx, num_y=numy)

	else:
		raise NotImplementedError('A staggered layout is not implemented at this time')

	return


def place_binary_turbines(binary_farm, locations):
	"""Place turbines with randomly initialized binary coefficients in a binary farm

	Args:
		binary_farm: An instance of the BinaryFarm class
		locations: A list of x,y tuples represening optimal discrete turbine locations as determined
				   by the conversion of the optimal continuous turbine density field to discrete
				   turbines

	Returns:
		Nothing, but the binary_farm is mutated such that a number of turbines and binary variables
		have been placed
	"""
	for loc in locations:
		binary_farm.addBinaryTurbine(loc, numpy.random.uniform(0, .2))

	return


def get_turbines_and_farm(user_params, regular_layout=True, locs=None, continuous=False, mipdeco=False):
	"""Get a discrete turbine farm and its associated turbines using default (and optional user-supplied) params

	Args:
		user_params: A dictionary containing all default and user-specified params for TSTO optimization
		regular_layout: A boolean determining if the default turbine layout is regular or staggered
		locs: An optional list of x,y tuples denoting thelocation for individual turbines to be placed
		continuous: A boolean denoting whether the farm is continuous or not
		mipdeco: A boolean denoting whether the farm will be optimized via MIPDECO methods or not

	Returns:
		t_farm: A turbine farm inside the specified domain with n_turbines placed
		farm_solver: A coupled SW solver created with a blend of default and user0-defined params
		farm_control: A control object for the farm
		farm_problem: problem parameters for the farm
	"""
	friction = user_params['turbine_friction']
	diameter = user_params['diameter']
	minimum_distance = user_params['minimum_distance']
	num_turbines = user_params['turbine_num']
	domain = user_params['domain']
	farm_area = user_params['farm_area']

	if continuous:
		turbine = SmearedTurbine()
		function_space = FunctionSpace(domain.mesh, "DG", 0)
		t_farm = ContFarm(domain, turbine, function_space=function_space)

	elif mipdeco:
		turbine = BinaryTurbine(friction=friction, diameter=diameter, minimum_distance=minimum_distance)
		t_farm = BinaryFarm(domain, turbine)
		place_binary_turbines(t_farm, locs)

	else:
		turbine = BumpTurbine(friction=friction, diameter=diameter, minimum_distance=minimum_distance)
		t_farm = RectangularFarm(domain, site_x_start=farm_area[0], site_x_end=farm_area[1], site_y_start=farm_area[2],
								 site_y_end=farm_area[3], turbine=turbine)
		place_turbines_in_farm(regular_layout, num_turbines, t_farm, locs)

	# mark subdomain to allow the farm to discriminate between farm/non-farm domain regions
	mark_farm_subdomain(farm_area, domain, t_farm)

	prob_params = user_params['prob_params']
	prob_params.tidal_farm = t_farm
	farm_problem = SteadySWProblem(prob_params)
	farm_solver = get_basic_solver(farm_problem, continuous)
	farm_control = TurbineFarmControl(t_farm)

	return t_farm, farm_solver, farm_control, farm_problem


def optimize_turbine_farm(turbine_farm, problem_solver, controls, functional, revenue_params, algo_name):
	"""Optimize a discrete, MIPDECO, or continuous turbine farm

	Args:
		turbine_farm: A opentidalfarm Farm object (could be a ContFarm or BinaryFarm)
		problem_solver: An SW solver
		controls: An instance of the TurbineFarmControl() class
		functional: A profit-based functional
		revenue_params: Parameters such as the discount rate, LCOE, and efficiency rate that
						affect revenue and are dependent on the particular algorithm chosen
		algo_name: Name of the algorithm, one of ['cont', 'dicrete, 'mipdeco', 'init_two_step', 'full_two_step']

	Returns:
		output_dict: A dictionary with the algorithm name as the key, and dictionary as its value.
					 The keys of this nested dict are 'profit', 'runtime', 'total_friction', 'num_turbines'
	"""
	iternum = revenue_params['iters']
	tolerance = revenue_params['tolerance']
	rf_params = BinaryReducedFunctional.default_parameters()
	rf_params.automatic_scaling = revenue_params.get('as')
	rf_params.save_checkpoints = revenue_params.get('save_checkpts')
	rf_params.load_checkpoints = revenue_params.get('load_checkpts')

	# note that this supports discrete, continuous, and binary optimization
	rf = get_reduced_functional(functional, controls, problem_solver, rf_params, algo_name)

	if algo_name == 'discrete' or algo_name == 'full_two_step':
		lb, ub = turbine_farm.site_boundary_constraints()
		ieq = turbine_farm.minimum_distance_constraints()
		start = timer.time()
		maximize(rf, bounds=[lb, ub], constraints=ieq, method="SLSQP", options={'maxiter': iternum, 'ftol': tolerance})
		runtime = timer.time() - start
		total_friction = 0
		num_turbines = turbine_farm.number_of_turbines

	elif algo_name == 'cont':

		def cb(w):
			"""callback function for continuous turbine farms"""
			total_friction = assemble(turbine_farm.friction_function * turbine_farm.site_dx(1))
			print "The total friction is:", float(total_friction)

		max_turbine_density = revenue_params['max_smeared_friction']
		init_friction = max_turbine_density / 33
		turbine_farm.friction_function.assign(Constant(init_friction))

		start = timer.time()
		maximize(rf, bounds=[0., max_turbine_density], method="L-BFGS-B", callback=cb, options={'maxiter': iternum,
																								'ftol': tolerance})
		runtime = timer.time() - start
		total_friction = assemble(turbine_farm.friction_function*turbine_farm.site_dx(1))
		num_turbines = total_friction / revenue_params['friction_per_turbine']

	elif algo_name == 'mipdeco':
		lower_bounds, upper_bounds = turbine_farm.binary_constraints(0, 1)
		start = timer.time()
		maximize(rf, bounds=[lower_bounds, upper_bounds], method="L-BFGS-B", options={'maxiter': iternum,
																					  'ftol': tolerance, 'disp': True})
		runtime = timer.time() - start
		num_turbines = sum(turbine_farm.control_array)  # optimal num of turbines is sum of binary variables

		# extract optimal binary variables and their corresponding locations
		binvars = copy.deepcopy(turbine_farm._parameters['binary'])
		locs = copy.deepcopy(turbine_farm._parameters['position'])
		binaries = sorted(zip(binvars, locs), key=lambda x: x[0])
		locs_only = [x[1] for x in binaries[::-1]]
		total_friction = locs_only[0:int(num_turbines)]

	profit = -rf(turbine_farm.control_array)

	if algo_name == 'discrete' or algo_name == 'full_two_step':
		# discrete methods are given a number of turbines and maximize a revenue functional -- need to include cost
		profit = profit - revenue_params['cost'] * num_turbines

	output_dict = {
					algo_name: {
						'profit': profit,
						'runtime': runtime,
						'total_friction': total_friction,
						'num_turbines': num_turbines
					}
	}

	# save file to the cwd
	filename = 'optimal_turbines_{}_algorithm.pvd'.format(algo_name)
	File(filename) << turbine_farm.friction_function

	return output_dict
