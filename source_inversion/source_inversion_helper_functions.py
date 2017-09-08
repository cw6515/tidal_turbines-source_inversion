import copy
from dolfin import *
from dolfin_adjoint import *
import heapq
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
import numpy
import time

 # this ensures that the parameters calculated by the PDE stay in the same index as the mesh coordinates
parameters["reorder_dofs_serial"] = False

# Constant parameters
U0 = Constant(0.0)
S = Constant(3.0)  # Maximum number of activated sources
MINIMUM_BINVAR_THRESHOLD = .1
REFERENCE_CENTROID_LOCATIONS = [[.1, .4], [.8,.8], [.9, .2]]
STARTING_ALPHA = .01
ITERNUM = 200

def get_output_dict(n, m, V):
	"""A dictionary which will be populated with hyperparameters and optimization output

		Args:
			n: n**2 source functions are embedded into the mesh
			m: The mesh will be m x m
			V: The function space of the mesh, source, and reference functions

		Returns:
			d: A dictionary with the parameters for the source inversion problem
	"""
	source_locs = get_source_function_locations(n)
	bcs = DirichletBC(V, U0, boundary)
	sigma = -(source_locs[0][1] - source_locs[1][1])**2 / numpy.log(.01)
	d = {
		 'params':
			{
			 'n': n,
			 'm': m,
			 'V': V,
			 'u_ref': create_reference_solution(V, sigma, bcs),
			 'source_locs': source_locs,
			 'sigma': sigma,
			 'bcs': bcs
			}
		}
	return d

def get_source_function_locations(n):
	"""Get a list of source function locations given source function number parameter n"""
	coords = numpy.linspace(0, 1, n+1)[:-1]
	coords += 0.5 * coords[1]
	coords_list = []
	for a in coords:
		for b in coords:
			coords_list.append([a,b])
	return coords_list

def get_bandb_reference_function(V, reference_function):
	"""Get the reference function to use in the b&b algorithm, as Casadi needs the function assembled in a slightly
	   different way"""
	u = TrialFunction(V)
	v = TestFunction(V)
	a = inner(grad(u), grad(v)) * dx + u * v * ds
	A = assemble(a)

	# Create the linear form for the exact source term and perform assembly
	L = reference_function * v * dx
	b = assemble(L)

	# Solve for the reference solution
	reference_solution = Function(V)
	solve(A, reference_solution.vector(), b)

	return reference_solution

def create_reference_solution(V, sigma, boundary_conditions, b_and_b=False,
							  centroid_locations=REFERENCE_CENTROID_LOCATIONS):
	"""Create the reference function from predefined component centroids in the appropriate function space V"""
	reference_function = create_source_term(centroid_locations[0][0], centroid_locations[0][1], 1, sigma, V, 1)
	second_component = create_source_term(centroid_locations[1][0], centroid_locations[1][1], 1, sigma, V, 1)
	third_component = create_source_term(centroid_locations[2][0], centroid_locations[2][1], 1, sigma, V, 1)

	reference_function.vector()[:] += second_component.vector() + third_component.vector()

	if b_and_b:
		return get_bandb_reference_function(V, reference_function)

	reference_solution = Function(V)
	v_test = TestFunction(V)

	# Solve the reference PDE for the complex function
	F_reference = (inner(grad(v_test), grad(reference_solution)) - reference_function*v_test)*dx
	solve(F_reference == 0, reference_solution, boundary_conditions)

	return reference_solution



def get_mesh(m):
	"""Creates a m x m mesh inside of a unit square"""
	return UnitSquareMesh(m, m)


def boundary(x):
	"""Returns TRUE if a [x,y] point is within the domain of the PDE, False otherwise"""
	return x[0] < DOLFIN_EPS or x[0] > 1 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1 - DOLFIN_EPS


def create_source_term(xloc, yloc, w, sigma, V, degree):
	"""Create a single source term as described in the source inversion problem

		Args:
			xloc: The x-location of the source function centroid
			yloc: the y-location of the source function centroid
			w: the binary coefficient determining activation of the source function
			sigma: The gaussian variance hyperparameter
			V: function space which the expression is interpolated into
			degree: degree of the function space V

		Returns:
			A fenics expression interpolated into function space V
	"""

	return interpolate(Expression('w*100*exp(-(pow(x[0] - xloc,2) + pow(x[1] - yloc,2)) / sigma)', w=w,
								  xloc=xloc, yloc=yloc, sigma=sigma, degree=degree), V)


def solve_pde(V, params, bcs):
	"""Solve the weak formulation of a PDE in a given function space"""
	u = Function(V)
	v = TestFunction(V)
	F = (inner(grad(v), grad(u)) - params*v)*dx
	solve(F == 0, u, bcs)
	return u


def findAlpha(V, reference_solution, n, tolerance=1e-9):
	"""Get the optimal penalty parameter alpha

		Args:
			V: The function space where the source and reference PDEs live
			reference_solution: The solution to the reference PDE
			n: The source function number parameter
			tolerance: tolerance for the SQP source inversion optimization problem with the penalty objective functional

		Returns:
			alpha: The optimal or near-optimal value of the penalty parameter alpha
	"""
	alpha = STARTING_ALPHA
	source_function_locations = get_source_function_locations()
	w = [0.0] * n
	while (sum(w) > S or abs(int(sum(w) + .5) - sum(w)) > .05 or sum(w) < 2.98):
		opt = solve_penalty_function_algorithm(reference_solution, alpha, 20, tolerance, n, source_function_locations)
		w = numpy.array([float(x) for x in opt])
		if sum(w) < 2.98:
			alpha /=  1.5
		else:
			alpha *= 2.0
	return alpha


class nodeConstraint(InequalityConstraint):
	"""This constraint class ensures that the source inversion maximum activated source number constraint is
		satisfied. It inherits from the dolfin_adjoint InequalityConstraint class"""
	def __init__(self, S):
		self.S = float(S)  # ie: the maximum number of turned on binary nodes
		return

	def function(self, w):
		"""The evaluation of the constraint that must be satisfied"""
		print("Evaluting constraint residual")
		return [self.S - sum(w)]  # S - sum_{kl} w_{kl} >= 0

	def jacobian(self, w):
		"""The jacobian of the penalty objective functional wrt binary vector w"""
		print("Computing constraint Jacobian")
		jac = -numpy.ones(len(w))
		return [jac]


def get_binary_bounds(w):
	"""A set of bounds to restrict each w_kl such that 0 < w_kl < 1"""
	return [[0] * len(w), [1] * len(w)]


def iter_cb(w):
	"""A callback function to see the development of w_kl values"""
	print("w = ", w[0:8])
	return


def optimization_info(**output_dict):
	"""Iterate through the output_dict printing algorithm, error, and runtime"""
	for algorithm, values in output_dict.items():
		print ('The {} error is {} and its runtime is '
			  '{}'.format(algorithm, values['error'], values['runtime']))
	plt.show()


def convert_continuous_w_to_integer(w_star, S=int(float(S))):
	"""Convert the optimal continuous values of the w vector to binary values by heuristic rounding

		Args:
			w_star: A numpy array of optimal continuous w_kl values
			S: The upper limit of the number of activated source functions; the largest S components of w_star
				are rounded to 1, and all else to 0
		Returns:
			w_copy: A copy of the optimal continuous vector which has had the largest S elements changed to 1, all
					else set to 0
	"""
	w_copy = copy.deepcopy(w_star)
	largest_continuous = heapq.nlargest(S, w_copy.flatten())
	w_copy[w_copy < largest_continuous[-1]] = 0
	w_copy[w_copy != 0] = 1
	return w_copy


def solve_penalty_function_algorithm(V, reference_solution, alpha, iternum, tolerance, n, source_function_locations,
									 bcs, sigma, w=None):
	"""Solve the source inversion problem with the penalty-augmented objective functional

		Args:
			V: The function space in which the PDEs live
			reference_solution: The soution to the reference PDE
			alpha: The value of penalty parameter alpha
			iternum: Number of iterations for which Scipy's SLSQP  will run
			tolerance: Tolerance for change between the objective functional value in two successive iterations of SLSQP
			n: The source function number parameter
			source_function_locations: x,y locations of the source functions
			bcs: The boundary conditions of the problem
			sigma: The gaussian variance hyperparameter
			w: The optional optimal continuous input vector for the two-step algorithm

		Returns:
			output_dict: A dictionary containing the name of the algorithm, its L1 error, and its runtime
	"""
	if w is None:
		init_w = [1.0]*(n**2)
		w = [Constant(x) for x in init_w]
		algorithm_name = 'penalty'

	else:
		w = [Constant(x) for x in w]
		algorithm_name = 'two_step'

	source_expression_list = [create_source_term(xloc, yloc, 1,  sigma, V, 1) for
							  xloc, yloc in source_function_locations]

	binaries_added = [a * b for a, b in zip(source_expression_list, w)]
	wHat = project(sum(binaries_added), V)
	u = solve_pde(V, wHat, bcs)  # the solution to the source PDE

	# create the penalty functional
	raw_penalty_term = [project(1.0-x, V) for x in w]
	penalty_term_with_binaries = [abs(a * b) for a, b in zip(raw_penalty_term, w)]
	barrier = project(alpha * sum(penalty_term_with_binaries), V)
	J = Functional((0.5*inner(u - reference_solution, u - reference_solution)*dx) + barrier*dx)
  
	w_control = [Control(x) for x in w]
	bounds = get_binary_bounds(w)
	reduced_functional = ReducedFunctional(J, w_control)

	# benchmark for speed
	start = time.time()
	optimal_w = minimize(reduced_functional, method='SLSQP', bounds=bounds, constraints=nodeConstraint(S),
								callback=iter_cb, options={'ftol': tolerance, 'maxiter' : iternum, 'disp': True})
	runtime = time.time() - start

	w_star = numpy.array([round(float(x), 3) for x in optimal_w])

	output_dict = {
					algorithm_name: {
						'binaries': w_star,
						'runtime': runtime
					}
				  }

	return output_dict


def solve_source_inversion(V, reference_solution, iternum, tolerance, n, source_function_locations,
						   boundary_conditions, sigma):
	"""Solve the source inversion problem with the penalty-augmented objective functional

		Args:
			V: The function space in which the PDEs live
			reference_solution: The soution to the reference PDE
			iternum: Number of iterations for which Scipy's SLSQP  will run
			tolerance: Tolerance for change between the objective functional value in two successive iterations of SLSQP
			n: The source function number parameter
			source_function_locations: x,y locations of the source functions
			boundary_conditions: The boundary conditions of the problem
			sigma: The gaussian variance hyperparameter

		Returns:
			output_dict: A dictionary containing the name of the algorithm, its L1 error, and its runtime
	"""
	init_w = [1.0]*(n**2)
	w = [Constant(x) for x in init_w]

	# create the sum of source functions
	source_expression_list = [create_source_term(xloc, yloc, 1, sigma, V, 1)
							  for xloc, yloc in source_function_locations]

	binaries_added = [a * b for a, b in zip(source_expression_list, w)]
	wHat = project(sum(binaries_added), V)
	u = solve_pde(V, wHat, boundary_conditions)

 
	J = Functional((0.5*inner(u - reference_solution, u - reference_solution)*dx))
	w_control = [Control(x) for x in w]
	bounds = get_binary_bounds(w)
	reduced_functional = ReducedFunctional(J, w_control)

	# benchmark for time
	start = time.time()
	optimal_w = minimize(reduced_functional, method='SLSQP', bounds=bounds, constraints=nodeConstraint(S),
						 callback=iter_cb, options={'ftol': tolerance, 'maxiter' : iternum, 'disp': True})
	runtime = time.time() - start

	# optimal binary vector
	w_star = numpy.array([float(x) for x in optimal_w])

	output_dict = {
					'continuous_w_control': {
						'binaries': w_star,
						'runtime': runtime

					},
					'heuristic': {
						'binaries': convert_continuous_w_to_integer(w_star),
						'runtime': runtime
					}

				  }

	return output_dict


def get_plots_and_error(output_dict):
	"""This function calculates the error and provides visualization of the
	   source and reconstruction PDEs

	   Args:
		   output_dict:  a dictionary containing all hyperparameters as well as
						 output from running one of the source inversion algorithms

	   Returns:
		   output_dict: An updated version of the input 'output_dict', with a figure depicting error as a contour plot
						and a figure depicting the solutions to the reconstruction PDE and reference PDE

		This function returns two graphs, one being a visual 3-d plot of the reconstructed solution to the
		reference PDE, and one being a 2-d error heatmap displaying the difference between the reconstructed solution
		and reference solution at each mesh vertex
	"""
	params = output_dict.pop('params')
	V, m, n = params['V'], params['m'], params['n']
	bcs, sigma = params['bcs'], params['sigma']
	source_function_vertices = params['source_locs']
	reference_solution = params['u_ref']

 

	for algorithm, values in output_dict.items():
		activated_source_list = []
		optimal_params = values['binaries']
		title = 'n = {}, m = {}, {}'.format(n, m, algorithm)

		# create tuple of (source_location_x, source_location_y, w_kl(xy))
		for a in range(len(source_function_vertices)):
			activated_source_list.append([source_function_vertices[a][0], source_function_vertices[a][1],
										  optimal_params[a]])

		optimal_source_functions = [create_source_term(xk, yk, w_k, sigma, V, 1)
									for xk, yk, w_k in activated_source_list]

		# recreate the reference PDE using the optimal w_kl parameters
		approx = optimal_source_functions[0]
		for elem in optimal_source_functions[1:]:
			approx.vector()[:] += elem.vector()

		u = solve_pde(V, approx, bcs)

		# find the error at each mesh vertex
		error = u.vector().array() - reference_solution.vector().array()

		# create plots
		figure, heat_map = create_plots(u, reference_solution, m, error, title, activated_source_list)

		abs_err = (1.0/m**2) * sum(abs(error))
		output_dict[algorithm]['error'] = abs_err
		output_dict[algorithm]['figure'] = figure
		output_dict[algorithm]['heat_map'] = heat_map

	return output_dict


def create_plots(u, u_ref, m, error, title, activated_source_list):
	"""Constructs a 3-d figure comparing the reconstruction solution and the reference solution,
		as well as an error heatmap
	"""
	# construct a mesh to plot upon
	mesh_side = numpy.linspace(0, 1, m+1)
	X, Y = numpy.meshgrid(mesh_side, mesh_side)

	# create a plot with the reference and reconstruction solutions
	fig = plt.figure()
	ax = fig.add_subplot(121, projection='3d')
	ax.set_xlabel('X coordinate')
	ax.set_ylabel('Y coordinate')
	ax.set_zlabel('Source Function Value')
	ax.set_title('Reconstruction of Reference Solution with %s' %(title))
	ax.plot_surface(X, Y, u.vector().array().reshape((m+1, m+1)), cmap='viridis', rstride=1, cstride=1,
					antialiased=True, alpha=None, linewidth=0)

	ax = fig.add_subplot(122, projection='3d')
	ax.set_xlabel('X coordinate')
	ax.set_ylabel('Y coordinate')
	ax.set_zlabel('Reference Function Value')
	ax.set_title('Reference Solution')
	ax.plot_surface(X, Y, u_ref.vector().array().reshape((m+1, m+1)), cmap='viridis', rstride=1, cstride=1,
					antialiased=True, alpha=None, linewidth=0)

	# create a contour plot of error over the mesh
	fig2 = plt.figure()
	ax1 = fig2.gca()
	ax1.set_xlabel('X coordinate')
	ax1.set_ylabel('Y coordinate')
	ax1.set_title('%s Reconstruction Error' %(title))
	levels = numpy.linspace(min(error), max(error), 20)
	cs = ax1.contourf(X, Y, error.reshape((m+1, m+1)), levels=levels , cmap='viridis', linewidth =.5)

	# plot the locations and strengths of the activated sources
	for activated_source in activated_source_list:
		if activated_source[2] > MINIMUM_BINVAR_THRESHOLD:
			plt.plot(activated_source[0], activated_source[1], 'x', markeredgewidth=3, markersize=10, color='black')
			ax1.annotate(str(round((activated_source[2]),2)), xy=(activated_source[0], activated_source[1]),
						 xytext=(activated_source[0]+.01, activated_source[1]+.01))

	# plot locations of reference function centroids
	for location in REFERENCE_CENTROID_LOCATIONS:
		plt.plot(location[0], location[1], 'o', markeredgewidth = 3, markersize = 10, color = 'black')

	fig2.colorbar(cs, ax=ax1, format="%.2f")

	return fig, fig2
