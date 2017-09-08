import copy
from math import pi

import dolfin_adjoint
from opentidalfarm import *


class BinaryModelTurbine(object):
	"""A model turbine for MIPDECO problems. Params may be overridden by `run_tidal_turbine_pipeline` user params"""
	def __init__(self,
				 blade_radius=8.35,
				 efficiency=.5,
				 velocity=2.0,
				 discount_rate=.1,
				 timesteps=5,
				 LCOE=107.89,
				 minimum_distance=40,
				 water_density=1e3,
				 Ct=0.86):

		# Radius of turbine blades in m
		self.blade_radius = blade_radius

		# Minimum distance between two turbines in m
		self.minimum_distance = minimum_distance

		# Water density in kg/m^3
		self.water_density = water_density

		# Turbine thrust coefficient
		self.Ct = Ct

		# efficiency of power extraction
		self.efficiency = efficiency

		# peak velocity of flow
		self.peak_velocity = velocity

		# discount rate
		self.discount_rate = discount_rate

		# number of time periods
		self.timesteps = timesteps

		# Levelized cost of Energy
		self.LCOE = LCOE

	@property
	def blade_diameter(self):
		''' Returns the turbine diameter. '''
		return 2 * self.blade_radius

	@property
	def turbine_cross_section(self):
		''' Returns the area that the turbine blades cover. '''
		return pi * self.blade_radius ** 2

	@property
	def maximum_density(self):
		''' Returns the maximum turbine density'''
		dt_max = 1. / self.minimum_distance ** 2
		return dt_max

	@property
	def maximum_smeared_friction(self):
		''' Returns the average friction of a densely packed turbine area.'''
		return self.friction * self.maximum_density

	@property
	def bump_height(self, apply_roc_correction=True):
		''' Returns the peak value of OpenTidalFarm's turbine representation.
			Note: The amount of friction in the smeared and discrete representation
			are not the same. Instead the computation is based on setting the applied
			force on the flow equal.

			This is based on:
			F_turbine = 0.5 * rho * C_t * A_t u_upstream**2                           # Thrust force of a single turbine
					  = 0.5 * rho * C_T * 4 / (1 + sqrt(1-C_T))**2 A_T u_turbine**2   # Apply Rocs correction
					  = rho int bump_function(x) c_t u**2                             # Set the force equal to the force of a discrete numpy turbine
					  is approximately
						rho int bump_function(x) c_t u_turbine**2                     # Assume that u and u_turbine are the same

		   =>
		   c_t = 4/(1+sqrt(1-C_T))**2 C_T A_T / (2 \int bump_function(x))
		'''

		roc_correction_factor = 4. / (1 + (1 - self.Ct) ** 0.5) ** 2
		A_T = self.turbine_cross_section
		int_bump_function = 1.45661 * self.blade_radius ** 2

		c_t = self.Ct * A_T / (2. * int_bump_function)
		if apply_roc_correction:
			c_t *= roc_correction_factor

		return c_t


	@property
	def discounted_cost_coefficient(self):
		"""Returns the total lifetime cost of a single turbine with the LCOE, efficiency of energy extraction
		   and discount rate into account"""

		# yearly energy produced in MWh
		yearly_energy = (.5 * self.turbine_cross_section * self.Ct * self.peak_velocity ** 3 * 24 * 365) / 1e6

		# numerator is in GBP
		num = self.water_density * self.LCOE * yearly_energy

		time = []

		# discount for each timestep
		cost = 0
		for t in range(0, self.timesteps):
			time.append((1 + self.discount_rate) ** t)

		for t in time:
			cost += num / t

		return cost


	def number_of_turbines(self, friction_integral):
		return friction_integral / self.friction


	@property
	def friction(self):
		return self.Ct * self.turbine_cross_section / 2.

	def __str__(self):
		s = """Model turbine specification
---------------------------
	Blade diameter                                   {} m.
	Minimum distance between two turbines:           {} m.
	Turbine cross section (A_T)                      {} m^2.
	Turbine induced friction:                        {}.
	Average turbine friction of densely packed farm: {}.
	Maximum turbine density:                         {}.
	OpenTidalFarm bump maximum:                      {}.
	OpenTidalFarm cost coefficient:                  {}.
		""".format(self.blade_diameter,
				   self.minimum_distance,
				   self.turbine_cross_section,
				   self.friction,
				   self.maximum_smeared_friction,
				   self.maximum_density,
				   self.bump_height,
				   self.discounted_cost_coefficient)
		return s


class BinaryControls(Controls):
	"""Specifies the controls for optimisation.
	This class extends the control class, allowing binary variables to be controls.
	The user initializes this class with their desired control parameters.
	"""
	def __init__(self, position=False, friction=False, dynamic_friction=False, binary=False):
		"""Initialize with the desired controls."""

		self._controls = {"position": False,
						  "friction": False,
						  "dynamic_friction": False,
						  "binary": False}

		def _process(key, value):
			"""Check value is of type bool. Raise ValueError if it is not."""
			try:
				assert isinstance(value, bool)
				# Change the control value in the dictionary.
				self._controls[key] = value
			# Raise an error if a boolean was not given.
			except AssertionError:
				raise ValueError("%s must be a boolean (%s)." %
								 (key.capitalize(), str(type(value))))

		# Process the given values
		_process("position", position)
		_process("friction", friction)
		_process("dynamic friction", dynamic_friction)
		_process("binary", binary)

	@property
	def binary(self):
		"""Whether binary variables are enabled as a control parameter"""
		return self._controls["binary"]


class NewBaseTurbine(object):
	"""A base turbine class from which others are derived, modified to add binary turbine support."""
	def __init__(self, friction=None, diameter=None, minimum_distance=None,
				 controls=None, binary=False, bump=False, smeared=False, thrust=False,
				 implicit_thrust=False):
		# Possible turbine parameters.
		self._diameter = diameter
		self._minimum_distance = minimum_distance
		self._friction = friction
		self._controls = controls

		# Possible parameterisations.
		self._binary = binary
		self._bump = bump
		self._smeared = smeared
		self._thrust = thrust
		self._implicit_thrust = implicit_thrust
		self._unit_bump_int = 1.45661


	@property
	def friction(self):
		"""The maximum friction coefficient of a turbine.
		:returns: The maximum friction coefficient of the turbine.
		:rtype: float
		"""
		if self._friction is None:
			raise ValueError("Friction has not been set!")
		return self._friction


	@property
	def diameter(self):
		"""The diameter of a turbine.
		:returns: The diameter of a turbine.
		:rtype: float
		"""
		if self._diameter is None:
			raise ValueError("Diameter has not been set!")
		return self._diameter


	@property
	def radius(self):
		"""The radius of a turbine.
		:returns: The radius of a turbine.
		:rtype: float
		"""
		return self.diameter*0.5


	@property
	def minimum_distance(self):
		"""The minimum distance allowed between turbines.
		:returns: The minimum distance allowed between turbines.
		:rtype: float
		"""
		if self._minimum_distance is None:
			raise ValueError("Minimum distance has not been set!")
		return self._minimum_distance


	@property
	def integral(self):
		"""The integral of the turbine bump function.
		:returns: The integral of the turbine bump function.
		:rtype: float
		"""
		return self._unit_bump_int*self._diameter/4.


	def _set_controls(self, controls):
		"""Set the type of controls"""
		self._controls = controls

	def _get_controls(self):
		"""Get the type of controls"""
		if self._controls is not None:
			return self._controls
		else:
			raise ValueError("The controls have not been set.")

	controls = property(_get_controls, _set_controls, "The turbine controls.")


	@property
	def bump(self):
		return self._bump

	@property
	def binary(self):
		return self._binary

	@property
	def smeared(self):
		return self._smeared

	@property
	def thrust(self):
		return self._thrust

	@property
	def implicit_thrust(self):
		return self._implicit_thrust


class BinaryTurbine(NewBaseTurbine):
	"""A binary turbine class with default values for friction and diameter -- this overriden by user_params"""

	def __init__(self, friction=12.0, diameter=20., minimum_distance=None, controls=BinaryControls(binary=True)):
		# Check for a given minimum distance.
		if minimum_distance is None:
			minimum_distance = diameter * 1.5

		# Initialize the base class.
		super(BinaryTurbine, self).__init__(friction=friction,
											diameter=diameter,
											minimum_distance=minimum_distance,
											controls=controls,
											binary=True)


class FarmDomain(SubDomain):
	"""A class which enables a (non-binary) farm to discriminate between points inside and outside the farm subdomain"""

	def __init__(self, x_low, x_high, y_low, y_high):
		super(FarmDomain, self).__init__()
		self.x_low = x_low
		self.x_high = x_high
		self.y_low = y_low
		self.y_high = y_high

	def inside(self, x, on_boundary):
		return (self.x_low <= x[0] <= self.x_high and
				self.y_low <= x[1] <= self.y_high)



class ContFarm(Farm):
	"""Extends the opentidalfarm Farm class. Includes a method to convert continuous to discrete solutions with
	minimum distance and within-farm-domain checking enforced.
	"""
	def checkdistance(self, x, locations, mindist):
		"""Checks euclidean distance between new turbine and all
		turbines whose coordinates have already been selected"""

		for loc in locations:
			if (numpy.sqrt((x[0] - loc[0])**2 + (x[1] - loc[1])**2)) < mindist:
				return False
		return True


	def inDomain(self, x):
		"""Checks if a given point is in the farm area of the domain"""
		if self.subDomain is None:
			raise ValueError("A subdomain has not been specified.")

		return (self.subDomain[0] <= x[0] <= self.subDomain[1] and
				self.subDomain[2] <= x[1] <= self.subDomain[3])


	def placediscrete(self, num_turbines, dbar, mindist):
		"""Randomly select points on the mesh and place a turbine/adds coordinates
		 to a list with P(turbine_density(x,y) / maximum_turbine_density) if minimum
		 distance constraints are satisfied
		"""

		x_locs = []
		y_locs = []
		indices = numpy.arange(self.domain.mesh.num_vertices())
		numpy.random.shuffle(indices)
		placed = 0.
		iternum = 0.

		while (placed < num_turbines and iternum < 15):
			for i in indices:

				if placed == num_turbines:
					self.update()
					return x_locs, y_locs

				potential_loc = self.domain.mesh.coordinates()[i]

				if self.inDomain(potential_loc):
					prob = self.turbine_cache['turbine_field'](self.domain.mesh.coordinates()[i])/dbar
					fair_toss = numpy.random.random()

					if prob > fair_toss:
						placed_locs = zip(x_locs, y_locs)
						if self.checkdistance(potential_loc, placed_locs, mindist):
							x_locs.append(potential_loc[0])
							y_locs.append(potential_loc[1])
							placed += 1.0
							self.update()
			iternum += 1

		return x_locs, y_locs


class BinaryFarm(Farm):
	"""Extends opentidalfarm farm class by adding a list of binary variables to the farm parameters. These binary
	   variables also have support in all related farm functions so that this farm can be use with the appropriate
	   functional in a MIPDECO framework.
	"""
	def __init__(self, domain, turbine, site_ids=None, order=2,
				 n_time_steps=None):
		"""Initializes an empty binary farm."""
		# Initialize the base clas
		super(BinaryFarm, self).__init__(domain, turbine, site_ids, n_time_steps=None)

		self.n_time_steps = n_time_steps

		# Create a (binary) turbine function space and add binary controls to the farm parameters
		self.turbine_cache = BinaryTurbineCache()
		self._parameters = {"friction": [], "position": [], "binary": []}


		# set turbine function space in this class and in the cache
		self._set_turbine_specification(turbine)
		self._turbine_function_space = FunctionSpace(self.domain.mesh, "CG", order)
		self.turbine_cache.set_function_space(self._turbine_function_space)


	def update(self):
		self.turbine_cache.update(self)

	@property
	def friction_function(self):
		self.update()
		return self.turbine_cache["turbine_field"]

	@property
	def final_binary_function(self):
		"""A cache to return the current values of the binary variables"""
		self.update()
		return self.turbine_cache['final_binary']


	def inDomain(self, x):
		"""Checks if a given point is in the farm area of the domain"""

		if self.subDomain is None:
			raise ValueError("A subdomain has not been specified.")

		return (self.subDomain[0] <= x[0] <= self.subDomain[1] and
				self.subDomain[2] <= x[1] <= self.subDomain[3])


	def addBinaryTurbine(self, coordinates, init_var):
		"""Add a turbine to the farm at the given coordinates. Creates a new turbine of the same specification
		   as the prototype turbine and places it at coordinates.

		   Args:
			   coordinates: An [x,y] list of coordinates describing the location that a binary variable should be
							tethered to
			   init_var: The initial value of the binary coefficient
		"""
		if self._turbine_specification is None:
			raise ValueError("A turbine specification has not been set.")

		# add a turbine at the given coordinates and a binary variable at the corresponding index
		turbine = self._turbine_specification
		self._parameters["position"].append(coordinates)
		self._parameters["binary"].append(init_var)
		self._parameters["friction"].append(turbine.friction)

		dolfin.info("Turbine added at (%.2f, %.2f)." % (coordinates[0],coordinates[1]))

		return


	def create_binaryField(self):
		"""A method to create an entire binary field from scratch -- by randomly iterating over all mesh
		   coordinates. This is often a bad idea -- use `addBinaryTurbine` with pre-chosen locations instead."""
		coords = self.domain.mesh.coordinates()
		for c in coords:
			if self.inDomain(c):
				locs = self._parameters['position']
				binary = self._parameters['binary']
				if self.checkdistance(c, locs, binary, self.turbine_specification.minimum_distance):
					num = numpy.random.random()
					self.addBinaryTurbine(c, num)
				else:
					self.addBinaryTurbine(c, 0.0)

		# update the cache
		self.update()
		return


	@property
	def control_array(self):
		"""Returns the farm control array -- a list of the current value of the binary variables"""
		m = []
		m += numpy.reshape(self._parameters["binary"], -1).tolist()

		return numpy.asarray(m)


	def binary_constraints(self, lower_bounds, upper_bounds):
		if upper_bounds is None:
			raise ValueError("Please supply upper bounds for the binary control variables.")

		n_turbines = len(self._parameters['binary'])

		lower_bounds = n_turbines*[dolfin_adjoint.Constant(lower_bounds)]
		upper_bounds = n_turbines*[dolfin_adjoint.Constant(upper_bounds)]
		return lower_bounds, upper_bounds


	def minimum_distance_constraints(self, large=False):
		"""Returns an instance of MinimumDistanceConstraints. This has not yet been implemented for the binary case.
		"""
		# TODO(christian): implement for a BinaryFarm

		# Check we have some turbines.
		n_turbines = len(self.turbine_positions)
		if (n_turbines < 1):
			raise ValueError("Turbines must be deployed before minimum "
							 "distance constraints can be calculated.")
		if large:
			raise NotImplementedError("Large Arrays minimum constraints not implemented for binary case!")
		else:
			raise NotImplementedError('Inequality constraints are not implemented for the binary case!')

	@property
	def control_array_global(self):
		"""A serialized representation of the farm based on the controls.
		:returns: A serialized representation of the farm based on the controls.
		:rtype: numpy.ndarray
		"""
		return self.control_array


class BinaryTurbineFunction(object):
	"""A replacement of opentidalfarm's TurbineFunction object, with added binary support"""
	def __init__(self, cache, V, turbine_specification):
		self._parameters = copy.deepcopy(cache._parameters)
		self._turbine_specification = turbine_specification
		self._cache = cache

		# Precompute some turbine parameters for efficiency.
		self.x = interpolate(Expression("x[0]", degree=1), V).vector().array()
		self.y = interpolate(Expression("x[1]", degree=1), V).vector().array()
		self.V = V


	def __call__(self, name="", derivative_index=None, derivative_var=None,
				 timestep=None):
		"""If the derivative selector is i >= 0, the Expression will compute the
		derivative of the turbine with index i with respect to either the x or y
		coorinate or its friction parameter. """

		params = self._parameters

		if derivative_index is None:
			position = params["position"]
			binary = params["binary"]

			if timestep is None:
				friction = params["friction"]
			else:
				friction = params["friction"][timestep]
		else:
			binary = [params["binary"][derivative_index]]
			position = [params["position"][derivative_index]]

			if timestep is None:
				friction = [params["friction"][derivative_index]]
			else:
				friction = [params["friction"][timestep][derivative_index]]

		# Infeasible optimisation algorithms (such as SLSQP) may try to evaluate
		# the functional with negative turbine_frictions. Since the forward
		# model would crash in such cases, we project the turbine friction
		# values to positive reals.
		friction = [max(0, f) for f in friction]
		ff = numpy.zeros(len(self.x))

		# Ignore division by zero.
		numpy.seterr(divide="ignore")
		eps = 1e-12
		var = zip(position, binary, friction)
		for (x_pos, y_pos), binvar, fric in var:
			radius = self._turbine_specification.radius
			x_unit = numpy.minimum(
				numpy.maximum((self.x-x_pos)/radius, -1+eps), 1-eps)
			y_unit = numpy.minimum(
				numpy.maximum((self.y-y_pos)/radius, -1+eps), 1-eps)

			# Apply chain rule to get the derivative with respect to the binary varialbes.
			exp = numpy.exp(-1./(1-x_unit**2)-1./(1-y_unit**2)+2)

			if (derivative_index is None and derivative_var != 'binary_sum'):
				ff += binvar*exp*fric

			# this is a hacky way to get the value of the binary variables into the cost functional
			elif derivative_var == "binary_sum":
				ff[0] += binvar

			if derivative_var == "binary":
				ff += fric*exp

		# Reset numpy to warn for zero division errors.
		numpy.seterr(divide="warn")
		f = Function(self.V, name=name, annotate=False)
		f.vector().set_local(ff)
		f.vector().apply("insert")
		return f


class BinaryTurbineCache(dict):
	"""A replacement for opentidalfarm's TurbineCache, stripping out friction and positional optimization support
		in lieu of binary support"""
	def __init__(self, *args, **kw):
		super(BinaryTurbineCache, self).__init__(*args, **kw)
		self.itemlist = super(BinaryTurbineCache, self).keys()
		self._function_space = None
		self._specification = None
		self._controlled_by = None
		self._parameters = None
		self.only_friction = None

	def __setitem__(self, key, value):
		self.itemlist.append(key)
		super(BinaryTurbineCache,self).__setitem__(key, value)

	def __iter__(self):
		return iter(self.itemlist)

	def keys(self):
		return self.itemlist

	def values(self):
		return [self[key] for key in self]

	def itervalues(self):
		return (self[key] for key in self)

	def set_function_space(self, function_space):
		self._function_space = function_space

	def set_turbine_specification(self, specification):
		self._specification = specification
		self._controlled_by = specification.controls


	def update(self, farm):
		"""Creates a list of all turbine function/derivative interpolations.
		This list is used as a cache to avoid the recomputation of the expensive
		interpolation of the turbine expression."""

		try:
			assert(self._specification is not None)
		except AssertionError:
			raise ValueError("The turbine specification has not yet been set.")

		binary = farm._parameters["binary"]
		friction = farm._parameters["friction"]
		position = farm._parameters["position"]


		# If the parameters have not changed, there is nothing to do
		if self._parameters is not None:
			if (len(self._parameters["binary"]) == len(binary) and
				(self._parameters["binary"]==binary).all()):
				return

		else:
			self._parameters = {"friction": [], "position": [], "binary":[]}

		# Update the cache.
		log(INFO, "Updating the turbine cache")

		# Update the binary variables.
		self._parameters["binary"] = numpy.copy(binary)
		self._parameters["friction"] = numpy.copy(friction)
		self._parameters["position"] = numpy.copy(position)

		turbines = BinaryTurbineFunction(self, self._function_space, self._specification)

		# calculate the value of the turbine field and the value of the binary params
		self["turbine_field"] = turbines(name="turbine_friction_cache")
		self['final_binary'] = turbines(name='final_binary', derivative_var='binary_sum')
		self["turbine_field_individual"] = [BinaryTurbineFunction(self, self._function_space, self._specification)()]

		# Precompute the derivatives with respect to the binary variable of each turbine.
		self["turbine_derivative_binary"] = []

		for n in xrange(len(self._parameters["binary"])):
			tfd = turbines(derivative_index=n,
						   derivative_var ="binary",
						   name=("turbine_fricition_derivative_with_"
								 "respect_to_binary_controls_" + str(n)))
			self["turbine_derivative_binary"].append(tfd)
