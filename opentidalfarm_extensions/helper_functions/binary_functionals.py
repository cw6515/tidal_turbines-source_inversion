from opentidalfarm import *

class BinaryCostFunctional(PrototypeFunctional):
	r""" Implements a cost functional of the form:
 
	.. math:: J(u, m) = \int c_t~ dx,
 
	where :math:`c_t` is the friction due to the turbines.
 
	:param problem: The problem for which the functional is being computed.
	:type problem: Instance of the problem class.
	"""
	def __init__(self, problem):
		self.farm = problem.parameters.tidal_farm
		self.farm.update()


	def cost_converter(self):
		"""A hacky fix to get the ratio between the desired cost and the actual cost"""
		# the sum of all binary variables (cost should be cost_per_turbine * sum(binvars))
		desired = sum(self._cost().vector().array())

		# need to use this as we are integrating and can't have a scalar
		actual = assemble(self._cost()*self.farm.site_dx(1))
		cost_coefficient = desired / actual

		return cost_coefficient


	def Jt(self, state, turbine_field):
		""" Computes the cost of the farm based on the value of the binary parameters and current state"""
		cost_coeff = self.cost_converter()
		total_cost = cost_coeff * (self._cost() * self.farm.site_dx(1))

		return total_cost
 

	def _cost(self):
		""" Computes the cost"""
		return self.farm.turbine_cache['final_binary']



class BinaryRevenueFunctional(PowerFunctional):
	r"""Implements a revenue-based functional"""

	def __init__(self, problem, user_params, cut_in_speed=None, cut_out_speed=None, eps=1e-10):
		super(BinaryRevenueFunctional, self).__init__(problem)
		self.discount_rate = user_params['discount_rate']
		self.efficiency = user_params['efficiency']
		self.timesteps = user_params['timesteps']
		self.income_per_unit = user_params['I']

	def Jt(self, state, turbine_field):
		""" Computes the power output of the farm"""
		return self.discounted_power(state, turbine_field)*self.farm.site_dx(1)

	def discounted_power(self, state, turbine_field):
		"""A time-discounted revenue output"""
		eff = self.efficiency * self.income_per_unit * (1.0 / 1e6)
		numerator = eff * 24 * 365 * self.rho * turbine_field * self._speed_squared(state)**1.5
		time = []

		# discount for each timestep
		revenue = 0
		for t in range(0, self.timesteps):
			time.append((1 + self.discount_rate)**t)

		for t in time:
			revenue += numerator / t

		return revenue
