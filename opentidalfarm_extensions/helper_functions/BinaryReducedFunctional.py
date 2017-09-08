import dolfin_adjoint
from dolfin_adjoint import *
import numpy
from opentidalfarm import ReducedFunctional


class BinaryReducedFunctional(ReducedFunctional):
	""" This class extends the Reduced Functional class in order to support optimization with respect to binary controls
	"""
	def __init__(self, functional, controls, solver, parameters):
		super(BinaryReducedFunctional, self).__init__(functional, controls, solver, parameters)


	def _update_turbine_farm(self, m):
		""" Update the turbine farm from the flattened parameter array m. """
		farm = self.solver.problem.parameters.tidal_farm
		farm._parameters["binary"] = m[0:]

		# Update the farm cache.
		farm.update()


	def _compute_gradient(self, m, forget=True):
		""" Compute the functional gradient for the binary variables control array """
		farm = self.solver.problem.parameters.tidal_farm

		# If any of the parameters changed, the forward model needs to be re-run
		if self.last_m is None or numpy.any(m != self.last_m):
			self._compute_functional(m, annotate=True)

		J = self.time_integrator.dolfin_adjoint_functional(self.solver.state)

		# Output power
		if self.solver.parameters.dump_period > 0:

			if self._solver_params.output_turbine_power:
				turbines = farm.turbine_cache["turbine_field"]
				power = self.functional.power(self.solver.state, turbines)
				self.power_file << project(power, farm._turbine_function_space, annotate=False)

		parameters = FunctionControl("turbine_friction_cache")
		djdtf = dolfin_adjoint.compute_gradient(J, parameters, forget=forget)
		dolfin.parameters["adjoint"]["stop_annotating"] = False
		dj = []

		for tfd in farm.turbine_cache["turbine_derivative_binary"]:
			farm.update()
			dj.append(djdtf.vector().inner(tfd.vector()))

		dj = numpy.array(dj)

		return dj
