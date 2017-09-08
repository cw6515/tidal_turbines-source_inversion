import click
from source_inversion.source_inversion_helper_functions import *
from source_inversion.casadi_bnb import *

@click.command()
@click.option('--n', default=4, help='Please define n, where n**2 is the total number of source functions')
@click.option('--m', default=16, help='Please define m, where the domain of this problem is an m x m unit square')
@click.option('--alpha/--no-alpha', default=True, help='Set the value of alpha')
@click.option('--cont/--no-cont', default=True, help="Runs the continuous (w control) algorithm)")
@click.option('--penalty/--no-penalty', default=False,  help='Runs the penalty functional algorithm')
@click.option('--two_step/--no-two_step', default=False, help='Runs the two-step algorithm')
@click.option('--b_and_b/--no-b_and_b', default=False, help='Runs the branch and bound algorithm')
@click.option('--tolerance', default=1e-4, help='Tolerance used by all algorithms')
def run_source_inversion_pipeline(n, m, alpha, cont, penalty, two_step, b_and_b, tolerance):
    mesh = get_mesh(m)
    V = FunctionSpace(mesh, 'CG', 1)
    output_dict = get_output_dict(n, m, V)
    sigma = output_dict['params']['sigma']
    reference_solution = output_dict['params']['u_ref']
    source_function_locations = output_dict['params']['source_locs']
    bcs = output_dict['params']['bcs']

    # set alpha to a default value if not pre-specified
    if alpha:
        alpha = .016

    if (penalty or two_step) and not alpha:
        # this function can take forever to run if using an n > 4. I would recommend manually specifying alpha
        alpha = findAlpha(V, reference_solution, n, tolerance=1e-5)

    if cont:
        output_dict.update(solve_source_inversion(V, reference_solution, ITERNUM, tolerance, n,
                                                  source_function_locations, bcs, sigma))

    if two_step:

        if not cont:
            raise AssertionError('The continuous algorithm cannot be disabled if you want to run the two_step'
                                 ' algorithm. Please enable the continuous algorithm by removing --no-cont')

        optimal_cont_w = output_dict['continuous_w_control']['binaries']
        output_dict.update(solve_penalty_function_algorithm(V, reference_solution, alpha, ITERNUM,
                                                            tolerance, n, source_function_locations, bcs, sigma,
                                                            w=optimal_cont_w))
        # two_step algorithm runtime must include the continuous algorithm (w control) runtime
        two_step_time = output_dict['two_step']['runtime']
        output_dict['two_step']['runtime'] = two_step_time + output_dict['continuous_w_control']['runtime']

    if penalty:
        output_dict.update(solve_penalty_function_algorithm(V, reference_solution, alpha, ITERNUM,
                                                            tolerance, n, source_function_locations, bcs, sigma))

    if b_and_b:
        reference_solution = create_reference_solution(V, sigma, bcs, b_and_b=True)
        output_dict.update(solve_bnb_algorithm(V, reference_solution, n, source_function_locations, sigma))

    # get error and plots
    updated_output_dict = get_plots_and_error(output_dict)

    # display error and runtime info
    optimization_info(**updated_output_dict)

# run the source inversion pipeline with --args
if __name__ == '__main__':
    run_source_inversion_pipeline()