from source_inversion_helper_functions import *
import scipy, scipy.sparse
import casadi

# Simple Branch and Bound solver
def solve_bnb(solver, root, integers):
    """A branch and bound solver which uses IPOPT as the IP subproblem module"""
    # Initialize the node queue
    Q = [(numpy.inf, root)]
 
    # Global solution data
    x_opt = None
    f_opt = numpy.inf
 
    # Continuous solution data
    x0 = None
    f0 = numpy.inf
 
    # Main loop
    seqnum = 0
    while Q:
        seqnum += 1
 
        # Extract node with lowest lower bound and check if fathoming is necessary
        # (should not happen)
        node = Q.pop()
        if f_opt < numpy.inf and node[0] >= f_opt:
            print "Node %4d (%4d left): %9g >= %9g - PRE-FATHOM" % (seqnum, len(Q), node[0], f_opt)
            continue
        node = node[1]
 
        # Solve the node
        result = solver(lbx=node['lbx'], ubx=node['ubx'], lbg=node['lbg'], ubg=node['ubg'])
 
        # Local solution data
        x = result['x']
        f = result['f']
 
        # Store continuous solution data if none has been stored yet
        if x0 is None:
            x0 = casadi.DM(x)
            f0 = f
 
        # Check if branch can be fathomed
        if f >= f_opt:
            print "Node %4d (%4d left): %9g >= %9g - POST-FATHOM" % (seqnum, len(Q), f, f_opt)
            continue
 
        # Check for violations of integrality (fixed tolerance 1e-5)
        viol = [abs(casadi.floor(x[i] + 0.5) - x[i]) for i in integers]
        idx = [(integers[i], viol[i]) for i in range(len(integers)) if viol[i] > 1e-5]
        if not idx:
            # Register new global solution
            x_opt = x
            f_opt = f
 
            # Cull the branch and bound tree
            pre = len(Q)
            Q = [n for n in Q if n[0] < f_opt]
            post = len(Q)
 
            print "Node %4d (%4d left): f_opt = %9g - *** SOLUTION *** (%d culled)" % (seqnum, len(Q), f, pre - post)
        else:
            # Branch on first violation
            idx = idx[0][0]
 
            # Generate two new nodes (could reuse node structure)
            ln = {
                'x0':     casadi.DM(x),
                'lbx':    node['lbx'],
                'ubx':    casadi.DM(node['ubx']),
                'lbg':    node['lbg'],
                'ubg':    node['ubg'],
                'lam_x0': casadi.DM(result['lam_x']),
                'lam_g0': casadi.DM(result['lam_g'])
            }
            un = {
                'x0':     casadi.DM(x),
                'lbx':    casadi.DM(node['lbx']),
                'ubx':    node['ubx'],
                'lbg':    node['lbg'],
                'ubg':    node['ubg'],
                'lam_x0': casadi.DM(result['lam_x']),
                'lam_g0': casadi.DM(result['lam_g'])
            }
 
            ln['ubx'][idx] = casadi.floor(x[idx])
            un['lbx'][idx] = casadi.ceil(x[idx])
 
            lower_node = (f, ln)
            upper_node = (f, un)
 
            # Insert new nodes in queue (inefficient for large queues)
            Q.extend([lower_node, upper_node])
            Q.sort(cmp=lambda x,y: cmp(y[0], x[0]))
            print "Node %4d (%4d left): %9g - BRANCH ON %d" % (seqnum, len(Q), f, idx)
 
    return {
        'x0': x0,
        'f0': f0,
        'x': x_opt,
        'f': f_opt
    }
 
 
# Converts FEniCS matrix to scipy.sparse.coo_matrix
def matrix_to_coo(A, shape=None):
    nnz = A.nnz()
    col = numpy.empty(nnz, dtype='uint64')
    row = numpy.empty(nnz, dtype='uint64')
    val = numpy.empty(nnz)
 
    if shape is None:
        shape = (A.size(0), A.size(1))
 
    it = 0
    for i in range(A.size(0)):
        col_local, val_local = A.getrow(i)
        nnz_local = col_local.shape[0]
 
        col[it:it+nnz_local] = col_local[:]
        row[it:it+nnz_local] = i
        val[it:it+nnz_local] = val_local[:]
        it += nnz_local
 
    return scipy.sparse.coo_matrix((val, (row, col)), shape=shape)


def solve_bnb_algorithm(V, reference_solution, n, source_function_locations, sigma):
    """Solve the source inversion problem with the the CasADi branch and bound algorithm

        Args:
            V: The function space in which the PDEs live
            reference_solution: The soution to the reference PDE
            n: The source function number parameter
            source_function_locations: x,y locations of the source functions
            sigma: The gaussian variance hyperparameter

        Returns:
            output_dict: A dictionary with the algorithm name, runtime, and optimal binary coefficients,
                         in the form: {algorithm_name: {'binvar': optimal_binary_vars, runtime': runtime}}
    """
    # create the sum of source functions
    source_expression_list = [create_source_term(xloc, yloc, 1, sigma, V, 1) for
                              xloc, yloc in source_function_locations]


   # Assemble the linear forms corresponding to the elementary source terms and
    # construct the second half of the coefficient matrix
    # Define the bilinear form and assemble it
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx + u * v * ds
    A = assemble(a)
    b = [assemble(-f * v * dx) for f in source_expression_list]
    B = scipy.column_stack([v.array() for v in b])
     
    # Construct the full coefficient matrix
    k = scipy.sparse.hstack([matrix_to_coo(A), B])
    C = scipy.sparse.vstack([
        scipy.sparse.hstack([matrix_to_coo(A), B]),
        scipy.sparse.coo_matrix(([1.0]*n**2, ([0]*n**2, [V.dim() + i for i in range(n**2)])), 
                                shape=(1, V.dim() + n**2))
    ])
 
    # Transfer the optimization constraints to CasADi 
    C = casadi.DM(C)
    lb = casadi.DM.zeros(C.shape[0], 1)
    ub = casadi.DM.zeros(C.shape[0], 1)
 
    lb[-1] = 0.0
    ub[-1] = 5.0
 
    lbx = numpy.array([-numpy.inf]*V.dim() + [0.0]*(n**2))
    ubx = numpy.array([ numpy.inf]*V.dim() + [1.0]*(n**2))
 
    #  Define the objective functional and assemble it
    u = Function(V)
    J = inner(reference_solution - u, reference_solution - u) * dx
    dJdu = derivative(J, u)
    dJdu2 = derivative(dJdu, u)
 
    H = casadi.DM(matrix_to_coo(assemble(dJdu2), shape=(V.dim() + n**2, V.dim() + n**2)))
    g = casadi.DM(numpy.concatenate((assemble(dJdu).array(), [0.0]*(n**2))))
    c = assemble(J)
 
    # Build the CasADi NLP
    x = casadi.MX.sym('X', V.dim() + n**2)
    f = 0.5*casadi.dot(x,casadi.mtimes(H, x)) + casadi.dot(g, x) + c
    g = casadi.mtimes(C, x)

    nlp = {'x': x, 'f': f, 'g': g}

    # Build NLP solver
    solver = casadi.nlpsol('solver', 'ipopt', nlp, {'verbose': True})

    start = time.time()
    result = solve_bnb(solver, {'lbx': lbx, 'ubx': ubx, 'lbg': lb, 'ubg': ub}, 
                       range(V.dim(), V.dim() + n**2))
    optimal_w = [x[0] for x in numpy.array(result['x'][-n**2:])]
    runtime = time.time() - start
    output_dict = {
                    'branch_and_bound': {
                        'binaries': numpy.array(optimal_w),
                        'runtime': runtime
                    }
                  }

    return output_dict


