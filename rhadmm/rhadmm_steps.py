from numpy import zeros, inf
from numpy.random import randn
from scipy.optimize import minimize, NonlinearConstraint

from rhadmm.gurobi_qcp import gurobi_qcp


def update_primary_vars(rhadmm_obj, rhadmm_grad, n_vars, constrs = None):
    initial_condition = zeros((n_vars,))
    method = 'cg'
    quad_constr = []
    options = {}
    if constrs is not None:
        method = 'trust-constr'
        quad_constr = []
        for constr in constrs:
            quad_constr.append(NonlinearConstraint(
                lambda x: x.T @ constr['hessian_mat'] @ x + constr['grad_vec'].T @ x + constr['const'],
                -inf,
                0,
                jac = lambda x: (2 * constr['hessian_mat'] @ x.reshape(n_vars, 1) + constr['grad_vec']).reshape(
                    n_vars, ),
                hess = lambda x, y: 2 * constr['hessian_mat']
            ))
        options = {
            'disp': False
        }

        slack_solver = minimize(quad_constr[0].fun, x0 = randn(n_vars, ))
        if slack_solver.success & (slack_solver.fun <= 0):
            initial_condition = slack_solver.x
        else:
            raise ValueError("INNER PROBLEM FAILED: PROBLEM INFEASIBLE")

        x = gurobi_qcp(rhadmm_obj, quad_constr[0].fun, init = initial_condition)
    solver = minimize(rhadmm_obj, jac = rhadmm_grad, x0 = initial_condition, method = method, options = options,
                      constraints = quad_constr)
   # print(solver)
    #if solver.success:

    return solver.x.reshape(n_vars, 1)
   # else:
        #raise ValueError(f"Inner Solver inside RHADMM failed. {method}")
