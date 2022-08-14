import gurobipy as gp
from gurobipy import GRB


def gurobi_qcp(problem, y, z, rho):
    model = gp.Model('qcqp')

    n = problem.nVars
    x = model.addMVar(shape = n, lb = -GRB.INFINITY)

    obj = x @ (problem.problem_instance.obj_hess * 0.5) @ x + \
          problem.problem_instance.obj_vec.T @ x + problem.problem_instance.obj_const + \
          y.T @ x - y.T @ z + (rho / 2) * (x @ x - 2 * z.T @ x + z.T @ z)

    model.setObjective(obj, GRB.MINIMIZE)
    const = x @ problem.problem_instance.constr[0]['hessian_mat'] @ x + problem.problem_instance.constr[0][
        'grad_vec'].T @ x + problem.problem_instance.constr[0]['const']
    model.addConstr(const <= 0, name = 'qcp')
    model.setParam('OutputFlag', 0)
    # model.setParam('BarQCPConvTol', 1e-5)
    model.optimize()
    return x.x.reshape(n, 1), model.objval
