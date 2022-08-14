from numpy import eye
from numpy.random import randn, rand, seed
from sklearn import preprocessing


def gen_qcqp(nvars, num_quad_consts):
    seed(0)
    problem_data = {
        'obj': {
            'hessian_mat': None,
            'grad_vec': None,
            'const': randn()
        },

        'constr': None

    }

    if num_quad_consts != 0:
        problem_data['constr'] = []

    for i in range(num_quad_consts + 1):
        _hess = preprocessing.normalize(randn(nvars, nvars), norm = 'l2')
        _hess = 0.5 * (_hess.T + _hess)
        diag_mat = (1 + rand()) * eye(nvars)
        hess = _hess.T @ _hess + diag_mat
        grad = rand(nvars, 1)
        if i == 0:
            problem_data['obj']['hessian_mat'] = hess
            problem_data['obj']['grad_vec'] = grad
        else:
            constr = {
                'hessian_mat': hess,
                'grad_vec': grad,
                'const': - rand()
            }
            problem_data['constr'].append(constr)
    return problem_data
