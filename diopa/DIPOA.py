"""
Main loop of the DIPOA Algorithm
"""
import sys
from time import time

from numpy import zeros
from numpy.linalg import eig

from diopa.cut_store_gen import CutStoreGen
from diopa.heuristics import sfp
from masters.master_problem import solve_master
from rhadmm.rhadmm import rhadmm


def dipoa(problem_instance, comm, mpi_class):

    rank = comm.Get_rank()
    size = comm.Get_size()
    max_iter = 200
    n = problem_instance.nVars
    binvar = zeros((problem_instance.nVars, 1))  # initial binary
    if problem_instance.sfp:
        binvar = sfp(problem_instance, rank, comm, mpi_class)

    cut_manager = CutStoreGen()
    rcv_x = None
    rcv_gx = None
    rcv_eig = None
    upper_bound = 1e8
    lower_bound = -upper_bound
    eps = 0.015
    data_memory = {
        'x': None,
        'lb': [],
        'ub': [],
        'iter': []
    }
    x = zeros((n, 1))
    min_eig = 0
    problem_instance.sfp = False
    min_const_eig = 0

    if (rank == 0) & problem_instance.soc & (problem_instance.problem_instance.constr is not None):
        min_const_eig = min(eig(problem_instance.problem_instance.compute_const_hess_at(index = 0))[0])

    if problem_instance.name == 'dsqcqp':
        min_eig = min(eig(problem_instance.problem_instance.compute_hess_at())[0])

    if rank == 0:
        _print_header()
    start = time()
    for k in range(max_iter):

        x, fx, gx = rhadmm(problem_instance, bin_var = binvar, comm = comm, mpi_class = mpi_class)

        if problem_instance.soc & (problem_instance.name == 'dslr'):
            min_eig = min(eig(problem_instance.problem_instance.compute_hess_at(x))[0])

        ub = comm.reduce(fx, op = mpi_class.SUM, root = 0)

        if rank == 0:
            upper_bound = min(ub, upper_bound)
            rcv_x = zeros((size, n))
            rcv_gx = zeros((size, n))

        rcv_fx = comm.gather(fx, root = 0)
        if problem_instance.soc:
            rcv_eig = comm.gather(min_eig, root = 0)

        comm.Gather([x, mpi_class.DOUBLE], rcv_x, root = 0)
        comm.Gather([gx, mpi_class.DOUBLE], rcv_gx, root = 0)

        if rank == 0:
            for node in range(size):
                if problem_instance.soc:
                    cut_manager.store_cut(k, node, rcv_x[node, :].reshape(n, 1), rcv_fx[node],
                                          rcv_gx[node, :].reshape(n, 1), rcv_eig[node])
                else:
                    cut_manager.store_cut(k, node, rcv_x[node, :].reshape(n, 1), rcv_fx[node],
                                          rcv_gx[node, :].reshape(n, 1))
            if problem_instance.problem_instance.constr is not None:
                gx = problem_instance.problem_instance.compute_const_at(0, x)
                ggx = problem_instance.problem_instance.compute_const_grad_at(0, x)
                if problem_instance.soc:
                    cut_manager.store_const_cut(x, gx, ggx, min_const_eig)
                else:
                    cut_manager.store_const_cut(x, gx, ggx)

            lower_bound, binvar, x = solve_master(problem_instance, cut_manager)
            data_memory['ub'].append(upper_bound)
            data_memory['lb'].append(lower_bound)
            data_memory['iter'].append(k)

        rel_gap = comm.bcast((upper_bound - lower_bound) / abs(upper_bound + 1e-8), root = 0)
        current_time = time() - start
        if rank == 0:
            print(
                f"k: {k} lb: {lower_bound:8.4f}, ub:{upper_bound:8.4f} gap: {abs(rel_gap):8.3f} "
                f"elapsed: {current_time:4.3f}")
        if rel_gap <= eps:
            if rank == 0:
                print("dipoa converged. check solution.json file for the results\n")
            break

    data_memory['x'] = [item[0] for item in x]
    data_memory['obj'] = lower_bound
    data_memory['gap'] = (upper_bound - lower_bound) / abs(upper_bound + 1e-8)

    return data_memory


def _print_header():
    header = \
        f"""
                                   Distributed Primal Outer Approximation (DiPOA) Algorithm    (c) Alireza Olama   

                                   Federal University of Santa Catarina  ---  UFSC
                                   current version: {"0.1.0"}

                                   running platform: {sys.platform}

                                   github: {"https://github.com/alirezalm"}
                                   email: {"alireza.lm69@gmail.com"}
               """
    print(header)
    print(
        f"Python Version: {sys.version}"
    )
    print(
        f"python executable: {sys.executable}"
    )
