import json
from pprint import pprint

import click
from mpi4py import MPI

from problem.problem import Problem

DSLR = "dslr"
DSQCQP = "dsqcqp"


@click.group()
def cli():
    pass


@click.command()
@click.argument('data', nargs = -1, type = int)
@click.option('--sfp', is_flag = True, help = "perform specialized feasibility pump")
@click.option('--soc', is_flag = True, help = "perform event triggerd second order cuts")
def dslr(data, sfp: str, soc: str):
    """Generate and solve random DSLR"""
    if len(data) < 3:
        print("insufficient problem arguments. running default scenario ...")
        pdata = {"name": DSLR,
                 "nVars": 10,
                 "nSamples": 1000,
                 "nZeros": 2,
                 "soc": soc, "sfp": sfp}
    else:
        pdata = {"name": DSLR,
                 "nVars": data[1],
                 "nSamples": data[0],
                 "nZeros": data[2],
                 "soc": soc, "sfp": sfp}
    solution = run(pdata)
    return solution


@click.command()
@click.argument('data', nargs = -1, type = int)
@click.option('--sfp', is_flag = True, help = "perform specialized feasibility pump")
@click.option('--soc', is_flag = True, help = "perform event triggerd second order cuts")
def dsqcp(data, sfp: str, soc: str):
    """ Generate and solve random DSQCP """
    if len(data) < 2:
        print("insufficient problem arguments. running default scenario ...")
        pdata = {"name": DSQCQP,
                 "nVars": 10,
                 "nZeros": 2,
                 "soc": soc, "sfp": sfp}
    else:
        pdata = {"name": DSQCQP,
                 "nVars": data[0],
                 "nZeros": data[1],
                 "soc": soc, "sfp": sfp}
    solution = run(pdata)
    return solution


def run(data):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    data['nNodes'] = comm.Get_size()
    problem = Problem(problem_data = data)
    solution_data = problem.solve(comm, MPI)

    if rank == 0:
        with open('solution.json', 'w') as json_ans:
            json.dump(solution_data, json_ans)
        return solution_data


cli.add_command(dslr)
cli.add_command(dsqcp)

if __name__ == '__main__':
    cli()
