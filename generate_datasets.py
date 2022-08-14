import os.path
import pathlib

import numpy as np
from numpy.random import seed, randn
from sklearn import preprocessing
from problem.random_problems import gen_qcqp


class DataGenerator:
    def __init__(self):
        self.current_path = os.path.dirname(__file__)
        self.parent_folder = os.path.join(self.current_path, "data")
        self.dslr = os.path.join(self.parent_folder, "dslr")
        self.dsqcp = os.path.join(self.parent_folder, "dsqcp")
        self.num_sc_dslr = 5
        self.num_sc_dscqp = 2
        seed(0)
        self.init_dirs()

    def init_dirs(self):
        pathlib.Path(self.parent_folder).mkdir(exist_ok = True)
        pathlib.Path(self.dslr).mkdir(exist_ok = True)
        pathlib.Path(self.dsqcp).mkdir(exist_ok = True)

        for i in range(self.num_sc_dslr):
            pathlib.Path(os.path.join(self.dslr, f"sc_{i + 1}")).mkdir(exist_ok = True)

        for i in range(self.num_sc_dscqp):
            pathlib.Path(os.path.join(self.dsqcp, f"sc_{i + 1}")).mkdir(exist_ok = True)

    def generate(self):
        self.gen_dslr()
        self.gen_dsqcp()

    def gen_dsqcp(self):
        self.gen_dsqcp_sc_1()
        self.gen_dsqcp_sc_2()

    def gen_dsqcp_sc_1(self):
        print("generating data for dsqcp with 100 vars.")
        data = gen_qcqp(100, 1)
        np.savetxt(os.path.join(self.dsqcp, "sc_1", "Q.csv"), data['obj']['hessian_mat'], delimiter = ', ')
        np.savetxt(os.path.join(self.dsqcp, "sc_1", "q.csv"), data['obj']['grad_vec'], delimiter = ', ')
        np.savetxt(os.path.join(self.dsqcp, "sc_1", "Ph.csv"), data['constr'][0]['hessian_mat'], delimiter = ', ')
        np.savetxt(os.path.join(self.dsqcp, "sc_1", "ch.csv"), data['constr'][0]['grad_vec'], delimiter = ', ')

    def gen_dsqcp_sc_2(self):
        print("generating data for dsqcp with 200 vars.")
        data = gen_qcqp(200, 1)
        np.savetxt(os.path.join(self.dsqcp, "sc_2", "Q.csv"), data['obj']['hessian_mat'], delimiter = ', ')
        np.savetxt(os.path.join(self.dsqcp, "sc_2", "q.csv"), data['obj']['grad_vec'], delimiter = ', ')
        np.savetxt(os.path.join(self.dsqcp, "sc_2", "Ph.csv"), data['constr'][0]['hessian_mat'], delimiter = ', ')
        np.savetxt(os.path.join(self.dsqcp, "sc_2", "ch.csv"), data['constr'][0]['grad_vec'], delimiter = ', ')

    def gen_dslr(self):
        self.gen_sc_1()
        self.gen_sc_2()
        self.gen_sc_3()
        self.gen_sc_4()
        self.gen_sc_5()

    def gen_sc_1(self):
        cols = 20
        for row in range(2000, 54000, 4000):
            X = preprocessing.normalize(randn(row, cols), norm = 'l2')
            y = randn(row, 1)
            y[y >= 0.5] = 1
            y[y < 0.5] = 0
            dataset = np.concatenate((X, y), axis = 1)
            np.savetxt(os.path.join(self.dslr, "sc_1", f"data_{row}_{cols}.csv"), dataset, delimiter = ', ')
            print(f"generating data for SC-I: nrows: {row}, ncols: {cols}")

    def gen_sc_2(self):
        rows = 50000
        for cols in range(40, 220, 20):
            X = preprocessing.normalize(randn(rows, cols), norm = 'l2')
            y = randn(rows, 1)
            y[y >= 0.5] = 1
            y[y < 0.5] = 0
            dataset = np.concatenate((X, y), axis = 1)
            np.savetxt(os.path.join(self.dslr, "sc_2", f"data_{rows}_{cols}.csv"), dataset, delimiter = ', ')
            print(f"generating data for SC-II: nrows: {rows}, ncols: {cols}")

    def gen_sc_3(self):
        rows, cols = 10000, 20
        print(f"generating data for SC-III: nrows: {rows}, ncols: {cols}")
        X = preprocessing.normalize(randn(rows, cols), norm = 'l2')
        y = randn(rows, 1)
        y[y >= 0.5] = 1
        y[y < 0.5] = 0
        dataset = np.concatenate((X, y), axis = 1)
        np.savetxt(os.path.join(self.dslr, "sc_3", f"data_{rows}_{cols}.csv"), dataset, delimiter = ', ')

    def gen_sc_4(self):
        rows, cols = 100000, 200
        print(f"generating data for SC-IV: nrows: {rows}, ncols: {cols}")
        X = preprocessing.normalize(randn(rows, cols), norm = 'l2')
        y = randn(rows, 1)
        y[y >= 0.5] = 1
        y[y < 0.5] = 0
        dataset = np.concatenate((X, y), axis = 1)
        np.savetxt(os.path.join(self.dslr, "sc_4", f"data_{rows}_{cols}.csv"), dataset, delimiter = ', ')

    def gen_sc_5(self):
        rows, cols = 300000, 300
        print(f"generating data for SC-V: nrows: {rows}, ncols: {cols}")
        X = preprocessing.normalize(randn(rows, cols), norm = 'l2')
        y = randn(rows, 1)
        y[y >= 0.5] = 1
        y[y < 0.5] = 0
        dataset = np.concatenate((X, y), axis = 1)
        np.savetxt(os.path.join(self.dslr, "sc_5", f"data_{rows}_{cols}.csv"), dataset, delimiter = ', ')


if __name__ == '__main__':
    dg = DataGenerator()
    dg.generate()
