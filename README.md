### Distributed Primal Outer Approximation Algorithm (DiPOA)

This repository holds the code of the paper titled **Distributed Primal Outer Approximation Algorithm for
Sparse Convex Programming with Separable Structures** which introduces a distributed sparse convex optimization
algorithm, named,  **Distributed Primal
Outer Approximation Algorithm ( ```DiPOA``` )**. ```DiPOA``` is implemented within ```dccp``` python package. In the
following, a general instruction to install and run ```dccp``` is provided.

### 1. External Dependencies

1. Python Programming Language 3.7 or higher.
2. Gurobi Optimizer with a license [quick start guide](https://www.gurobi.com/documentation/quickstart.html)
3. Message Passing Interface - MPI
    1. [OpenMPI](https://www.open-mpi.org/)
    2. [MPICH](https://www.mpich.org/)
    3. [Microsoft MPI (for windows)](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
4. GAMS with a license for solver comparison (optional) [guide](https://www.gams.com/latest/docs/)
5. Git

### 2. Supporting Platforms

1. Ubuntu 21.10 or WSL2 (preferred).
2. OSX
3. Microsoft Windows 10.

### 3. MPI Installation on Ubuntu OS

run the following commands

```commandline
sudo apt update
sudo apt install openmpi-bin
sudo apt install libopenmpi-dev
```

### 4. MPI Installation on Microsoft Windows

please read the instruction [here](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi).

make sure ```mpiexec``` or ```mpirun``` executables are in the PATH environment variable of your preferred OS.

### 5. Prepare Running Environment

### Ubuntu OS

1. Make sure ```Gurobi```, ```MPI```, and ```Python``` are installed properly.

2. Install python3-dev for installed python version (if not already installed)

```commandline
sudo apt install python3-dev
```

3. Install python virtual environment

```commandline
sudo apt install virtualenv
```

4. Clone ```dccp``` repository by executing the following command

```commandline
git clone http://www.github.com/Alirezalm/dccp.git
```

5. Change directory to ```dccp``` and create a python virtual environment

```commandline
cd dccp
python3 -m venv env
```

6. Activate the virtual environment

```commandline
source ./env/bin/activate
```

7. Install python dependencies

```commandline
pip install -r requirements.txt
```

Note: for the full list of requirements please see ```requirements.txt``` file

#### Microsoft Windows OS

1. Make sure ```Gurobi```, ```MPI```, and ```Python``` are installed properly.

2. Clone ```dccp``` repository by executing the following command

```shell
git clone http://www.github.com/Alirezalm/dccp.git
```

3. Change directory to ```dccp``` and create a python virtual environment

```shell
cd dccp
python -m venv env
```

4. Activate the virtual environment

```shell
.\env\Scripts\activate
```

5. Install python dependencies

```shell
pip install -r requirements.txt
```

### 6. General Usage

```dccp``` package provides ```dipoa-cli.py```, a simple command-line interface (CLI), to
_generate_ and _solve_ random Distributed Sparse Logistic Regression (DSLR) and Distributed Sparse
Quadratically Constrained Programming (DSQCP) problems. ```dipoa-cli.py``` consists of two sub-commands,
namely, ```dslr``` and ```dsqcp``` that are responsible for solving DSLR and DSQCP problems, respectively. Finally,
after the problem is solved successfully, a ```solution.json``` file will be generated.

#### DSLR

To generate and solve a DSLR problem, use the following command template

```commandline 
mpiexec -np <number-of-nodes> python dipoa-cli.py dslr <m> <n> <nz> --soc
```

where

1. ```m``` is the number of rows per node.
2. ```n``` is the number of decision variables (e.g. columns in DSLR).
3. ```nz``` is the number of non-zero elements.
4. ```--soc``` is a flag to indicate whether a second order cut generation is used.

##### Example

The following command generates and solves a DSLR problem with total ```2000``` rows per node, ```50``` variables with
```5``` non-zero elements, and ```4``` computational nodes (for this scenario, the total number of nodes is ```8000```.)
As for the option ```--soc``` are used.

```commandline 
mpiexec -np 4 python dipoa-cli.py dslr 2000 50 5 --soc
```

#### DSQCP

Command template

```commandline 
mpiexec -np <number-of-nodes> python dipoa-cli.py dsqcp <n> <nz> --soc
```

where

1. ```n``` is the number of decision variables (e.g. columns in DSLR).
2. ```nz``` is the number of non-zero elements.
3. ```--soc``` is a flag to indicate whether a second order cut generation is used.

##### Example

The following command generates and solves a DSQCP problem with ```50``` variables with
```5``` non-zero elements, and ```4``` computational nodes.
As for the option, ```--soc``` are used.

```commandline 
mpiexec -np 4 python dipoa-cli.py dsqcp 50 5  --soc
```

### 7. Comparison with GAMS MINLP Solvers

1. make sure ```gams``` executable is on OS PATH.
2. GAMS MINLP solvers are compared with ```DiPOA``` for DSLR problem instances.
3. To use gams MINLP solvers use the following command template

```commandline
python gams-run.py <solver-name> <m> <n> <nz> <np>
```

where ```np``` is the number of nodes.

#### Example:

The following command runs ```bonmin``` for the DSLR problem with `1000` rows, `50` variables, `5` nonzero
elements and, `4` nodes.

```commandline
python gams-run.py bonmin 1000 50 5 4

```
The MINLP solver that are used in the paper are
1. bonmin
2. dicopt
3. knitro
4. dicopt
5. shot

### 8. Datasets Used in the Paper

The datasets used in the paper for both DSLR and DSQCP problems can are generated by executing
the ```generate_datasets.py``` script as the following command shows.

```commandline
python generate_datasets.py
```

```generate_datasets.py``` creates ```data``` directory in the same directory that
```generate_datasets.py``` exists.
The following directory tree, shows the structure of the ```data``` directory.

```commandline
│── dslr
│   ├── sc_1
│   │   ├── data_10000_20.csv
│   │   ├── data_14000_20.csv
│   │   ├── data_18000_20.csv
│   │   ├── data_2000_20.csv
│   │   ├── data_22000_20.csv
│   │   ├── data_26000_20.csv
│   │   ├── data_30000_20.csv
│   │   ├── data_34000_20.csv
│   │   ├── data_38000_20.csv
│   │   ├── data_42000_20.csv
│   │   ├── data_46000_20.csv
│   │   ├── data_50000_20.csv
│   │   └── data_6000_20.csv
│   ├── sc_2
│   │   ├── data_50000_100.csv
│   │   ├── data_50000_120.csv
│   │   ├── data_50000_140.csv
│   │   ├── data_50000_160.csv
│   │   ├── data_50000_180.csv
│   │   ├── data_50000_200.csv
│   │   ├── data_50000_40.csv
│   │   ├── data_50000_60.csv
│   │   └── data_50000_80.csv
│   ├── sc_3
│   │   └── data_10000_20.csv
│   ├── sc_4
│   │   └── data_100000_200.csv
│   └── sc_5
│       └── data_300000_300.csv
└── dsqcp
    ├── sc_1
    │   ├── ch.csv
    │   ├── Ph.csv
    │   ├── q.csv
    │   └── Q.csv
    └── sc_2
        ├── ch.csv
        ├── Ph.csv
        ├── q.csv
        └── Q.csv

```

The ```dslr``` directory holds the datasets used in DSLR problem instances and consists of five subdirectories. Each
directory holds the corresponding dataset for a specific scenario. Similarly, the ```dsqcp``` directory keeps the
datasets of the scenarios used in DSQCP problem. All datasets are also generated in CSV format.

