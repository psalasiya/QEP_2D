# QEP_2D
This package containes python codes to calculate the quadratic eigenspactrum of the 2D unit-cell. It is capable of generating skew unit-cells that is a cut-out from an infinite orthogonal lattice made of square unit-cell with a circular inclusion. The skew unit-cell is defined by a non-orthogonal basis a1, a2. Before running the code, create a virtual python environment with a Python based FE package, NGSolve as follows:  
1.  Download Python 3.13.7 from https://www.python.org and follow the instructions,  
2.  In the terminal, create a directory `mkdir` and create a virtual environment with the command `python3.13 -m venv name-of-the-environment`,
3.  Activate the directory with `source name-of-the-environment\bin\activate`,
4.  Install ngsolve with `pip install ngsolve`,
5.  Install other dependencies using `pip install matplotlib numpy scipy` and
6.  Check if everything is installed properly by running `netgen`, a ngsolve GUI should open.  

Once ngsolve is installed succesfully, run the following code to calculate the eigenspectrum of a skew-cell cut out from orthogonal lattice of square with circular inclusion unit-cell.  
1.  Run QEP_2D.py. The file is pre-populated with omega, ky, a1, a2 and mesh module. The eigenspectrum will be saved in Data directory.
