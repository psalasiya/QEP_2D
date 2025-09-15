#%%
from solver_QEP_2D import *
from geo_mat_mesh_2D_QEP import *
import multiprocessing as mp
from functools import partial
import numpy as np
import pickle

#%% Function
def QEP(ky, a1, a2, msh, rh, mu, omega):
    """
    Main function to run a single QEP simulation for a given ky value.

    This function encapsulates the entire simulation pipeline:
    1. Sets numerical parameters for the solver and mesh.
    2. Defines and meshes the unit cell geometry.
    3. Solves the QEP to find eigenvalues (kx) and eigenfunctions.
    4. Post-processes the results to select physically relevant modes.
    5. Interpolates the solution onto a regular grid for MATLAB compatibility.

    Args:
        ky (float): The y-component of the wave vector.
        a1 (np.array): The first lattice vector.
        a2 (np.array): The second lattice vector.
        omega (float): The angular frequency.
        num_ev (int): The maximum number of eigenvalues to store.

    Returns:
        tuple: A tuple containing data ready for MATLAB:
               (x, y, phi, G, kappa, ky)
    """

    ############################ QEP solver ####################################
    # Call the main solver function
    ev, ef = QEP_solver_2D(a1, a2, msh, rh, mu, omega, ky, Z=1, num_itr = 200, feorder = 3) # Z=1 is a placeholder, num_itr is number of Arnoldi iterations, feorder is FE element order
    
    return ev, ef, ky, msh, rh, mu

#%% Input parameters for the simulation sweep
omega = 1.4
# Define the range of ky values to sweep over
ky = np.linspace(1.7, 1.8, 100) * np.pi

# Define the lattice vectors for the unit cell
a1 = np.array([1, 0]) # lattice vector 1
a2 = np.array([1, 2]) # lattice vector 2

# Define the unit cell geometry
msh, rh, mu  = build_skew_cell(a1, a2)

#%% Parallelising
if __name__ == "__main__":
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(partial(QEP, a1 = a1, a2 = a2, msh = msh, rh = rh, mu = mu, omega = omega), ky)

    for i in range(len(ky)):
        my_tuple = (results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[i][5])
        file_s = 'Data/ky_'+str(i)
        with open(file_s, "wb") as picklefile:
            pickle.dump(my_tuple, picklefile)
