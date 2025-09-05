#%%
from solver_QEP_2D import *
from geo_mat_mesh_2D_QEP import *
from evs_class import *
import multiprocessing as mp
from functools import partial
from scipy.io import savemat

#%% Function
def QEP(ky, a1, a2, msh, rh, mu, omega, num_ev):
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

    ########################## post-processing ##################################
    # Create an eigenspectrum object to handle filtering
    evs = eigenspectrum(ev, ef)
    # Select modes within the first Brillouin Zone
    evs.pick_first_BZ()

    ##################### lattice-properties ####################################
    # Initialize skew coordinate system handler
    lattice = skew(a1, a2)
    # Compute transformation matrix for coordinate mapping
    lattice.lattice_to_cart_coeff()

    ########################### MATLAB readable form #############################
    # This section interpolates the NGSolve GridFunction solution onto a regular
    # Cartesian grid, and then maps to skew coordinates that is easier to work with in MATLAB.
    inter_x = 101 # Number of interpolation points in x
    inter_y = 101 # Number of interpolation points in y

    x, y = np.meshgrid(np.linspace(0, 1, inter_x), np.linspace(0, 1, inter_y)) # local coordinates
    u = lattice.C[0, 0]*x + lattice.C[0, 1]*y
    v = lattice.C[1, 0]*x + lattice.C[1, 1]*y  # global coordinates

    # Initialize arrays to store the interpolated data
    kappa = np.zeros(len(evs.ev_BZ), dtype=complex)                   # eigenvalue
    phi = np.zeros((inter_y, inter_x, len(evs.ev_BZ)), dtype=complex) # eigenfunction
    G = np.zeros((inter_y, inter_x))                                # shear modulus

    # Loop through the selected modes and evaluate them at each grid point
    for k in range(len(evs.ev_BZ)):
        temp_phi = evs.ef_BZ[k]
        kappa[k] = evs.ev_BZ[k]

        for i in range(inter_y):
            for j in range(inter_x):
                phi[i, j, k] = temp_phi(msh(u[i, j], v[i, j]))
                G[i, j]  = mu(msh(u[i, j], v[i, j]))  
    
    return x, y, phi, G, kappa, ky

#%% Input parameters for the simulation sweep
omega = 1.95
# Define the range of ky values to sweep over
ky = np.linspace(0, 2, 100) * np.pi

# Set the maximum number of eigenvalues/modes to save for each ky
num_ev = 9 # maximum number of eigenvalues

# Define the lattice vectors for the unit cell
a1 = np.array([1, 0]) # lattice vector 1
a2 = np.array([1, 1]) # lattice vector 2

# Define the unit cell geometry
msh, rh, mu  = build_skew_cell(a1, a2)

#%% Parallelising
if __name__ == "__main__":
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(partial(QEP, a1 = a1, a2 = a2, msh = msh, rh = rh, mu = mu, omega = omega, num_ev = num_ev), ky)

    for i in range(len(ky)):
        matfile_s    = 'Matlab_data/ky_'+str(i)+".mat"
        mdict = {'x' : results[i][0], 'y' : results[i][1], 'phi' : results[i][2], 'G':results[i][3], 'kappa' : results[i][4], 'ky' : results[i][5]}
        print("num_ev = " + str(len(results[i][4])), ", ind = " + str(i+1))
        savemat(matfile_s, mdict)

# %% 
# for i in range(len(ky)):
#     results = QEP(ky[i], a1, a2, omega, num_ev)
#     matfile_s    = 'Matlab_data/ky_'+str(i)+".mat"
#     mdict = {'x' : results[i][0], 'y' : results[i][1], 'phi' : results[i][2], 'G':results[i][3], 'kappa' : results[i][4], 'ky' : results[i][5]}
#     print("num_ev = " + str(len(results[i][4])), ", ind = " + str(i+1))
#     savemat(matfile_s, mdict)