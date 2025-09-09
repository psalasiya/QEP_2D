from ngsolve import *
import numpy as np

def QEP_solver_2D(a1, a2, mesh, rh, mu, omega, ky, Z, num_itr, feorder):
    """
    Solves the 2D quadratic eigenvalue problem (QEP) for a given ky.

    This function formulates the problem arising from the Floquet-Bloch periodic
    boundary conditions on the Helmholtz equation. The wave equation is of the form:
    ∇ ⋅ (μ ∇u) + ω²ρu = 0

    Applying Floquet-Bloch conditions u(r) = u_p(r) * exp(-j(kx*x + ky*y)) leads to
    a QEP of the form (A*kx² + B*kx + C)u_p = 0. This is linearized into a
    standard linear eigenvalue problem A*x = λ*B*x and solved for the
    eigenvalues (kx) and eigenfunctions (u_p).

    Args:
        a1, a2 (np.array): Lattice vectors.
        mesh (ngsolve.Mesh): The computational mesh of the unit cell.
        rh (ngsolve.CoefficientFunction): Density distribution.
        mu (ngsolve.CoefficientFunction): Shear modulus distribution.
        omega (float): Angular frequency.
        ky (float): Wave vector component in the y-direction.
        Z (int): Not currently used, placeholder.
        num_itr (int): Number of iterations for the Arnoldi solver.
        feorder (int): Order of the finite element space.

    Returns:
        tuple: A tuple containing:
            - ev_srt (list): Sorted list of complex eigenvalues (kx).
            - ef_srt_norm_li (list): List of corresponding normalized eigenfunctions.
    """
    # Define the finite element spaces. We use a mixed space for linearization.
    # A is the space for the periodic field u_p.
    # B is the space for the auxiliary variable s = kx * u_p.
    A = Periodic(H1(mesh, order = feorder, complex=True))
    B = Periodic(H1(mesh, order = feorder, complex=True))
    fes = FESpace([A, B])

    # Define trial and test functions for the mixed formulation.
    u, s = fes.TrialFunction()
    v, h = fes.TestFunction()
    
    # Contravariant vectors
    b1, b2 = contravariant(a1, a2)
    
    metric_contravariant(b1, b2)
    # Compute the transformation matrix for NGSolve's partial derivatives.
    grad_cart_to_lattice_coeff(a1, a2)

    # Extract components of the contravariant metric tensor G^ij.
    # Compute the contravariant metric tensor for gradient calculations in skew coordinates.
    Gcon = metric_contravariant(b1, b2)
    G11 = Gcon[0, 0]
    G12 = Gcon[0, 1]
    G21 = Gcon[1, 0]
    G22 = Gcon[1, 1]

    # Gradient matrix
    D = grad_cart_to_lattice_coeff(a1, a2)

    # --- Linearization of the QEP ---
    # The QEP (A*kx² + B*kx + C)u = 0 is converted to a linear system:
    #  s = kx*u.
    # 'a' represents the matrix on the left-hand side.
    # 'b' represents the matrix on the right-hand side.

    # Bilinear form for the left-hand side matrix of the linear system.
    a = BilinearForm(fes)
    a += omega ** 2 * rh * u * v * dx
    a += -mu * G11 * partial(u, "x1", D) * partial(v, "x1", D) * dx
    a += -mu * G12 * partial(u, "x1", D) * partial(v, "x2", D) * dx
    a += -mu * G21 * partial(u, "x2", D) * partial(v, "x1", D) * dx
    a += -mu * G22 * partial(u, "x2", D) * partial(v, "x2", D) * dx
    a += -G22 * ky**2 * mu * u * v  * dx
    a += G21 * 1j * mu * (partial(u, "x1", D) * v - u * partial(v, "x1", D)) * ky  * dx
    a += G22 * 1j * mu * (partial(u, "x2", D) * v - u * partial(v, "x2", D)) * ky  * dx
    a += -G12 * 1j * mu * s * partial(v, "x2", D)  * dx
    a += -G11 * 1j * mu * s * partial(v, "x1", D) * dx
    a += Z * s * h * dx

    # Bilinear form for the right-hand side matrix of the linear system.
    b = BilinearForm(fes)
    b += G11 * mu * s * v * dx
    b += 2*G12 * ky * mu * u * v * dx
    b += -G12 * 1j * mu * partial(u, "x2", D) * v * dx
    b += -G11 * 1j * mu * partial(u, "x1", D) * v * dx
    b += Z * u * h * dx

    # Assemble the matrices.
    a.Assemble()
    b.Assemble()

    gf = GridFunction(fes, multidim = num_itr)  # array of eigenfunctions (gf is not normalized)
    vlam = np.zeros((num_itr), dtype=complex)
    
    # Solve the generalized eigenvalue problem a.mat * x = lambda * b.mat * x
    vlam = ArnoldiSolver(a.mat, b.mat, fes.FreeDofs(), list(gf.vecs), shift=-2)
    
    ev_unsrt = vlam
    
    eig_fun_u = GridFunction(A, multidim=num_itr)
    eig_fun_s = GridFunction(B, multidim=num_itr)

    gfu = gf.components[0]
    gfs = gf.components[1]
        
    for i in range(0, num_itr):
        eig_fun_u.vecs[i].data = gfu.vecs[i].data
        eig_fun_s.vecs[i].data = gfs.vecs[i].data
    
    ################################ sorting ##################################
    ev_srt = np.sort(ev_unsrt)
    ind    = np.argsort(ev_unsrt)
    ef_srt = GridFunction(A, multidim=len(ev_srt))
    for i in range(len(ev_srt)):
        ef_srt.vecs[i].data = eig_fun_u.vecs[ind[i]].data
    
    ############################# (ortho)normalising #########################
    ef_srt_li = gf_li(ef_srt, mesh, feorder,'c')
    ef_srt_norm_li = normalise(ef_srt_li, mesh, feorder)

    return ev_srt, ef_srt_norm_li

def gf_li(ef_gf, mesh, feorder, flag = 'r'):
    """
    Converts a multi-dimensional GridFunction containing eigenfunctions into a
    list of standard GridFunctions.

    Args:
        ef_gf (ngsolve.GridFunction): A multidim GridFunction where each
                                      component is an eigenvector.
        mesh (ngsolve.Mesh): The mesh object.
        feorder (int): The finite element order.
        flag (str): If 'r' (default), returns the real part of the eigenfunctions.
                    If 'c', returns the complex eigenfunctions.

    Returns:
        list: A list of single-component ngsolve.GridFunction objects.
    """
    if flag == 'c':
        fes = Periodic(H1(mesh, order = feorder, complex=True))
        num = len(ef_gf.vecs)
        temp = GridFunction(fes)
        ef_li = [GridFunction(fes) for i in range(num)]
   
        for i in range(num):
            temp.vec.data = ef_gf.vecs[i].data
            ef_li[i].Set(temp)
    else:
        fes = Periodic(H1(mesh, order = feorder, complex=True))   
        fer = Periodic(H1(mesh, order = feorder, complex=False))
        num = len(ef_gf.vecs)
        temp = GridFunction(fes)
        ef_li = [GridFunction(fer) for i in range(num)]
   
        for i in range(num):
            temp.vec.data = ef_gf.vecs[i].data
            ef_li[i].Set(temp.real)
  
    return ef_li

def normalise(ef, mesh, feorder):
    """
    Normalizes a list of eigenfunctions.

    Each eigenfunction is normalized such that the integral of its squared
    magnitude over the domain is equal to 1.

    Args:
        ef (list): A list of ngsolve.GridFunction objects (eigenfunctions).
        mesh (ngsolve.Mesh): The mesh object.
        feorder (int): The finite element order.

    Returns:
        list: A list of normalized ngsolve.GridFunction objects.
    """
    fes = Periodic(H1(mesh, order = feorder, complex=True))
    ef_n = [GridFunction(fes) for i in range(len(ef))]
    
    for i in range(len(ef)):
        norm = sqrt(Integrate(ef[i] * Conj(ef[i]), mesh))
        ef_n[i].Set(ef[i] / norm)
    
    return ef_n

def partial(fun, coord, D):
    ''' 
    Partials w.r.t. skew-coordinates.

    Args:
        fun (GridFunction or trial or test function)
        coord ("x1" or "x2") : skew coordinate with respect to which the gradient is to be computed
        D(matrix) : coefficients of gradient conversion
    '''
    if coord == "x1":
        dfun = grad(fun) * (D[0, 0], D[0, 1])

    elif coord == "x2":
        dfun = grad(fun) * (D[1, 0], D[1, 1])

    return dfun

def contravariant(a1, a2):
     """
     Computes the contravariant vectors to a_1, a_2 s.t. a_i . b^j = delta_{ij}
     """
     J = np.column_stack((a1, a2))

     # Reciprocal lattice vectors are columns of (J^-1)^T
     B = np.linalg.inv(J).T
     b1 = B[:, 0]
     b2 = B[:, 1]

     return b1, b2

def metric_covariant(a1, a2):
        """
        Computes the covariant metric tensor G_ij = a_i . a_j.
        This tensor is used for operations involving covariant vector components.
        """
        Gcov = np.array([
            [np.dot(a1, a1), np.dot(a1, a2)],
            [np.dot(a2, a1), np.dot(a2, a2)]
        ])

        return Gcov

def metric_contravariant(b1, b2):
        """
        Computes the contravariant metric tensor G^ij = b^i . b^j.
        This tensor is essential for calculating vector magnitudes and dot products
        from contravariant components, such as those of the gradient.
        """
        Gcon = np.array([
            [np.dot(b1, b1), np.dot(b1, b2)],
            [np.dot(b2, b1), np.dot(b2, b2)]
        ])

        return Gcon

def grad_cart_to_lattice_coeff(a1, a2):
        """
        Computes the transformation matrix 'D' for converting the components
        of a Cartesian gradient (∂f/∂z1, ∂f/∂z2) to lattice gradient components
        (∂f/∂x1, ∂f/∂x2) used in NGSolve's partial derivatives.
        
        NGSolve's `grad(u)` is in Cartesian coordinates. `partial(u, "x1", D)`
        transforms it.
        """

        return np.column_stack((a1, a2)).T

def lattice_to_cart_coeff(a1, a2):
        """
        Computes the transformation matrix 'C' for converting lattice
        position coordinates (x, y) to Cartesian coordinates (u1, u2).
        
        [u1, u2]^T = C @ [x, y]^T
        """

        return np.column_stack((a1, a2))