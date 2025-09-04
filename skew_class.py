import numpy as np
from numpy.typing import NDArray

class skew(object):
    """
    A class to handle coordinate transformations for 2D skew lattices.

    This class computes metric tensors, reciprocal lattice vectors, and
    transformation matrices required to express vector components (like gradients)
    and positions in both Cartesian and lattice coordinate systems.

    Attributes:
        a1 (NDArray[np.float64]): The first lattice vector.
        a2 (NDArray[np.float64]): The second lattice vector.
        b1 (NDArray[np.float64]): The first reciprocal lattice vector.
        b2 (NDArray[np.float64]): The second reciprocal lattice vector.
        Gcov (NDArray[np.float64]): The covariant metric tensor.
        Gcon (NDArray[np.float64]): The contravariant metric tensor.
        D (NDArray[np.float64]): Transformation matrix for Cartesian gradients.
        C (NDArray[np.float64]): Transformation matrix for Cartesian coordinates.
    """
    def __init__(self, a1: NDArray[np.float64], a2: NDArray[np.float64]):
        """
        Initializes the skew lattice object and computes reciprocal vectors.

        Args:
            a1 (NDArray[np.float64]): The first lattice vector, e.g., np.array([1, 0]).
            a2 (NDArray[np.float64]): The second lattice vector, e.g., np.array([0.5, 1]).
        """
        if a1.shape != (2,) or a2.shape != (2,):
            raise ValueError("Lattice vectors a1 and a2 must be 2D numpy arrays.")
            
        self.a1 = a1
        self.a2 = a2
        
        # Transformation matrix from lattice to Cartesian coordinates
        J = np.column_stack((a1, a2))
        
        # Reciprocal lattice vectors are columns of (J^-1)^T
        B = np.linalg.inv(J).T
        self.b1 = B[:, 0]
        self.b2 = B[:, 1]
        
        # Initialize attributes to None, they will be computed on demand
        self.Gcov = None
        self.Gcon = None
        self.D = None
        self.C = None


    def metric_covariant(self) -> None:
        """
        Computes the covariant metric tensor G_ij = a_i . a_j.
        This tensor is used for operations involving covariant vector components.
        """
        self.Gcov = np.array([
            [np.dot(self.a1, self.a1), np.dot(self.a1, self.a2)],
            [np.dot(self.a2, self.a1), np.dot(self.a2, self.a2)]
        ])

    def metric_contravariant(self) -> None:
        """
        Computes the contravariant metric tensor G^ij = b^i . b^j.
        This tensor is essential for calculating vector magnitudes and dot products
        from contravariant components, such as those of the gradient.
        """
        self.Gcon = np.array([
            [np.dot(self.b1, self.b1), np.dot(self.b1, self.b2)],
            [np.dot(self.b2, self.b1), np.dot(self.b2, self.b2)]
        ])

    def grad_cart_to_lattice_coeff(self) -> None:
        """
        Computes the transformation matrix 'D' for converting the components
        of a Cartesian gradient (∂f/∂z1, ∂f/∂z2) to lattice gradient components
        (∂f/∂x1, ∂f/∂x2) used in NGSolve's partial derivatives.
        
        NGSolve's `grad(u)` is in Cartesian coordinates. `partial(u, "x1", D)`
        transforms it.
        """
        J = np.column_stack((self.a1, self.a2))
        self.D = J.T

    def lattice_to_cart_coeff(self) -> None:
        """
        Computes the transformation matrix 'C' for converting lattice
        position coordinates (x, y) to Cartesian coordinates (u1, u2).
        
        [u1, u2]^T = C @ [x, y]^T
        """
        J = np.column_stack((self.a1, self.a2))
        self.C = J

