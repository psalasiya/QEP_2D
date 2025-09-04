from ngsolve import *
from netgen.occ import *
import numpy as np
from math import floor, ceil

def build_skew_cell(a1, a2, R=0.2, maxh=0.05):
    """
    Build a skew (parallelogram) unit cell using netgen.occ.
    
    Inputs
    ------
    a1, a2 : length-2 iterables
        Lattice vectors a1 = (a1x, a1y), a2 = (a2x, a2y).
    R : float
        Radius of the circular inclusion at each orthogonal cell center (m+0.5, n+0.5).
    maxh : float
        Global mesh size hint.
    curve_order : int
        Spline order for curved geometry in the final mesh.

    Returns
    -------
    geo : OCCGeometry
        OCC geometry with materials & periodic identifications.
    mesh : ngsolve.Mesh
        The generated mesh.
    """
    # a1, a2 are your lattice vectors (same ones used to build the parallelogram)
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)

    # Parallelogram vertices (origin-based)
    p0 = Pnt(0,               0,               0)
    p1 = Pnt(a1[0],           a1[1],           0)
    p2 = Pnt(a1[0]+a2[0],     a1[1]+a2[1],     0)
    p3 = Pnt(a2[0],           a2[1],           0)

    # Edges built explicitly so we can identify opposite pairs (periodic BCs)
    e01 = Segment(p0, p1)   # along +a1
    e12 = Segment(p1, p2)   # along +a2
    e23 = Segment(p2, p3)   # along -a1
    e30 = Segment(p3, p0)   # along -a2

    # Build the outer boundary as a wire/face from the edges (preserves our edge objects)
    wire = Wire([e01, e12, e23, e30])
    outer = Face(wire)

    # Find all circle centers that can intersect the (clipped) cell
    # Bounding box of the parallelogram (expand by R so partially-cut inclusions are included)
    xs = [0, a1[0], a1[0]+a2[0], a2[0]]
    ys = [0, a1[1], a1[1]+a2[1], a2[1]]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    expand = R + 1e-9
    xminE = xmin - expand
    xmaxE = xmax + expand
    yminE = ymin - expand
    ymaxE = ymax + expand

    # Integer index ranges for centers at (m+0.5, n+0.5)
    mmin = floor(xminE - 0.5)
    mmax = ceil (xmaxE - 0.5)
    nmin = floor(yminE - 0.5)
    nmax = ceil (ymaxE - 0.5)

    # Build the union of all circular faces that might intersect the cell
    inc_union = None
    for m in range(mmin, mmax+1):
        for n in range(nmin, nmax+1):
            cx = m + 0.5
            cy = n + 0.5
            cface = Circle(Pnt(cx, cy), R).Face()
            inc_union = cface if inc_union is None else (inc_union + cface)

    # Clip union of inclusions to the cell and split into two materials
    inc_clipped = outer*inc_union # inclusion pieces inside the cell
    inc_clipped.faces.name = "inclusion"
    mat = outer - inc_union
    mat.faces.name = "matrix"

    # Glue pieces to get a single connected OCC shape with shared edges
    shape = Glue([mat, inc_clipped])

    # Periodic edge identification using level-set selection (robust to cuts)
    L, R, T, B = classify_parallelogram_sides(shape, a1, a2)
    L, R, T, B = edge_correspondance(L, R, T, B, a1, a2)
    
    [T[i].Identify(B[i], "tb") for i in range(len(T))]
    [L[i].Identify(R[i], "lr") for i in range(len(L))]
    # T[0].Identify(B[0], "tb")
    # L[0].Identify(R[0], "lr")

    # Final geometry & mesh
    geo  = OCCGeometry(shape, dim=2)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))

    # Assign materials
    mat_ls = [1, 1, 5, 0.1] # [rh_matrix, mu_matrix, rh_inclusion, mu_inclusion]

    rhd = {}
    mud = {}
    rhd['matrix'] = mat_ls[0]
    mud['matrix'] = mat_ls[1]
    rhd['inclusion'] = mat_ls[2]
    mud['inclusion'] = mat_ls[3]
    
    rho = CoefficientFunction([rhd[mat] for mat in mesh.GetMaterials()])
    mu  = CoefficientFunction([mud[mat] for mat in mesh.GetMaterials()])

    return mesh, rho, mu

def unit_normal(v):
    v = np.asarray(v, float)
    n = np.array([v[1], -v[0]], float)   # rotate 90°
    n /= np.linalg.norm(n)
    return n

def edge_mid_xy(edge):
    # Your build returns ((xmin,ymin,zmin),(xmax,ymax,zmax))
    (x0, y0, _), (x1, y1, _) = edge.bounding_box
    return 0.5*(x0+x1), 0.5*(y0+y1)

def classify_parallelogram_sides(shape, a1, a2, tol_factor=1e-8):
    """
    Classify boundary sub-edges into (left, right, bottom, top) for a cell spanned by a1, a2.
    Works even if inclusions split sides into many pieces.
    """
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)

    # Normals for grouping opposite sides:
    #   n1 ⟂ a2  -> distinguishes planes normal to a1 (left/right)
    #   n2 ⟂ a1  -> distinguishes planes normal to a2 (bottom/top)
    n1 = np.array([a1[1], -a1[0]])  # left/right classifier
    n2 = np.array([a2[1], -a2[0]])  # bottom/top classifier
    def dot(u, p): return u[0]*p[0] + u[1]*p[1]

    # We can iterate over all edges of the final 'shape'; edges not on the outermost loop
    # won’t sit on the min/max projection planes and so won’t be picked.
    edges = list(shape.edges)
    mids  = [edge_mid_xy(e) for e in edges]
    vals1 = [abs(dot(n1, m)) for m in mids]
    vals2 = [abs(dot(n2, m)) for m in mids]

    B, T = min(vals1), max(vals1)
    L, R = min(vals2), max(vals2)

    eps = tol_factor

    top     = [e for e,v in zip(edges, vals1) if abs(v - T) < eps]
    bottom  = [e for e,v in zip(edges, vals1) if abs(v - B) < eps]
    left    = [e for e,v in zip(edges, vals2) if abs(v - L) < eps]
    right   = [e for e,v in zip(edges, vals2) if abs(v - R) < eps]

    return left, right, top, bottom

def edge_correspondance(left, right, top, bottom, a1, a2, tol_factor=1e-8):
    
    eps = tol_factor

    T = top
    B = []
    for t in top:
        for b in bottom:
            if abs(edge_mid_xy(t)[1] - edge_mid_xy(b)[1] - a2[1]) < eps:
                B.append(b)
    L = left
    R = []
    for l in left:
        for r in right:
            if abs(edge_mid_xy(r)[0] - edge_mid_xy(l)[0] - a1[0]) < eps:
                R.append(r)

    return L, R, T, B