import numpy as np
import itertools

class ConewisePolynomial:
    """
    Bad implementation of conewise polynomial. Doesn't keep track of where the pieces
    are actually defined.
    """

    # TODO: Replace list with dict -> easier to compare domains

    def __init__(self, fan, data):
        """
        data should be a list of rational polynomials with as many entries as the fan has maximal cones
        """
        self.fan = fan
        self.data = data

    def __mul__(self, other):
        if self.fan != other.fan:
            raise ValueError("Polynomials are defined on different fans.")
        result = []
        for i in self.fan.cones(self.fan.dim()):
            idx = self.fan.cones(self.fan.dim()).index(i)
            result.append(self.data[idx] * other.data[idx])
        return ConewisePolynomial(fan=self.fan, data=result)

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def ones(fan):
        """
        Accepts a fan.
        Returns the conewise polynomial on the fan which is constantly one. 
        """

        data = []
        for c in range(fan.ngenerating_cones()):
            data.append(1)

        return ConewisePolynomial(fan=fan, data=data)

class EvaluationMap:
    """
    Implements the evaluation map on the top degree cohomology of a complete simplicial fan.
    """

    def __init__(self, Theta, Sigma):
        """
        Arguments:
        Theta: list of the denominators of the summands of the evaluation map
        Sigma: the fan on which the evaluation map is defined
        """
        self.Theta = Theta
        self.Sigma = Sigma

    def __call__(self, f):
        """
        Accepts a ConewisePolynomial and returns its evaluation
        """

        #Check that f is actually defined on Sigma
        if f.fan != self.Sigma:
            raise ValueError("Cannot evaluate a function defined on a different fan.")

        result = 0
        for i, Theta_i in enumerate(self.Theta):
            result += f[i] / Theta_i
        return result

    def __repr__(self):
        s = f"<f> = "
        for i, Theta_i in enumerate(self.Theta):
            s += f"(f[{i}] / {Theta_i}) + "
        s = s[:-3] #truncate last " + "
        return s


def get_characteristic_function_of_ray(rho, Sigma, restrict_to=None):
    """
    Accepts a ray rho of a rational polyhedral fan Sigma
    Returns the corresponding characteristic function, normalized to take the value
        1 on the minimal ray generator of rho.
        The characteristic function is appropriately signed to ensure that it takes positive
        values on the ray (i.e. is a support function in the sense of Fleming-Karu.)
    If restrict_to is not None, but a maximal cone sigma in Sigma, returns the restriction
        to sigma as a globally defined polynomial.
    """

    # check which maximal cones of Sigma contain rho
    supporting_cones = rho.star_generators()
    f = [0]*len(Sigma.cones(Sigma.dim())) # f is locally defined on each maximal cone
    for sigma in supporting_cones:
        reordered_rays = [r for r in sigma.rays() if r is not rho.rays()[0]]
        reordered_rays.append(rho.rays()[0])
        R = PolynomialRing(QQ, "x", Sigma.dim())
        
        # compute projection onto orthogonal complement of the facet of sigma opposite rho
        M = matrix(QQ, reordered_rays)
        M = M.gram_schmidt()[0].T
        v = M[:, -1] # last column holds the orthogonalized version of rho
        
        # define projection onto the subspace spanned by v: \pi(x) = <x, v> * v
        f_sigma = 0
        for i in range(Sigma.dim()):
            delta_i = [0]*Sigma.dim()
            delta_i[i] = 1
            f_sigma += v[i] * R.monomial(*delta_i)
        
        f_sigma = f_sigma[0] # unwrap tuple

        # evaluate f_sigma on minimal ray generator of rho
        scale = f_sigma(*rho.rays()[0])
        # normalize f_sigma such that f_sigma(minimal ray generator) = 1
        f_sigma = 1/scale * f_sigma
        
        i_sigma = Sigma.cones(Sigma.dim()).index(sigma)
        f[i_sigma] = f_sigma

    if restrict_to is not None:
        if restrict_to not in Sigma:
            raise ValueError("argument restrict_to is not a cone in Sigma.")
        if restrict_to.dim() is not Sigma.dim():
            raise ValueError("dimension of argument restrict_to is not maximal.")
        idx = Sigma.cones(Sigma.dim()).index(restrict_to)
        return f[idx]
    return ConewisePolynomial(fan=Sigma, data=f)

def get_characteristic_function_of_cone(sigma, Sigma, make_global=False):
    """
    Accepts a cone sigma of a rational polyhedral fan Sigma
    Returns a characteristic function of sigma
    If make_global is True, returns the restriction of the characteristic function
        to sigma as a globally defined polynomial.
    TODO implement restrict_to like in the ray case
    """
    f = ConewisePolynomial(fan=Sigma, data=[1]*len(Sigma.cones(Sigma.dim())))
    for rho in sigma.faces(1):
        f *= get_characteristic_function_of_ray(rho, Sigma)
    if make_global:
        idx = Sigma.cones(sigma.dim()).index(sigma)
        return f[idx]
    return f
        

def get_basis_functions(P, deg=None, return_order=False):
    """
    Accepts a polytope P.
    Returns a list of basis functions for the cohomology ring of the normal fan of P.
    If deg is not None but an integer, returns only the basis elements of degree deg.
    If return_order is True, return the shelling order as a list of indices.
    """

    # "Shelling order" on the vertices (Idea by Fleming-Karu): Find generic hyperplane in the ambient space
    n_tries = 100
    for i in range(n_tries):
        # generate random linear form on QQ^{P.dim()}
        w = vector([QQ.random_element() for _ in range(P.dim())])
        f = lambda x: x.vector().dot_product(w)
        seen_values = []
        vertices = P.Vrepresentation()
        for v in vertices:
            if f(v) in seen_values:
                continue
            seen_values.append(f(v))
        if len(seen_values) == P.n_vertices():
            # have found a generic hyperplane
            break
        if i == n_tries-1:
            raise Exception("Unable to find a generic hyperplane in relation to the polytope.")
    
    # sort vertices in ascending order based on their value at f 
    shelling_order_idcs = sorted(list(range(P.n_vertices())), key=lambda i: seen_values[i])

    # get ordered normal fan of P
    normal_fan = ordered_normal_fan(P)
    rays = normal_fan.cones(1)

    M = Matrix.zero(normal_fan.nrays(), normal_fan.ngenerating_cones())
    for i, rho in enumerate(normal_fan.cones(1)):
        for idx in rho.star_generator_indices():
            M[i, idx] = 1
    # M[i, j] = 1 means the characteristic function of ray i is supported on the maximal cone j

    basis_functions = []
    for i in range(normal_fan.ngenerating_cones()):
        # For every maximal cone sigma, find minimum degree products of characteristic functions which
        # are supported on the relative interior of sigma and zero on every max cone that has already been seen

        if i == 0:
            basis_functions.append((ConewisePolynomial.ones(normal_fan), 0))
            continue
        
        for k in range(1, normal_fan.dim()+1):
            success = False
            for t in itertools.combinations(range(normal_fan.nrays()), k):
                # compute support of product of characteristic functions of the rays indexed by elements of t
                E = Matrix([1]*normal_fan.ngenerating_cones())
                for idx in t:
                    E = E.elementwise_product(M[idx, :])
                if not any([E[0, j] for j in range(i)]) and E[0, i]:
                    # We have found a minimum degree function which satisfies the required vanishing properties
                    # Multiply the respective characteristic functions
                    f = ConewisePolynomial.ones(normal_fan)
                    for idx in t:
                        f *= get_characteristic_function_of_ray(rays[idx], normal_fan)
                    basis_functions.append((f, k))
                    success = True
                    break
            if success:
                break
    
    # Order basis functions by degree
    basis_functions = sorted(basis_functions, key=lambda x: x[1])
    basis_functions = list(map(lambda x: x[0], basis_functions))

    if return_order:
        return basis_functions, shelling_order_idcs
    return basis_functions

def get_evaluation_map(Sigma):
    """
    Accepts a two-dimensional complete polytopal simplicial fan.
    Returns the evaluation map for the top degree cohomology.
    """

    def e(i):
        x = [0]*Sigma.dim()
        x[i] = 1
        return x

    Theta = [] # Theta[i] = denominator of i-th summand of evaluation function

    for sigma in Sigma.cones(Sigma.dim()):
        f_sigma = 1
        M = Matrix.zero(Sigma.dim(), Sigma.dim())
        for i, r in enumerate(sigma.faces(1)):
            f_i = get_characteristic_function_of_ray(r, Sigma, restrict_to=sigma)
            f_sigma *= f_i
            for j in range(Sigma.dim()):
                M[i, j] = f_i(e(j))

        # norm of f_1 \wedge f_2 \wedge \ldots \wedge \f_n.
        # (only e_1 \wedge \ldots \wedge e_n)-factors survive. Here, n = Sigma.dim()
        norm = abs(1 / M.det()) 
        Theta.append(norm * f_sigma)
    
    return EvaluationMap(Theta, Sigma)

def compute_poincare_pairing(Sigma):
    """
    Accepts a two-dimensional complete polytopal simplicial fan.
    Returns a matrix representing the Poincare pairing on the H^1
    """

    # get a basis of the degree one part of the cohomology ring, constructed
    # from Courant functions
    basis_functions = get_basis_functions(Sigma, deg=1)

    # define evaluation map on H^2
    evaluation_map = get_evaluation_map(Sigma)

    # compute pairing
    size = len(basis_functions)
    M = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            M[i, j] = evaluation_map(basis_functions[i] * basis_functions[j])
    return M

def HirzebruchFan(r):
    """
    Accepts a nonnegative integer r.
    Returns the fan of the Hirzebruch surface with parameter r.
    """

    return Fan2d([(0, 1), (1, 0), (0, -1), (-1, r)])

def FakeHirzebruchFan(r):
    """
    Accepts a nonnegative integer r.
    Returns a fan that Finn falsely believed to belong to the
        Hirzebruch surface with parameter r.
    """

    return Fan2d([(0, 1), (1, 0), (0, -1), (-r, 1)])

def P2():
    """
    Returns the fan for P2.
    """

    return Fan2d([(0, 1), (1, 0), (-1, -1)])

def HirzebruchPolytope(r):
    """
    Accepts a non-negative integer r._[]
    Returns a lattice polytope whose normal fan is the Hirzebruch fan with parameter r.
    """

    return Polyhedron([(0, 0), (0, 1), (1, 0), (1+r, 1)])

def FakeHirzebruchPolytope(r):
    """
    Accepts a non-negative integer r.
    Returns a lattice polytope whose normal fan is the fake Hirzebruch fan with parameter r.
    """

    return Polyhedron([(0, 0), (1, 0), (0, r), (2, r)])

def Simplex(n):
    """
    Accepts a non-negative integer r.
    Returns the n-simplex.
    """

    vertices = [tuple([0]*n)]
    for i in range(n):
        e_i = [0]*n
        e_i[i] = 1
        vertices.append(tuple(e_i))
    return Polyhedron(vertices)

def ordered_normal_fan(P):
    """
    Accepts a lattice polytope P.
    Returns the normal fan of P, constructed in a way such that the bijection
        {vertices of P} <-> {maximal cones of the normal fan of P}
        is index-preserving.
    """

    # instead of invoking P.normal_fan(), we build the normal fan from scratch
    # in a way that guarantees that the maximal cones are ordered in the same
    # way as the vertices of P.

    vertices = list(P.Vrepresentation())
    halfspaces = list(P.Hrepresentation())

    rays = []
    cones = []

    # Create ordered list of rays of the normal fan
    for facet in P.facets():
        vertices_in_facet = list(facet.vertex_generator())
        common_hypersurface = [ieq for ieq in halfspaces if all([ieq in v.incident() for v in vertices_in_facet])][0]
        rays.append(common_hypersurface.vector()[1:])
    
    # Create list of cones
    for v in vertices:
        rays_of_v = []
        for ieq in v.incident():
            ray = ieq.vector()[1:]
            rays_of_v.append(rays.index(ray))
        cones.append(rays_of_v)

    # build the normal fan
    return Fan(rays=rays, cones=cones)

def get_support_function_from_polytope(P):
    """
    Accepts a simplicial lattice polytope P.
    Returns the normal fan of P together with the convex support function of the ample Cartier divisor defined by P.
    """

    if not P.is_simplicial():
        raise ValueError("The polytope P must be simplicial.")

    # get the normal fan
    normal_fan = ordered_normal_fan(P)
    R = PolynomialRing(QQ, "x", normal_fan.dim())

    # construct the support function
    data = []
    for i in range(normal_fan.ngenerating_cones()):
        f_i = 0
        v = vertices[i]
        for j in range(normal_fan.dim()):
            delta_j = [0]*normal_fan.dim()
            delta_j[j] = 1
            x_j = R.monomial(*delta_j)
            f_i += v[j]*x_j
        data.append(f_i)

    phi = ConewisePolynomial(fan=normal_fan, data=data)

    return normal_fan, phi

def compute_lefschetz_determinant(P):
    """
    Accepts a (two-dimensional) simple lattice polytope P.
    Returns the determinant of the Lefschetz operator associated to the natural ample line bundle on the
    toric variety associated to P.
    """

    normal_fan, phi = get_support_function_from_polytope(P)
    ev = get_evaluation_map(normal_fan)

    det = ev(phi*phi)
    return det

def random_lattice_polytope(n_vertices, dim):
    """
    Accepts a nonnegative integer n_vertices and a nonnegative integer dim.
    Returns a random lattice polytope in dimension dim which is the convex hull of n_vertices vertices
        (V-representation need not be minimal)
    """

    # initialize lattice
    M = matrix.identity(dim)
    L = IntegralLattice(M)

    # get n_vertices random elements of the lattics
    vertices = [L.random_element() for i in range(n_vertices)]

    return Polyhedron(vertices)

def 