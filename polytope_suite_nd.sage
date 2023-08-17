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

    def __rmul__(self, other):
        """
        left-action of a globally defined polynomial by multiplication
        """

        result = [other * x for x in self.data]
        return ConewisePolynomial(fan=self.fan, data=result)

    def __sub__(self, other):
        """
        subtraction of conewise polynomials. returns self - other
        """

        result = [self.data[i] - other.data[i] for i in range(len(self.data))]
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
        if restrict_to.dim() != Sigma.dim():
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
        

def get_basis_functions(P, deg=None, return_order=False, deterministic=False):
    """
    Accepts a polytope P.
    Returns a list of basis functions for the cohomology ring of the normal fan of P.
    If deg is not None but an integer, returns only the basis elements of degree deg.
    If return_order is True, return the shelling order as a list of indices.
    If deterministic is True, set the random seed before attempting to find a generic hyperplane.
    """

    # set seed for PRNG to ensure reproducibility
    if deterministic:
        set_random_seed(27)

    # "Shelling order" on the vertices (idea by Fleming-Karu): Find generic hyperplane in the ambient space
    n_tries = 150
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
    # inverse permutation
    inverse_shelling_order = [shelling_order_idcs.index(i) for i in range(P.n_vertices())]

    # get ordered normal fan of P
    normal_fan = ordered_normal_fan(P)
    rays = normal_fan.cones(1)

    M = Matrix.zero(normal_fan.nrays(), normal_fan.ngenerating_cones())
    for i, rho in enumerate(normal_fan.cones(1)):
        for idx in rho.star_generator_indices():
            M[i, inverse_shelling_order[idx]] = 1
    # M[i, j] = 1 means the characteristic function of ray i is supported on the j-th (wrt the shelling order) maximal cone

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

    if deg is not None:
        basis_functions = [x[0] for x in basis_functions if x[1] == deg]
    else:
        basis_functions = list(map(lambda x: x[0], basis_functions))

    if return_order:
        return basis_functions, shelling_order_idcs
    return basis_functions

def get_evaluation_map(Sigma):
    """
    Accepts a complete polytopal simplicial fan.
    Returns the evaluation map for the top degree cohomology.
    """

    def e(i):
        x = [0]*Sigma.dim()
        x[i] = 1
        return x

    Theta = [] # Theta[i] = denominator of i-th summand of evaluation function

    for sigma in Sigma.cones(Sigma.dim()):
        f_sigma = 1
        M = Matrix.zero(QQ, Sigma.dim(), Sigma.dim())
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

def ordered_normal_fan(P, order=None):
    """
    Accepts a lattice polytope P.
    Returns the normal fan of P.
    If order == None, the normal fan is constructed in a way such that the bijection
        {vertices of P} <-> {maximal cones of the normal fan of P}
        is index-preserving.
    Else if order is a permutation of the indices of maximal cones of the normal fan,
        the maximal cones are ordered in that way.
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
        vertices_in_facet = list(facet.vertices())
        common_hypersurface = [ieq for ieq in halfspaces if all([ieq in v.incident() for v in vertices_in_facet])]
        #common_hypersurface = [ieq for ieq in facet.ambient_Hrepresentation() if isinstance(ieq, sage.geometry.polyhedron.representation.Inequality)]
        common_hypersurface = common_hypersurface[0]
        rays.append(common_hypersurface.vector()[1:])
    
    # Create list of cones
    for v in vertices:
        rays_of_v = []
        for ieq in v.incident():
            ray = ieq.vector()[1:]
            rays_of_v.append(rays.index(ray))
        cones.append(rays_of_v)

    if order is not None:
        cones = [cones[i] for i in order]

    # build the normal fan
    return Fan(rays=rays, cones=cones)

def squash(P, F):
    """
    Accepts a polytope P and a facet F.
    Returns 
        - a polytope which is the projection of F into a vector space of relative dimension -1
        - the normal fan of this polytope, obtained from the normal fan of P by squashing the facet normal on F
    """

    rho_F = -F.normal_cone().rays()[0].vector() # Outwards pointing facet normal of F
        
    # find an invertible matrix with the ray generator of rho_F in the first column
    while True:
        M = matrix.random(QQ, P.dim(), P.dim())
        M[:, 0] = rho_F
        if M.det() != 0:
            break
    M = M.T.gram_schmidt()[0]
    projection_matrix = M[1:, :]

    # Realize F as polytope in relative dimension -1
    new_vertices = [projection_matrix * vector(x) for x in F.vertices()]
    F_proj = Polyhedron(new_vertices)

    Sigma = ordered_normal_fan(F_proj)

    return F_proj, Sigma, rho_F, projection_matrix

def get_support_function_from_polytope(P, order=None):
    """
    Accepts a simple lattice polytope P.
    Returns the normal fan of P together with the convex support function of the ample Cartier divisor defined by P.
    """

    if not P.is_simple():
        raise ValueError("The polytope P must be simple.")

    # get the vertices
    vertices = P.Vrepresentation()

    # get the normal fan
    normal_fan = ordered_normal_fan(P, order=order)
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

def compute_lefschetz_determinant(P, k=0):
    """
    Accepts a simple lattice polytope P.
    Returns the determinant of the Lefschetz operator from H^k to H^{n-k} given by the
        strictly convex support function of P.
    """

    normal_fan, phi = get_support_function_from_polytope(P)
    ev = get_evaluation_map(normal_fan)
    basis_functions = get_basis_functions(P, deg=k)
    # initialize Lefschetz operator in degree k
    n = P.dim()
    L = ConewisePolynomial.ones(normal_fan)
    for i in range(n-2*k):
        L *= phi
    # L = phi^{n-k}
    M = matrix.zero(QQ, len(basis_functions), len(basis_functions))
    for i, f_1 in enumerate(basis_functions):
        for j, f_2 in enumerate(basis_functions):
            x = ev(f_1 * L * f_2)
            M[i, j] = x(*[0]*P.dim())

    return M.det()

def random_lattice_polytope(n_vertices, dim, simple=False):
    """
    Accepts a nonnegative integer n_vertices and a nonnegative integer dim.
    Returns a random lattice polytope in dimension dim which is the convex hull of n_vertices vertices
        (V-representation need not be minimal)
    If simple=True, returns a simple polytope.
    """

    # initialize lattice
    M = matrix.identity(dim)
    L = IntegralLattice(M)

    while True:
        # get n_vertices random elements of the lattics
        vertices = [L.random_element() for i in range(n_vertices)]
        P = Polyhedron(vertices)
        if P.is_simple():
            return P

def random_lattice_polytope_from_halfspaces(dim, n_halfspaces):
    """
    Accepts a nonnegative integer dim and a nonnegative integer n_halfspaces.
    Returns a simple polytope in dimension dim which is constructed by sampling n_halfspaces random halfspaces.
    """
    
    while True:
        halfspaces = []
        for i in range(n_halfspaces):
            h = [ZZ.random_element() for _ in range(dim+1)]
            halfspaces.append(h)
        P = Polyhedron(ieqs = halfspaces)
        if P.is_compact() and P.is_simple() and not P.is_empty():
            break
    
    #scale up vertices such that they become lattice points
    vertex_denominators = [v.vector().denominator() for v in P.Vrepresentation()]
    total_lcm = lcm(vertex_denominators)
    return P * total_lcm

def express_as_linear_combination(P, f, verbose=False):
    """
    Accepts a polytope P and a conewise polynomial f on the normal fan of P.
    Returns the structure constants for the linear decomposition of f into basis elements
        of the equivariant(!) cohomology ring.
    """

    # TODO: Implement ConewisePolynomial.data as dict, not list. Makes indexing much easier.
    g = f
    structure_constants = []
    basis_functions, shelling_order = get_basis_functions(P, return_order=True)
    normal_fan = ordered_normal_fan(P)
    for i, cone_idx in enumerate(shelling_order):
        a = g[cone_idx] / basis_functions[i][cone_idx]
        a = a.numerator() # even though this is always a polynomial, sage will treat it as a fraction field element
        structure_constants.append(a)
        g -= a*basis_functions[cone_idx]
    
    if verbose:
        print("ordering of the maximal cones:")
        for idx in shelling_order:
            c = normal_fan.generating_cones()[idx]
            print(list(c.rays()))
            print("---------")

        print("basis functions:")
        print(basis_functions)

        print("structure constants:")
        print(structure_constants)
        return

    return structure_constants

def compute_all_lefschetz_determinants(P, log=False, outfile=None):
    if outfile is not None: 
        f = open(outfile, "a")
    if P.volume() == 0:
        summary_str = "P has volume 0, skipping"
    else:
        summary_str = f"P is a polytope of dimension {P.dim()} with volume {P.volume()}. Prime factors of the volume are {[x[0] for x in factor(P.volume()) if x[1] > 0]}"

    if outfile is None:
        print(summary_str)
    else:
        f.write(summary_str + "\n")

    if log:
        if outfile is None:
            print(P.Vrepresentation())
        else:
            f.write(repr(P.Vrepresentation()))
            f.write("\n")

    if P.volume() == 0:
        if outfile is None:
            return
        f.write("-----------------------\n")
        return

    for k in range((P.dim() + 1) // 2):
        det = compute_lefschetz_determinant(P, k)
        prime_factors = [x[0] for x in det.factor()]
        expected_bad_primes = [x[0] for x in factor(factorial(P.dim()))] + [x[0] for x in factor(P.volume()) if x[1] > 0]
        max_prime_divisor_of_volume = max(factor(P.volume()))[0]
        if any([x > max_prime_divisor_of_volume for x in prime_factors]):
            if outfile is None:
                print("Bad prime bigger than biggest prime in volume")
            else:
                f.write("Bad prime bigger than biggest prime in volume")
        if any([x not in expected_bad_primes for x in prime_factors]):
            if outfile is None:
                print("BAD PRIME FOUND")
            else:
                f.write("BAD PRIME FOUND!\n")
        det_factor_summary = f"k = {k}: {det}. Prime factors: {det.factor()}."
        if outfile is None:
            print(det_factor_summary)
        else:
            f.write(det_factor_summary + "\n")
    
    if outfile is not None:
        f.write("--------------------------------\n")
        f.close()

def test_squash():
    P = random_lattice_polytope_from_halfspaces(4, 5)
    Sigma_P = ordered_normal_fan(P)
    F = P.faces(3)[0]

    F_downstairs, Sigma_F, rho_F, projection_matrix = squash(P, F)
    rho_F = [x for x in Sigma_P.cones(1) if x.rays()[0] == rho_F][0]
    chi_F = get_characteristic_function_of_ray(rho_F, Sigma_P)

    # build correspondence between maximal cones of Sigma_F and Sigma_P
    corr = []
    for v in F_downstairs.vertices():
        for idx_w, w in enumerate(P.vertices()):
            if projection_matrix * vector(w) == vector(v):
                corr.append(idx_w)

    ev_upstairs = get_evaluation_map(Sigma_P)
    ev_downstairs = get_evaluation_map(Sigma_F)

    phi_F = get_support_function_from_polytope(F_downstairs)[1]
    
    # build conewise polynomial g := \chi_{\rho_F} \cdot \pi_F^* \phi_F on Sigma_P
    data_upstairs = []
    for idx, v in enumerate(P.vertices()):
        if idx not in corr:
            data_upstairs.append(0) # support of g is the star of \rho_F
        else:
            # f_downstairs is the polynomial which defines phi_F on the cone of Sigma_F
            # to which the maximal cone of Sigma_P with index idx maps.
            idx_downstairs = corr.index(idx)
            print(idx_downstairs)
            f_downstairs = phi_F[idx_downstairs]
            data_upstairs.append(projection_matrix.act_on_polynomial(f_downstairs) * chi_F[idx_downstairs])
    pi_star_phi = ConewisePolynomial(Sigma_P, data_upstairs)

    return pi_star_phi