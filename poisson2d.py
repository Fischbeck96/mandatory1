import numpy as np  
import sympy as sp  
import scipy.sparse as sparse  # (dense matrices would be O(N^4) in 2D, but sparse are O(N^2)).
import scipy.interpolate as interpolate  # SciPy's interpolate module provides tools for interpolating data on a grid, used here for evaluating the solution at arbitrary points.

# Define symbolic variables x and y for use in SymPy expressions.
x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.
    """
    
    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        # Method of Manufactured Solutions (MMS)
        self.L = L
        self.ue = ue
        self.f = sp.diff(ue, x, 2) + sp.diff(ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        # Cartesian grid with spacing h = L/N.
        # indexing='ij' means x varies along rows, y along columns
        x = np.linspace(0, self.L, N+1)
        y = np.linspace(0, self.L, N+1)
        self.N = N
        self.h = self.L / N  # Grid spacing h
        self.xij, self.yij = np.meshgrid(x, y, indexing="ij")

    def D2(self):
        """Return second order differentiation matrix"""
        
        # O(h^2.
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        
        return D / self.h**2  #  to match the derivative approximation.

    def laplace(self):
        """Return vectorized Laplace operator"""
        # Laplacian matrix using Kronecker product.
        
        d2 = self.D2()
        I = sparse.eye(self.N+1)
        return sparse.kron(d2, I) + sparse.kron(I, d2)  # Full discrete Laplacian A.

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
      
        B = np.ones((self.N+1, self.N+1), dtype=bool)
        B[1:-1, 1:-1] = False  # Interior is False.
        return np.where(B.ravel())[0]  # 1D indices of boundary points.

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        
        ffunc = sp.lambdify((x, y), self.f, 'numpy')
        uefunc = sp.lambdify((x, y), self.ue, 'numpy')
        F = ffunc(self.xij, self.yij)  # f on the grid
        U_boundary = uefunc(self.xij, self.yij)  # ue on grid for boundaries.

        # Assemble Laplacian matrix
        A = self.laplace().tolil()  # Convert to LIL for row modifications.
        b = F.ravel()  # Flatten F to 1D vector 
        bnds = self.get_boundary_indices()  # Boundary indices.

        
        # For boundary nodes, set equation to u_i = ue_i, i.e., row i in A is [0,...,1,...0] with 1 at diagonal, b_i = ue_i.
        # u = ue on boundaries
        # Dirichlet means fixed potential/temperature on boundaries
        for i in bnds:
            A[i] = 0  
            A[i, i] = 1 
            b[i] = U_boundary.ravel()[i]  # Set b_i to the boundary value from ue.

        return A.tocsr(), b  # Convert back to CSR (Compressed Sparse Row) for efficient solving.

    def l2_error(self, u):
        """Return l2-error norm"""
        # l2 error: ||u - ue|| = sqrt(h^2 * sum((u_ij - ue_ij)^2)) over all points.
        uefunc = sp.lambdify((x, y), self.ue, 'numpy')  
        U_exact = uefunc(self.xij, self.yij)  # ue on grid.
        return np.sqrt(self.h**2 * np.sum((u - U_exact)**2))  # L2 error.

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array
        """
        self.create_mesh(N) 
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b).reshape((N+1, N+1))  # Solve sparse linear system, reshape back to 2D grid.
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates over m refinements"""
       
        E = []  
        h = []  
        N0 = 8 
        for _ in range(m):
            u = self(N0)  
            E.append(self.l2_error(u))  
            h.append(self.h)  
            N0 *= 2 
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, len(E))]  
        return r, np.array(E), np.array(h)

    def eval(self, x, y): # not my work
        """Return u(x, y) using bilinear interpolation

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y) #schipy interp2d
        """
        # Evaluate solution at arbitrary (x,y) using interpolation from grid.
        # Mathematically: Bilinear interpolation is linear in each direction, accurate to O(h^2) like the method.
        # Python-wise: RegularGridInterpolator from SciPy, with 'linear' method for bilinear.
        # bounds_error=False allows extrapolation if needed, but fill_value=None might raise if out of bounds.
        interp_func = interpolate.RegularGridInterpolator(
            (np.linspace(0, self.L, self.N+1), np.linspace(0, self.L, self.N+1)),
            self.U,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        return interp_func([x, y])[0]  # Evaluate at point, [0] to get scalar.



def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))  # Symbolic exact solution.
    sol = Poisson2D(1, ue)  
    r, E, h = sol.convergence_rates() 
    assert abs(r[-1]-2) < 1e-2  

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)  
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3  # Compare to exact, numerical eval with .n().
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h, y: 1-sol.h/2}).n()) < 1e-3

