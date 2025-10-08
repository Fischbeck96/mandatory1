import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False): #sparse?
        """Create 2D mesh and store in self.xij and self.yij"""
        x = np.linspace(0, 1, N+1)
        y = np.linspace(0, 1, N+1)
        self.xij, self.yij = np.meshgrid(x,y,indexing= "ij") #ij for computational purposes matrix
        
    def D2(self,N): #"had to"ad self
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D/self.h**2
        

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        #to make variables accessble elsewhere;
        self.mx =mx
        self.my=my
        self.create_mesh(N)
        t0=0
        Un= sp.lambdify((x,y,t),self.ue(mx,my))(self.xij,self.yij,t0+self.dt)
        Unm1= sp.lambdify((x,y,t),self.ue(mx,my))(self.xij,self.yij,t0)


        #Un= 0.5*(sp.lambdify((x,y,t),self.ue(mx,my))(self.xij,self.yij,t0+self.c*self.dt)
              #   +sp.lambdify((x,y,t),self.ue(mx,my))(self.xij,self.yij,t0-self.c*self.dt)) # (xyt ->xij,yij,t0) in eeu
        #Unm1= sp.lambdify((x,y,t),self.ue(mx,my))(self.xij,self.yij,t0) #worked when i removed +self.dt
        return Un,Unm1
        #sp.lambdify converts symbols into fct that can be evaluated in numpy
        # returns are meshfunctions
# gost node lec 5
    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.h / self.c 

        
    def l2_error(self, u, t0): # just at a single time t0?
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue = sp.lambdify((x, y, t), self.ue(self.mx, self.my))(self.xij, self.yij, t0)
        return np.sqrt(self.h**2 * np.sum((u - ue)**2))

    def apply_bcs(self):
        #Dirichlet boundary
        self.u[0, :] = 0
        self.u[-1, :] = 0
        self.u[:, 0] = 0
        self.u[:, -1] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.h = 1.0 / N
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my # so this changes all old sellf.my  im trying to do ?
        self.create_mesh(N)
        #start
        Un,Unm1 =self.initialize(N,mx,my)
        Unp1=np.zeros_like(Un) #think to update
        D=self.D2(N) #why i added self i think
        # Dictionary to store solutions if store_data > 0
        if store_data > 0:
            solutions = {0: Un.copy()}  # Store initial solution at t=0
        for n in range(1, Nt+1):#copied from lec 7
            Unp1[:] = 2*Un - Unm1 + (c*self.dt)**2*(D @ Un + Un @ D.T)
            self.u = Unp1
            self.apply_bcs()
            Unm1[:] = Un
            Un[:] = Unp1 #shoud this be changed since i wrote  self.u= Unp1?
            if store_data > 0 and n % store_data == 0:
                solutions[n] = Un.copy() # lagrer kopi i solutions, 
        if store_data > 0:
            return solutions
        return self.h, [self.l2_error(Un, (Nt+1) * self.dt)]  # Wrap l2_err in a list    

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = -2, 2, 0, 0
        D[-1, -4:] = 0, 0, 2, -2
        return D/self.h**2

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)


    def apply_bcs(self):
        return None

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    
    assert abs(r[-1]-2) < 1e-2 

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2.0, my=2.0,cfl=(1/np.sqrt(2)))
    assert np.max(E) < 1e-12
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=2,cfl=1/np.sqrt(2))
    assert np.max(E) < 1e-12
    
    
    
