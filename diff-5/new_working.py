import os
import sys
import time
import h5py
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from scipy.misc import derivative

from mpi4py import MPI
import dedalus.public as d3
from dedalus.extras.flow_tools import GlobalArrayReducer

import logging
logger = logging.getLogger(__name__)

args = sys.argv
# Parameters
radius = 1
Lmax = 127
L_dealias = 3/2
N_phi = 16
Nmax = 127
N_dealias = 3/2
dealias_tuple = (1, L_dealias, N_dealias)
# I decreased the timestep size for stability
timestep = 0.0005
t_end = 2000
ts = d3.SBDF2
dtype = np.float64
nu = float(args[1])
eta = float(args[1])
kappa = float(args[1])
mesh = [2,64] 

goodm = 3
omega = 2. 
r_0 = 0.875
delta_r = 0.04
tau = 3.
# location of top damping layer
r_top = 0.95

# Bases
c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor(c, dtype=dtype, mesh=mesh)
b = d3.BallBasis(c, shape=(N_phi, Lmax+1, Nmax+1), radius=radius, dealias=dealias_tuple, dtype=dtype)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, L_dealias, N_dealias))

# Fields
u = d.VectorField(c, bases=b, name='u')
#rho = d.Field(bases=b, name='rho')
A   = d.VectorField(c, bases=b, name='A')
p   = d.Field(bases=b, name='p') 
Phi_field = d.Field(bases=b, name='Phi')

tau_p = d.Field(name='tau_p')
tau_Phi = d.Field(name='tau_Phi')
tau_A = d.VectorField(c, bases=b_S2, name='tau_A')
tau_u = d.VectorField(c, bases=b_S2, name='tau_u')
#tau_rho = d.Field(bases=b_S2, name='tau_rho')

B_0   = d.VectorField(c, bases=b, name='B_0')
#rho_0 = d.Field(bases=b.radial_basis, name='rho_0')
g     = d.VectorField(c, bases=b.radial_basis, name='g')
#D_N   = d.Field(bases=b.radial_basis, name='D_N')

for fd in [B_0]:
    fd.set_scales(dealias_tuple)

#rho_0['g'] = -r**2 #change this to change N^2, must be function of r^2
#g['g'][2] = -9*r
#D_N['g'] = (1+np.tanh((r-r_top)/delta_r))/(2.*tau)
#read in ICs
f = h5py.File('../test_outputs/scalar/scalar_s1/scalar_s1_p0.h5')
ic = np.array(f['tasks/ui'])
u.load_from_global_grid_data(ic[0])
u.set_scales(dealias_tuple)

#Initial magneic field
def f_lambda(r1,r2):
    return special.spherical_jn(1,lam*r2)*special.spherical_yn(1,lam*r1)-special.spherical_jn(1,lam*r1)*special.spherical_yn(1,lam*r2)
def psi(r):
    psir = 0*r 
    for i in range(len(r[0,0,:])):
                
        pre = beta*lam*r[0][0][i]/special.spherical_jn(1,lam)
        first = f_lambda(r[0][0][i],1)*integrate.quad(lambda xi: xi**3*special.spherical_jn(1,lam*xi),0,r[0][0][i])[0]
        second = special.spherical_jn(1,lam*r[0][0][i])*integrate.quad(lambda xi: xi**3*f_lambda(xi,1),r[0][0][i],1)[0]
        psir[:,:,i] = pre*(first+second)
                                                            
    return psir
                                                                
beta = 1 
lam = 5.76346
b_phi = -lam*np.sin(theta)*psi(r)/r
b_theta = -np.sin(theta)*derivative(psi,r,dx=1e-6)/r
b_r = 2*np.cos(theta)*psi(r)/r**2
B_0.set_scales((1,L_dealias, N_dealias))                                                
B_0['g'][0] = b_phi
B_0['g'][1] = b_theta
B_0['g'][2] = b_r 

# Parameters and operators
ez = d.VectorField(c, bases=b, name='ez')
ez.set_scales(dealias_tuple)
ez['g'][1] = -np.sin(theta)
ez['g'][2] = np.cos(theta)

er = d.VectorField(c, name='er')
er['g'][2] = 1
LiftTau = lambda A: d3.LiftTau(A, b, -1)
r_out = 1
ell_func = lambda ell: ell+1
A_potential_bc = d3.radial(d3.grad(A)(r=1)) + d3.SphericalEllProduct(A, c, ell_func)(r=1)/r_out
stress = d3.grad(u) + d3.TransposeComponents(d3.grad(u))

grid_B0 = d3.Grid(B_0).evaluate()
grid_J0 = d3.Grid(d3.curl(B_0)).evaluate()

integ = lambda A: d3.Integrate(A, c)

# Problem
problem = d3.IVP([p, u, A, Phi_field, tau_u, tau_A], namespace=locals())
'''
problem.add_equation("div(u) + tau_p = 0") #incompressibility 
problem.add_equation("dt(u) + grad(p) - nu*lap(u) + LiftTau(tau_u) = -cross(lap(A), grid_B0) + cross(grid_J0, curl(A))") #momentum
problem.add_equation("div(A) + tau_Phi = 0")
problem.add_equation("dt(A) - grad(Phi_field) - eta*lap(A) + LiftTau(tau_A) = cross(u, grid_B0)")
problem.add_equation("integ(p) = 0")
problem.add_equation("integ(Phi_field) = 0")
problem.add_equation("radial(u(r=1)) = 0")
problem.add_equation("angular(radial(stress(r=1)),index=1) = 0")
problem.add_equation("A_potential_bc = 0")
'''
problem.add_equation("div(u) = 0", condition="ntheta != 0") #incompressibility
problem.add_equation("p = 0", condition="ntheta == 0")
problem.add_equation("dt(u) + grad(p) - nu*lap(u) + LiftTau(tau_u) = -cross(lap(A), grid_B0) + cross(grid_J0, curl(A))", condition="ntheta != 0") #momentum
problem.add_equation("u = 0", condition = "ntheta == 0")
problem.add_equation("div(A) = 0", condition="ntheta != 0")
problem.add_equation("Phi_field = 0", condition="ntheta == 0")
problem.add_equation("dt(A) - grad(Phi_field) - eta*lap(A) + LiftTau(tau_A) = cross(u, grid_B0)", condition="ntheta != 0")
problem.add_equation("A = 0", condition="ntheta == 0")
problem.add_equation("radial(u(r=1)) = 0", condition="ntheta != 0")
problem.add_equation("tau_u = 0", condition="ntheta == 0")
problem.add_equation("angular(radial(stress(r=1))) = 0", condition="ntheta != 0")
problem.add_equation("A_potential_bc = 0", condition="ntheta != 0")
problem.add_equation("tau_A = 0", condition="ntheta == 0")

# Solver
solver = problem.build_solver(ts)
solver.stop_sim_time = t_end
logger.info("Problem built")
integ = lambda A: d3.Integrate(A, c)

# Analysis
output_dir = './test_outputs/'
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(output_dir)):
        os.makedirs('{:s}/'.format(output_dir))

B_vec = d3.curl(A)
scalars = solver.evaluator.add_file_handler(output_dir+'scalar', max_writes=np.inf, iter=100)
scalars.add_task(integ(0.5*d3.dot(u, u)),  name='KE')
scalars.add_task(integ(0.5*d3.dot(B_vec, B_vec)),  name='ME')
scalars.add_task(integ(d3.dot(B_0, d3.curl(d3.cross(u,B_vec)))) , name='B_change')
KE_op = scalars.tasks[0]['operator']
ME_op = scalars.tasks[1]['operator']

slices = solver.evaluator.add_file_handler(output_dir+'slices', max_writes=20, sim_dt=1.0)
slices.add_task(u(phi=np.pi/4), name='u_mer(phi=pi/4)')
slices.add_task(u(phi=5*np.pi/4), name='u_mer(phi=5pi/4)')
slices.add_task(B_vec(phi=np.pi/4), name='B_mer(phi=pi/4)')
slices.add_task(B_vec(phi=5*np.pi/4), name='B_mer(phi=5pi/4)')
#slices.add_task(rho(phi=np.pi), name='rho(phi=pi)')
#slices.add_task(rho(phi=0), name='rho(phi=0)')

file_handler_mode = 'overwrite'
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=5, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

reducer = GlobalArrayReducer(d.comm_cart)

output_cadence = 10000
file_num = 0
logger.info("done with analysis")
#hermitian cadence so it doesn't blow up??
hermitian_cadence = 100
#build mask for zeroing not good m data
shape = p['c'].shape
grid_space = (False,False,False)
good = np.zeros(shape, dtype=bool)
zero_indices = np.ones(shape, dtype=bool)
for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            e = d.coeff_layout.local_elements(b.domain, scales=1)
            #elements = (np.array((i,)),np.array((j,)),np.array((k,)))
            elements = (np.array((e[0][i],)), np.array((e[1][j],)), np.array((e[2][k],)))
            m,l,n = b.elements_to_groups(grid_space, elements)
            
            if m[0]==goodm:
                good[i,j,k] = True
zero_indices[good] = False
# Main loop
start_time = time.time()
while solver.proceed:

    if solver.iteration % 100 == 0:
        op_output = KE_op.evaluate()['g']
        meop_output = ME_op.evaluate()['g']
        if d.comm_cart.rank == 0:
            KE0 = op_output.min()
            ME0 = meop_output.min()
        else:
            KE0 = ME0 = 0
        logger.info("t = %f, KE = %e, ME = %e" %(solver.sim_time, KE0, ME0))
    if solver.iteration % 1e4 == 0:
        p['c'][zero_indices] = 0
        u['c'][:,zero_indices] = 0
        A['c'][:,zero_indices] = 0
        Phi_field['c'][zero_indices] = 0
    if solver.iteration % hermitian_cadence in [0,1]:
        for f in solver.state:
            f.require_grid_space()
    #F['g'][2] = F_func(solver.sim_time)
    solver.step(timestep)
end_time = time.time()
logger.info(end_time-start_time)
