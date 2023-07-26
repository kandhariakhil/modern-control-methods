# Solving an OCP using CasADi

import casadi as ca
import numpy as np
import time
from math import *

from draw_sim import Draw_MPC_point_stabilization

def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0

# Setting up simulation parameters:

T = 0.2 # Sampling time in seconds
N = 3 # Prediction horizon
# Prediction time = 0.2*3 = 0.6 seconds
rob_diam = 0.3 # Robot dimensions

v_max = 0.6
v_min = -v_max

omega_max = ca.pi/4
omega_min = -omega_max

# Defining three symbols for the three states - x, y and  theta
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')

states =  ca.vertcat(
    x, 
    y, 
    theta) # np.array([[x],[y],[theta]])

n_states = states.numel() #len(states)

# Defining two symbols for the control actions - v and omega

v = ca.SX.sym('v')
omega = ca.SX.sym('omega')

controls = ca.vertcat(
    v, 
    omega)

n_controls = controls.numel()

# RHS of the system - 3x1 system
rhs = ca.vertcat(
    v*ca.sin(theta),
    v*ca.cos(theta),
    omega)

# Nonlinear function mapping using Casadi f(x,u)
f = ca.Function('f',[states,controls],[rhs])
# Decision variables (controls)
U = ca.SX.sym('U',n_controls,N)

# Parameters which includes the initial and the reference state of the robot
P = ca.SX.sym('P',n_states+n_states)

# Matrix that represents the states over the optimization problem
X = ca.SX.sym('X',n_states,(N+1))

# Compute solution symbolically

X[:,0] = P[0:n_states] # Initial state

for k in range(N):
    st = X[:,k] # Previous state extracted from X
    con = U[:,k] # Control input extracted from U
    f_value = f(st,con) # Pass input for function f, the output of which is RHS
    st_next = st+(T*f_value) #X0+dt*RHS
    X[:,k+1] = st_next # X matrix is now populated based on function f

# Function to get the optimal trajectory knowing the optimal solution
ff = ca.Function('ff',[U,P],[X])

# Calculating the objective function

obj = 0 # Objective function
g = [] # constraints vector

# Q and R are diagonal matrices used to tune the controller
Q11 = 1.0 
Q22 = 5.0
Q33 = 0.1
Q = ca.diagcat(Q11,Q22,Q33)

R11 = 0.5 
R22 = 0.05
R = ca.diagcat(R11,R22)

for k in range(N):
    st = X[:,k]
    con = U[:,k]
    # Note: Casadi multiplication is using mtimes and the .T transposes the array
    # P0,P1,P2 - stores the initial state and P3,P4,P5 stores the reference state
    # obj += ca.mtimes(ca.mtimes((st-P[3:6])T,Q),st-P[3:6]) + ca.mtimes(ca.mtimes(con.T,R),con) # Objective function summation over N iterations
    obj += (st-P[N:]).T @ Q @ (st-P[N:])+con.T @ R @ con # @ represents matrix mulitiplication (ca.mtimes)

# Compute constraints - box constraints due to map margins (x,y) -> cannot be outside of map margins
for k in range(N+1):
    g = ca.vertcat(g,X[0,k]) # state x
    g = ca.vertcat(g,X[1,k]) # state y

# Defining the non-linear programming structure

opt_variables = ca.vertcat(U.reshape((-1,1))) # Reshape U from a 2D array to a vector using Casadi reshape

nlp_prob = {
    'f':obj,
    'x':opt_variables,
    'g':g,
    'p':P
}

opts = {
    'ipopt': {
        'max_iter':1000,
        'print_level':0,
        'acceptable_tol':1e-8,
        'acceptable_obj_change_tol':1e-6
    },
    'print_time':0
}

solver = ca.nlpsol('solver','ipopt',nlp_prob,opts)

# Setting maximum and minimum for [v,omega,v,omega,v,omega]
args = {
    # Inequality constraints (state constraints)
    'lbg' : -2, # Lower bound of states x and y
    'ubg' : 2, # Upper bound of states x and y 
    # Input constraints
    'lbx' : [v_min]*(2*N),
    'ubx' : [v_max]*(2*N)
}

args['lbx'][1::2] = [omega_min]*N
args['ubx'][1::2] = [omega_max]*N

# Simulation Loop
t0 = 0
x0 = np.array([0,0,0.0]) #Initial condition
xs = np.array([1.5,1.5,0]) #Reference position
xx = np.zeros((3,N+1))
t = np.zeros(N+1)
xx[:,0] = x0
t[0] = t0
u0 = np.zeros((N,2)) # Two control inputs
sim_time = 20 #Maximum simulation time

#Start MPC
mpciter = 0
xx1 = []
u_cl = []

main_loop = time.time()
while (np.linalg.norm(x0- xs, 2) > 1e-1) and mpciter < sim_time/T:
    # args['p'] = np.concatenate((x0,xs)) # Set the values of the paremeters vector
    # args['x0'] = u0.flatten() #Initial value of the optimization variables

    args['p'] = ca.vertcat(
        x0,    # current state
        xs   # target state
        )
        # optimization variable current state
    args['x0'] = np.array(u0.T).flatten()
    sol = solver(x0=args['x0'], lbx = args['lbx'], ubx = args['ubx'], lbg = args['lbg'], ubg = args['ubg'], p= args['p'])
    # sol = (**args)

    # Extract the control input
    u = np.reshape(sol['x'], (N,2)).T

    # Compute optimal solution trajectory
    # Todo: Understand the following line
    ff_value = ff(u.T.reshape(2,N), args['p'])

    xx1.append(ff_value.full().T)
    u_cl = np.concatenate((u_cl,u[0]), axis=0) # Storing control actions

    t = ca.vertcat(
            t,
            t0
        )

    # Get the initialization of the next optimization step
    t0, x0, u0 = shift_timestep(T, t0, x0, u, f)
    xx[:,mpciter+1] = (x0).flatten()

    mpciter += 1

main_loop_time = time.time() - main_loop

ss_error = np.linalg.norm(x0 - xs, 2)

Draw_MPC_point_stabilization(t, xx, np.array(xx1), u_cl, xs, N, rob_diam)  # a drawing function

