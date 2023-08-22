# Solving an OCP using CasADi

import casadi as ca
import numpy as np
import time
from math import *
import matplotlib.pyplot as plt

from draw_sim_obs import simulate

#Initial and Final states
x_init = 0
y_init = 0
theta_init = 0
x_target = 1.5
y_target = 1.5
theta_target = 0

def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0

def DM2Arr(dm):
    return np.array(dm.full())
# Setting up simulation parameters:

T = 0.2 # Sampling time in seconds
N = 3 # Prediction horizon
sim_time = 20 #Maximum simulation time
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
    v*ca.cos(theta),
    v*ca.sin(theta),
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

## In multistep problem this is not required since we are optimizing for X also

# for k in range(N):
#     st = X[:,k] # Previous state extracted from X
#     con = U[:,k] # Control input extracted from U
#     f_value = f(st,con) # Pass input for function f, the output of which is RHS
#     st_next = st+(T*f_value) #X0+dt*RHS
#     X[:,k+1] = st_next # X matrix is now populated based on function f

# Function to get the optimal trajectory knowing the optimal solution
# ff = ca.Function('ff',[U,P],[X])

# Calculating the objective function

obj = 0 # Objective function
g = X[:, 0] - P[:n_states] # Constraints in the equation

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
    obj += ca.mtimes(ca.mtimes((st-P[n_states:]).T,Q),st-P[n_states:]) + ca.mtimes(ca.mtimes(con.T,R),con) # Objective function summation over N iterations
    # obj += (st-P[n_states:]).T @ Q @ (st-P[n_states:])+con.T @ R @ con # @ represents matrix mulitiplication (ca.mtimes)
    
    # The following lines are added for multi-shooting using Forward-Euler
    # f_value = f(st,con)
    # st_next = X[:, k+1]
    # st_next_fwd_euler = st+(T*f_value)
    # g = ca.vertcat(g,st_next-st_next_fwd_euler)

    # The following lines are added for multi-shooting using RK45
    k1 = f(st,con)
    k2 = f(st+((T/2)*k1),con)
    k3 = f(st+((T/2)*k2),con)
    k4 = f(st+(T*k3),con)
    st_next = X[:, k+1]
    st_next_rk45 = st+(T/6)*(k1+2*k2+2*k3+k4)
    g = ca.vertcat(g,st_next-st_next_rk45)

# Add constraints for collision avoidance
obs_x = 0.5 # in m
obs_y = 0.5 # in m
obs_diam = 0.3 # in m
for k in range(N+1):
    g = ca.vertcat(g,(-ca.sqrt((X[0,k]-obs_x)*(X[0,k]-obs_x)+(X[1,k]-obs_y)*(X[1,k]-obs_y))+(obs_diam/2+rob_diam/2)))

# Defining the non-linear programming structure
opt_variables = ca.vertcat(X.reshape((-1,1)),
                            U.reshape((-1,1))) # Reshape U from a 2D array to a vector using Casadi reshape

nlp_prob = {
    'f':obj,
    'x':opt_variables,
    'g':g,
    'p':P
}

opts = {
    'ipopt': {
        'max_iter':100,
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
    'lbg' : np.zeros((n_states+1)*(N+1)), # Lower bound of states x and y
    'ubg' : np.zeros((n_states+1)*(N+1)), # Upper bound of states x and y 
    # Input constraints
    'lbx' : np.zeros(n_states*(N+1)+(n_controls*N)),
    'ubx' : np.zeros(n_states*(N+1)+(n_controls*N))
}

# Inequality constraints for g
args['lbg'][3*(N+1):] = -np.inf
args['ubg'][3*(N+1):] = 0

# State constraints for x, y, and theta
args['lbx'] = [-2 if i % 3 == 0 else (-2 if (i-1) % 3 == 0 else -np.inf) for i in range(3*(N+1))]
args['ubx'] = [2 if i % 3 == 0 else (2 if (i-1) % 3 == 0 else np.inf) for i in range(3*(N+1))]

# Input constraints for v and omega
args['lbx'] += [v_min if i % 2 == 0 else omega_min for i in range(2*N)]
args['ubx'] += [v_max if i % 2 == 0 else omega_max for i in range(2*N)]

# Simulation Loop
t0 = 0

#DM is a dense matrix which is a 2D array with arbitrary dimensions and a fixed data type
x0 = ca.DM([x_init,y_init,theta_init])
xs = ca.DM([x_target, y_target, theta_target])

t = ca.DM(t0)
u0 = ca.DM.zeros((n_controls,N)) #Initial control - Is Nxn_controls in video lecture
X0 = ca.repmat(x0,1,N+1) #Initial state full

#Start MPC
mpc_iter = 0
cat_states = DM2Arr(X0) #Store predicted state 
cat_controls = DM2Arr(u0[:,0]) # Store control actions
times = np.array([[0]])


if __name__ == '__main__':
    main_loop = time.time()
    while (ca.norm_2(x0-xs) > 1e-2) and mpc_iter < sim_time/T:
        t1 = time.time()
        args['p'] = ca.vertcat(
            x0,    # current state
            xs     # target state
            )
        # optimization variable current state as a 1D vector
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1),1),
            ca.reshape(u0,n_controls*N,1))

        sol = solver(x0=args['x0'], 
                    lbx = args['lbx'], 
                    ubx = args['ubx'], 
                    lbg = args['lbg'], 
                    ubg = args['ubg'], 
                    p= args['p'])
        
        # Extract the control input
        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)
        # Compute optimal solution trajectory
        # Compute state given new control input

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        t = np.vstack((
            t,
            t0
        ))

        t0, x0, u0 = shift_timestep(T, t0, x0, u, f)

        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        # xx ...
        t2 = time.time()

        times = np.vstack((
            times,
            t2-t1
        ))

        mpc_iter += 1

    main_loop_time = time.time()
    ss_error = ca.norm_2(x0 - xs)
    control_actions = (ca.reshape(cat_controls,n_controls,-1))
    plt.plot(t,control_actions[0,:].T)
    plt.show
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)
    # simulate
    simulate(cat_states, cat_controls, times, T, N,
            np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), save=False, obs_x=obs_x, obs_y=obs_y, obs_dia=obs_diam)