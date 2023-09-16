# Autonomous Driving with MPC

import casadi as ca
import numpy as np
import time
from math import *
import matplotlib.pyplot as plt

from draw_sim import simulate

# Initial and final states: these will change as the simulation steps forward
x_init = 0
y_init = 0
psi_init = 0
v_init = 0

x_target = 10
y_target = 10
psi_target = pi/2
v_target = 0

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

dt = 0.2
N = 20
sim_time = 200

Lf = 2.67

x_max = 20
x_min = -20

y_max = 20
y_min = -20

psi_max = np.inf
psi_min = -np.inf

v_max = 25
v_min = -4

a_max = 3
a_min = -4

delta_max = radians(30)
delta_min = radians(-30)

jerk_max = 0.6
jerk_min = -1

d_delta_max = radians(20)
d_delta_min = radians(-20)

# Defining six symbols for the states
x = ca.SX.sym('x')
y = ca.SX.sym('y')
psi = ca.SX.sym('psi')
v = ca.SX.sym('v')

states = ca.vertcat(
    x,
    y,
    psi,
    v,
)

n_states = states.numel()

# Defining two symbols for the two control actions - a and delta

a = ca.SX.sym('a')
delta = ca.SX.sym('delta')

controls = ca.vertcat(
    a,
    delta
)

n_controls = controls.numel()

# RHS of the system - 6x1 system
rhs = ca.vertcat(
    v*ca.cos(psi),
    v*ca.sin(psi),
    (v/Lf)*delta,
    a
)

f = ca.Function('f',[states,controls],[rhs])

U = ca.SX.sym('U',n_controls,N)

P = ca.SX.sym('P',2*n_states)

X = ca.SX.sym('X',n_states,(N+1))

obj = 0
g = X[:,0]-P[:n_states]

Q11 = 8.0
Q22 = 8.0
Q33 = 1.0
Q44 = 10.0

Q = ca.diagcat(Q11,Q22,Q33,Q44)

R11 = 0.5
R22 = 0.05

R = ca.diagcat(R11,R22)

S11 = 10
S22 = 0.5

S = ca.diagcat(S11,S22)

for k in range(N):
    st = X[:,k]
    con = U[:,k]

    obj += ca.mtimes([(st-P[n_states:]).T,Q,(st-P[n_states:])]) + ca.mtimes([con.T,R,con])
    k1 = f(st,con)
    k2 = f(st+((dt/2)*k1),con)
    k3 = f(st+((dt/2)*k2),con)
    k4 = f(st+(dt*k3),con)
    st_next = X[:, k+1]
    st_next_rk45 = st+(dt/6)*(k1+2*k2+2*k3+k4)
    g = ca.vertcat(g,st_next-st_next_rk45)

    if k > 0:
        prev_con = U[:,k-1]
        control_rate = con-prev_con
        obj += ca.mtimes([control_rate.T,S,control_rate])
        g = ca.vertcat(g,control_rate)

opt_variables = ca.vertcat(X.reshape((-1,1)),
                            U.reshape((-1,1)))

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

args = {
    'lbg' : np.zeros(n_states*(N+1)+2*(N-1)), # Lower bound of states x and y
    'ubg' : np.zeros(n_states*(N+1)+2*(N-1)), # Upper bound of states x and y 
    # Input constraints
    'lbx' : np.zeros(n_states*(N+1)+(n_controls*N)),
    'ubx' : np.zeros(n_states*(N+1)+(n_controls*N))
}

for i in range(n_states*(N+1),(g.numel())-1,2):
    args['lbg'][i] = jerk_min
    args['lbg'][i+1] = d_delta_min
    args['ubg'][i] = jerk_max
    args['ubg'][i+1] = d_delta_max

for i in range(0,n_states*(N+1)-(n_states-1),n_states):
    args['lbx'][i] = x_min
    args['lbx'][i+1] = y_min
    args['lbx'][i+2] = psi_min
    args['lbx'][i+3] = v_min

    args['ubx'][i] = x_max
    args['ubx'][i+1] = y_max
    args['ubx'][i+2] = psi_max
    args['ubx'][i+3] = v_max


for i in range(n_states*(N+1),len(args['lbx'])-(n_controls-1),n_controls):
    args['lbx'][i] = a_min
    args['lbx'][i+1] = delta_min

    args['ubx'][i] = a_max
    args['ubx'][i+1] = delta_max

# Simulation Loop
t0 = 0

#DM is a dense matrix which is a 2D array with arbitrary dimensions and a fixed data type
x0 = ca.DM([x_init,y_init,psi_init,v_init])
xs = ca.DM([x_target, y_target, psi_target,v_target])

t = ca.DM(t0)
u0 = ca.DM.zeros((n_controls,N)) #Initial control - Is Nxn_controls in video lecture
X0 = ca.repmat(x0,1,N+1) #Initial state full

#Start MPC
mpc_iter = 0
cat_states = DM2Arr(X0) #Store predicted state 
cat_controls = DM2Arr(u0[:,0]) # Store control actions
state_outputs = DM2Arr(x0)
times = np.array([[0]])


if __name__ == '__main__':
    main_loop = time.time()
    while (ca.norm_2(x0-xs) > 1e-2) and mpc_iter < sim_time/dt:
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

        state_outputs = np.vstack((
            state_outputs,
            (x0)
        ))

        t0, x0, u0 = shift_timestep(dt, t0, x0, u, f)

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
    vehicle_state = (ca.reshape(state_outputs,n_states,-1))

    fig,ax = plt.subplots(2,1)
    ax[0].plot(t,control_actions[0,:].T)
    ax[0].set_title('Acceleration vs Time')
    ax[1].plot(t,control_actions[1,:].T)
    ax[1].set_title('Steering Angle vs Time')

    fig1, ax1 = plt.subplots(4,1)
    ax1[0].plot(t,vehicle_state[0,:].T)
    ax1[0].set_title('X-Position vs Time')
    ax1[1].plot(t,vehicle_state[1,:].T)
    ax1[1].set_title('Y-Position vs Time')
    ax1[2].plot(t,vehicle_state[2,:].T)
    ax1[2].set_title('Heading vs Time')
    ax1[3].plot(t,vehicle_state[3,:].T)
    ax1[3].set_title('Velocity vs Time')
    
    plt.show
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)
    # simulate
    simulate(cat_states, cat_controls, times, dt, N,
            np.array([x_init, y_init, psi_init, x_target, y_target, psi_target]), save=False)