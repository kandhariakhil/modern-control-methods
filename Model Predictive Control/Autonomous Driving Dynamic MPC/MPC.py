# Autonomous Driving with MPC using Tire and Drive Train Dynamics

import casadi as ca
import numpy as np
import time
from math import *
import matplotlib.pyplot as plt

from importParameters import importParameters

from draw_sim import simulate

parameters = importParameters()
mp = parameters.getModelParameters()
lim = parameters.getLimits()

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

# Ff,y
def calFrontForce(omega, vx, vy, delta):
    alpha = -ca.atan2(((omega * mp['lf']) + vy), vx) + delta
    frontForce = mp['Df'] * (ca.sin(mp['Cf'] * ca.atan2(mp['Bf'] * alpha,1)))
    
    return frontForce

# Fr,y
def calRearForce(omega, vx, vy):
    alpha = ca.atan2(((omega * mp['lr']) - vy), vx)
    rearForce = mp['Dr'] * ca.sin(mp['Cr'] * ca.atan2(mp['Br'] * alpha,1))
    
    return rearForce

# Fr,x
def calLongForce(vx,d):
    longForce = (mp['cm1']-mp['cm2']*vx)*d-mp['Cr']-mp['cd']*vx*vx
    
    return longForce

x_init = 0.0
y_init = 0.0
phi_init = 0.0
vx_init = 0.0
vy_init = 0.0
omega_init = 0.0

x_target = 30.0
y_target = 30.0
phi_target = pi/2
vx_target = 0.0
vy_target = 0.0
omega_target = 0.0

dt = 0.25
N = 1
sim_time = 100

x= ca.SX.sym('x')
y = ca.SX.sym('y')
phi = ca.SX.sym('phi')
vx = ca.SX.sym('vx')
vy = ca.SX.sym('vy')
omega = ca.SX.sym('omega')

states = ca.vertcat(
    x,
    y,
    phi,
    vx,
    vy,
    omega
)

n_states = states.numel()

d = ca.SX.sym('d')
delta = ca.SX.sym('delta')

controls = ca.vertcat(
    d,
    delta
)

n_controls = controls.numel()

front_force = calFrontForce(omega, vx, vy, delta)
front_force_func = ca.Function('front_force_func', [omega, vx, vy, delta], [front_force])

rear_force = calRearForce(omega, vx, vy)
rear_force_func = ca.Function('rear_force_func', [omega, vx, vy], [rear_force])

long_force = calLongForce(vx,d)
long_force_func = ca.Function('long_force',[vx,d],[long_force])

rhs = ca.vertcat(
    vx*ca.cos(phi)-vy*ca.sin(phi),
    vx*ca.sin(phi)+vy*ca.cos(phi),
    omega,
    (1/mp['m'])*(long_force_func(vx,d)-front_force_func(omega,vx,vy,delta)*ca.sin(delta)+mp['m']*vy*omega),
    (1/mp['m'])*(rear_force_func(omega,vx,vy)-front_force_func(omega,vx,vy,delta)*ca.cos(delta)-mp['m']*vx*omega),
    (1/mp['Iz'])*(front_force_func(omega,vx,vy,delta)*mp['lf']*ca.cos(delta)-rear_force_func(omega,vx,vy)*mp['lr'])
)

f = ca.Function('f',[states,controls],[rhs])

U = ca.SX.sym('U',n_controls,N)

P = ca.SX.sym('P',2*n_states)

X = ca.SX.sym('X',n_states,N+1)

obj = 0
g = X[:,0]-P[:n_states]

Q11 = 1.0
Q22 = 1.0
Q33 = 1.0
Q44 = 1.0
Q55 = 1.0
Q66 = 1.0

Q = ca.diagcat(Q11,Q22,Q33,Q44,Q55,Q66)

R11 = 1.0
R22 = 1.0

R = ca.diagcat(R11,R22)

S11 = 1.0
S22 = 1.0

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
    'lbg' : np.zeros(n_states*(N+1)), # Lower bound of states x and y
    'ubg' : np.zeros(n_states*(N+1)), # Upper bound of states x and y 
    # Input constraints
    'lbx' : np.zeros(n_states*(N+1)+(n_controls*N)),
    'ubx' : np.zeros(n_states*(N+1)+(n_controls*N))
}

for i in range(0,n_states*(N+1)-(n_states-1),n_states):
    args['lbx'][i] = lim['x_min']
    args['lbx'][i+1] = lim['y_min']
    args['lbx'][i+2] = lim['phi_min']
    args['lbx'][i+3] = lim['vx_min']
    args['lbx'][i+4] = lim['vy_min']
    args['lbx'][i+5] = lim['omega_min']

    args['ubx'][i] = lim['x_max']
    args['ubx'][i+1] = lim['y_max']
    args['ubx'][i+2] = lim['phi_max']
    args['ubx'][i+3] = lim['vx_max']
    args['ubx'][i+4] = lim['vy_max']
    args['ubx'][i+5] = lim['omega_max']

for i in range(n_states*(N+1),len(args['lbx'])-(n_controls-1),n_controls):
    args['lbx'][i] = lim['d_min']
    args['lbx'][i+1] = lim['delta_min']

    args['ubx'][i] = lim['d_max']
    args['ubx'][i+1] = lim['delta_max']

print(args)

# Simulation Loop
t0 = 0

#DM is a dense matrix which is a 2D array with arbitrary dimensions and a fixed data type
x0 = ca.DM([x_init,y_init,phi_init,vx_init,vy_init,omega_init])
xs = ca.DM([x_target, y_target, phi_target,vx_target,vy_target,omega_target])

t = ca.DM(t0)
u0 = ca.DM.zeros((n_controls,N)) #Initial control - Is Nxn_controls in video lecture
X0 = ca.repmat(x0,1,N+1) #Initial state full

#Start MPC
mpc_iter = 0
cat_states = DM2Arr(X0) #Store predicted state 
cat_controls = DM2Arr(u0[:,0]) # Store control actions
state_outputs = DM2Arr(x0)
times = np.array([[0]])

jac_g_x = ca.jacobian(g, opt_variables)
jac_g_x_function = ca.Function('jac_g_x_function', [opt_variables, P], [jac_g_x])

if __name__ == '__main__':
    main_loop = time.time()
    while (ca.norm_2(x0-xs) > 1e-1) and mpc_iter < sim_time/dt:
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

        # current_jac_g_x = jac_g_x_function(args['x0'], args['p'])
        # print(f"Jacobian of g at iteration {mpc_iter}:", DM2Arr(current_jac_g_x))

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

    plt.show
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)
    # simulate
    simulate(cat_states, cat_controls, times, dt, N,
            np.array([x_init, y_init, phi_init, x_target, y_target, phi_target]), save=False)
'''
if __name__ == '__main__':
    test_omega = 0.1
    test_vx = 5.0
    test_vy = 0.2
    test_delta = 0.05
    test_d = 0.1

    print("Front Force:", calFrontForce(mp['lf'], mp['Bf'], mp['Cf'], mp['Df'], test_omega, test_vx, test_vy, test_delta))
    print("Rear Force:", calRearForce(mp['lr'], mp['Br'], mp['Cr'], mp['Dr'], test_omega, test_vx, test_vy))
    print("Longitudinal Force:", calLongForce(mp['cm1'], mp['cm2'], mp['cr'], mp['cd'], test_vx, test_d))
'''