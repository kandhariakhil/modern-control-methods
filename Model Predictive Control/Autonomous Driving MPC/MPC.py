# Autonomous Driving with MPC

import casadi as ca
import numpy as np
import time
from math import *
import matplotlib.pyplot as plt

# Initial and final states: these will change as the simulation steps forward
x_init = 0
y_init = 0
psi_init = 0
v_init = 0

x_target = 15
y_target = 15
psi_target = pi
v_target = 0

dt = 0.1
N = 10
sim_time = 200

Lf = 2.67

v_min = -4
v_max = 25

a_min = -3
a_max = 3

delta_min = radians(-30)
delta_max = radians(30)

jerk_min = -0.6
jerk_max = 0.6

d_delta_min = radians(-6)
d_delta_max = radians(6)

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

Q11 = 1.0
Q22 = 1.0
Q33 = 0.25
Q44 = 2.0

Q = ca.diagcat(Q11,Q22,Q33,Q44)

R11 = 0.5
R22 = 0.05

R = ca.diagcat(R11,R22)

S11 = 1
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

