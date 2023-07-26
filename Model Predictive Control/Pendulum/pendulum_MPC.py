# Tutorial by Simone Bertoni
'''
Pendulum MPC Example
Variables used:
-> theta - angular position
-> dtheta - angular position
-> ddtheta - angular acceleration
-> tau - torque
-> l - length of pendulum
-> m - mass of pendulum
-> k - damping coefficient
-> g - acc. due to gravity
-> dt - time step 
'''

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import (
    minimize,
    LinearConstraint,
)  # We will write our own optimizer also


# EOM of pendulum to emulate system - this function is not used in MPC
def pendulum_eom(theta, dtheta, tau, l, k, m, g, dt):

    # Angular acceleration
    ddtheta = (tau - k * dtheta - m * g * l * math.sin(theta)) / (m * l * l)

    # Angular velocity
    dtheta = dtheta + ddtheta * dt
    
    # Angular position
    theta = theta + dtheta * dt

    return (theta, dtheta)

# Cost function to be minimized
def mpc_cost(tau, theta_ref, theta_init, dtheta_init, l, k, m, g, dt, Q11, Q22, R, N):
    cost = 0
    theta = theta_init
    dtheta = dtheta_init
    
    for i in range(N):
        ddtheta = (tau[i] - k * dtheta - m * g * l * math.sin(theta)) / (m * l * l)

        dtheta = dtheta + ddtheta * dt
        theta = theta + dtheta * dt

        cost += Q11 * dtheta**2 + Q22 * (theta_ref - theta) ** 2 + R * tau[i] ** 2

    return cost

def solve_mpc(theta_ref, theta, dtheta, tau_init, l, k, m, g, dt, Q11, Q22, R, N, tau_max, delta_tau_max):
    # Rate of change implemented using LinearConstraint as -delta_tau_max <= delta_tau_matrix*tau <= delta_tau_max
    delta_tau_matrix = np.eye(N)-np.eye(N,k=1)
    constraint1 = LinearConstraint(delta_tau_matrix, -delta_tau_max, delta_tau_max)
    
    # Constraint on rate of change of tau[0] wrt it's previous value, which is tau_init[0]
    first_element_matrix = np.zeros([N,N])
    first_element_matrix[0,0] = 1
    constraint2 = LinearConstraint(first_element_matrix, tau_init[0]-delta_tau_max,tau_init[0]+delta_tau_max)

    # Add constraints
    delta_tau_constraint = [constraint1, constraint2]

    # Bounds for max torque
    bounds = [(-tau_max,tau_max) for idx in range(N)]

    # Starting optimization point for theta nad dtheta are the current measurements
    theta0 = theta
    dtheta0 = dtheta
    
    # Minimization using in-built optimizer
    result = minimize(mpc_cost, tau_init, args=(theta_ref,theta0,dtheta0, l, k, m, g, dt, Q11, Q22, R, N), bounds=bounds, constraints=delta_tau_constraint)

    tau_mpc = result.x
    print(tau_mpc)
    return tau_mpc

# Simulation Initialization

l = 1
k = 0.5
m = 0.5
g = 9.81

dt = 0.1

time_range = 10

# Simulation steps
L = round(time_range/dt)

# Init time
time = 0

#Init state
theta0 = 0
dtheta0 = 0

# Arrays for logging
theta = np.zeros(L+1)
dtheta = np.zeros(L+1)
tau = np.zeros(L+1)
theta_ref = np.zeros(L)

# Init arrays to initial state
theta[0] = theta0
dtheta[0] = dtheta0

# Control system calibration

N = 20

Q11 = 0
Q22 = 1
R = 0

tau_max = 10
delta_tau_max = 1

l_est = 0.98
k_est = 0.52
m_est = 0.52
g_est = 9.81

tau_init = np.zeros(N)

# Simulation

for idx in range(L):

# Generate ref setpoint:
    if time < 5:
        theta_ref[idx] = math.pi
    else:
        theta_ref[idx] = math.pi*0.5

    time += dt

    #Control system loop
    tau_mpc = solve_mpc(theta_ref[idx],theta[idx],dtheta[idx], tau_init, l_est, k_est, m_est, g_est, dt, Q11, Q22, R, N, tau_max, delta_tau_max)

    # use first element of control input optimzal solution
    tau[idx] = tau_mpc[0]

    # Initial solution for next step = current solution
    tau_init = tau_mpc

    # Run dynamic system
    (theta[idx+1],dtheta[idx+1]) = pendulum_eom(theta[idx],dtheta[idx],tau[idx],l,k,m,g,dt)

# Plot results
plt.subplot(2,1,1)
plt.plot(np.arange(L+1)*dt,theta[:]*180/math.pi,label = "Theta")
plt.plot(np.arange(L+1)*dt,dtheta[:]*180/math.pi,label = "Angular Velocity")
plt.plot(np.arange(L)*dt,theta_ref*180/math.pi,label = "Theta Ref")
plt.xlabel("Time")
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(np.arange(L+1)*dt,tau,label ="Tau")
plt.xlabel("Time")
plt.legend()
plt.grid()
plt.show()