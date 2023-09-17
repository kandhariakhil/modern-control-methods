# Autonomous Driving with MPC using Tire and Drive Train Dynamics

import casadi as ca
import numpy as np
import time
from math import *
import matplotlib.pyplot as plt

from importParameters import importParameters

def calFrontForce(lf, B, C, D, omega, vx, vy, delta):
    alpha = -ca.arctan2(omega * lf + vy, vx) + delta
    frontForce = D * ca.sin(C * ca.arctan(B * alpha))
    
    return frontForce

def calRearForce(lr, B, C, D, omega, vx, vy):
    alpha = ca.arctan2(omega * lr - vy, vx)
    rearForce = D * ca.sin(C * ca.arctan(B * alpha))
    
    return rearForce

def calLongitudinalForce():
    pass

x_init = 0.0
y_init = 0.0
phi_init = 0.0
vx_init = 0.0
vy_init = 0.0
omega = 0.0

x_target = 30.0
y_target = 30.0
phi_target = pi/2
vx_target = 0.0
vy_target = 0.0
omega_target = 0.0

parameters = importParameters()

dt = 0.25
N = 10
sim_time = 100

mp = parameters.getModelParameters()
lim = parameters.getLimits()

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

front_force = calFrontForce(mp['lf'], mp['Bf'], mp['Cf'], mp['Df'], omega, vx, vy, delta)
front_force_func = ca.Function('front_force_func', [omega, vx, vy, delta], [front_force])

rear_force = calRearForce(mp['lf'], mp['Bf'], mp['Cf'], mp['Df'], omega, vx, vy)
rear_force_func = ca.Function('rear_force_func', [omega, vx, vy], [rear_force])
