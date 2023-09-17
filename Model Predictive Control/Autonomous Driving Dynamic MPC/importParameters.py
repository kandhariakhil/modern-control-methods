import numpy as np
from math import *

class importParameters:
    def __init__(self):
        self.m = 1573
        self.Iz = 2873
        self.lf = 1.35
        self.lr = 1.35

        self.wf = self.lr/(self.lf+self.lr)
        self.wr = self.lf/(self.lf+self.lr)

        self.cm1 = 17303
        self.cm2 = 175
        self.cr = 120
        self.cd = 0.5*1.225*0.35*2.5

        self.Br = 13
        self.Cr = 2
        self.Dr = self.wf*self.m*9.81*1.2

        self.Bf = 13
        self.Cf = 2
        self.Df = self.wr*self.m*9.81*1.2

        self.L = 5
        self.W = 2.5

        self.x_max = 40
        self.x_min = -40

        self.y_max = 40
        self.y_min = -40

        self.phi_max = np.inf
        self.phi_min = -np.inf

        self.vx_max = 25
        self.vx_min = -4

        self.vy_max = 5
        self.vy_min = -5

        self.omega_max = radians(10)
        self.omega_min = -radians(10)

        self.d_max = 1
        self.d_min = -0.1

        self.delta_max = radians(30)
        self.delta_min = radians(-30)

    
    def getModelParameters(self):
        return {
            'm': self.m,
            'Iz': self.Iz,
            'lf': self.lf,
            'lr': self.lr,
            'wf': self.wf,
            'wr': self.wr,
            'cm1': self.cm1,
            'cm2': self.cm2,
            'cr': self.cr,
            'cd': self.cd,
            'Br': self.Br,
            'Cr': self.Cr,
            'Dr': self.Dr,
            'Bf': self.Bf,
            'Cf': self.Cf,
            'Df': self.Df,
            'L': self.L,
            'W': self.W}
    
    def getLimits(self):
        return {
        'x_max': self.x_max,
        'x_min': self.x_min,
        'y_max': self.y_max,
        'y_min': self.y_min,
        'phi_max': self.phi_max,
        'phi_min': self.phi_min,
        'vx_max': self.vx_max,
        'vx_min': self.vx_min,
        'vy_max': self.vy_max,
        'vy_min': self.vy_min,
        'omega_max': self.omega_max,
        'omega_min': self.omega_min,
        'd_max': self.d_max,
        'd_min': self.d_min,
        'delta_max': self.delta_max,
        'delta_min': self.delta_min
        }