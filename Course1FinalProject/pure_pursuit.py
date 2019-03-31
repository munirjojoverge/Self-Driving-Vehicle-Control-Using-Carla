#**********************************************
#       Self-Driving Car - Coursera
#        Created on: March 23, 2019
#         Author: Munir Jojo-Verge
#*********************************************

import numpy as np

class PP(object):
    def __init__(self,L, k, k_Ld):
        self.L = L
        self.k = k
        self.k_Ld = k_Ld

    def update(self, coeffs, velocity):                
        eps = 0.001
        if velocity < eps:            
            steering_angle = 0.0
        else:
            # For Pure Pursuit if we look ahead a distance Ld = nnnn then the cte changes            
            Ld_x = self.k_Ld * velocity            
            cte = np.polyval(coeffs, Ld_x)            
            alpha = np.arctan(cte/Ld_x)
            sin_alpha = np.sin(alpha)
            steering_angle = np.arctan((2*self.L* sin_alpha) / (self.k * velocity))
        
        return steering_angle

