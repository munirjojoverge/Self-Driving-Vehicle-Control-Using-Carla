#**********************************************
#       Self-Driving Car - Coursera
#        Created on: March 23, 2019
#         Author: Munir Jojo-Verge
#*********************************************

import numpy as np
import ipopt
import numdifftools as nd
from scipy.sparse import coo_matrix

# Set the timestep length and duration
N = 2 # prediction Horizon
dt = 0.08
T = N*dt # This is the Prediction Horizon in seconds. 


# The solver takes all the state variables and actuator variables in a single vector. 
# Here, we establish when one variable starts and another ends to be able to address its indexes in an easy way.

# State
x_start = 0
y_start = x_start + N
yaw_start = y_start + N
v_start = yaw_start + N
cte_start = v_start + N
yaw_err_start = cte_start + N
speed_err_start = yaw_err_start + N

# Outputs (First)
steer_start = speed_err_start + N - 1
throttle_start = steer_start + N - 1
brake_start = throttle_start + N - 1


# define the WEIGTHS that we will use to quantify how "costly" (bad) are each component of the COST function
# Basically HOW important is each element of the COST function: For instance, it's very important that cte remains close to 0
# but also it's veru important to make sure that the changes in commands (steering and throattle) are smooth.
# For more explanations look below on the COST function construntion

W_cte          = 10
W_yaw_err      = 10
W_vel_err      = 99
W_steer_use    = 0
W_throttle_use = 0
W_brake_use    = 0
W_dSteer       = 999 # Differential Steering
W_dThrottle    = 0   # Differential Throttle
W_dBrake       = 0   # Differential Brake


# The constructor of the ipopt.problem class requires:
# n: the number of variables in the problem, 
# m: the number of constraints in the problem, 
# lb and ub: lower and upper bounds on the variables, and 
# cl and cu: lower and upper bounds of the constraints. 
# problem_obj is an object whose methods implement the objective, gradient, constraints, jacobian, and hessian of the problem:
class MPC(object):
    def __init__(self):
        self.speed_ref = 0
    
    def Solve(self, state, coeffs):
        ok = True
        
        num_state_variables = len(state)
        num_outputs = 3        

        # Set the number of model variables (includes both states and inputs).
        # # For example: If the state is a 4 element vector, the actuators is a 2
        # # element vector and there are 10 timesteps. The number of variables is:
        # #
        # # 4 * 10 + 2 * 9
        # # In "N" timesteps => "N - 1" actuations

        n_vars = N * num_state_variables + (N - 1) * num_outputs

        # Set the number of constraints over the State Variables.
        n_constraints = N * (num_state_variables) # 

        # For clarity
        x = state[0] # Always 0 since we moved to the Car Ref System
        y = state[1] # Always 0 since we moved to the Car Ref System
        yaw = state[2] # Always 0 since we moved to the Car Ref System
        v = state[3]
        cte = state[4]
        yaw_err = state[5]
        speed_err = state[6]
        self.speed_ref = speed_err + v

        # Initial value of the independent variables.
        # SHOULD BE 0 besides initial state.
        # Initial State:
        # Set the initial variable values
        vars = np.zeros(n_vars)
 
        vars[x_start] = x
        vars[y_start] = y
        vars[yaw_start] = yaw
        vars[v_start] = v
        vars[cte_start] = cte
        vars[yaw_err_start] = yaw_err
        vars[speed_err_start] = speed_err

        
        vars_lowerbound = np.zeros(n_vars)
        vars_upperbound = np.zeros(n_vars)

        # Set lower and upper limits for variables.
        # Set all non-actuators (x,y,yaw,v,cte,yaw_err) upper and lowerlimits to the max negative and positive values.
        # We can refine these limits but for simplicity we do this for now.
        for i in range(0, steer_start):
            vars_lowerbound[i] = -1.0e19
            vars_upperbound[i] = 1.0e19

        # The upper and lower limits for Steering is -1.22 to 1.22 Radians
        for i in range(steer_start, throttle_start):
            vars_lowerbound[i] = -1.22
            vars_upperbound[i] = 1.22

        # The upper and lower limits Throttle is 0 to 1. (%)
        for i in range(throttle_start, brake_start):
            vars_lowerbound[i] = 0
            vars_upperbound[i] = 1.0
        
        # The upper and lower limits for Brake ARE 0 to 1.(%)
        for i in range(brake_start, n_vars):
            vars_lowerbound[i] = 0
            vars_upperbound[i] = 1.0

        # Lower and upper limits for the constraints
        # Should be 0 besides initial state.
        constraints_lowerbound = np.zeros(n_constraints)
        constraints_upperbound = np.zeros(n_constraints)
                
        # Initial state should have same upper and lower bounds since it will NOT change
        constraints_lowerbound[x_start] = x
        constraints_lowerbound[y_start] = y
        constraints_lowerbound[yaw_start] = yaw
        constraints_lowerbound[v_start] = v
        constraints_lowerbound[cte_start] = cte
        constraints_lowerbound[yaw_err_start] = yaw_err
        constraints_lowerbound[speed_err_start] = speed_err

        constraints_upperbound[x_start] = x
        constraints_upperbound[y_start] = y
        constraints_upperbound[yaw_start] = yaw
        constraints_upperbound[v_start] = v
        constraints_upperbound[cte_start] = cte
        constraints_upperbound[yaw_err_start] = yaw_err
        constraints_upperbound[speed_err_start] = speed_err

        # object that computes objective
        #FG_eval fg_eval(coeffs)

        # The constructor of the ipopt.problem class requires:
        # n: the number of variables in the problem, 
        # m: the number of constraints in the problem, 
        # lb and ub: lower and upper bounds on the variables, and 
        # cl and cu: lower and upper bounds of the constraints. 
        # problem_obj is an object whose methods implement the objective, gradient, constraints, jacobian, and hessian of the problem:        
        problem = lon_lat_vehicle_control_problem(coeffs, n_constraints, num_state_variables,self.speed_ref)
        nlp = ipopt.problem(            
            n=n_vars,
            m=n_constraints,
            problem_obj=problem,
            lb=vars_lowerbound,
            ub=vars_upperbound,
            cl=constraints_lowerbound,
            cu=constraints_upperbound
            )        

        #nlp.addOption('hessian_approximation','limited-memory')
        #
        # NOTE: You don't have to worry about these options
        #
        # options for IPOPT solver
        options = ""
        # Uncomment this if you'd like more print information
        options += "Integer print_level  0\n"
        # NOTE: Setting sparse to true allows the solver to take advantage
        # of sparse routines, this makes the computation MUCH FASTER. If you
        # can uncomment 1 of these and see if it makes a difference or not but
        # if you uncomment both the computation time should go up in orders of
        # magnitude.
        options += "Sparse  true        forward\n"
        options += "Sparse  true        reverse\n"
        # NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
        # Change this as you see fit.
        options += "Numeric max_cpu_time          0.5\n"        

        solution, info = nlp.solve(vars)

        # TODO: Return the first actuator values. The variables can be accessed with
        # `solution.x[i]`.
        #
        # {...} is shorthand for creating a vector, so auto x_t = {1.0,2.0}
        # creates a 2 element double vector.        
        steer_cmd = solution[steer_start]
        throttle_cmd = solution[throttle_start]
        brake_cmd = solution[brake_start]

        # return np.array([solution])        
        return [steer_cmd, throttle_cmd, brake_cmd]



class lon_lat_vehicle_control_problem(object):
    
    def __init__(self, coeffs, n_constraints, num_state_variables, speed_ref):        
        self.coeffs = coeffs
        self.n_constraints = n_constraints
        self.num_state_variables = num_state_variables        
        self.speed_ref = speed_ref

    def objective(self, vars): ## THIS IS THE COST FUNCTION. OUR OBJECTIVE IS TO MINIMIZE THIS EQ
        #
        # The callback for calculating the objective
        #
        # return vars[0] * vars[3] * np.sum(vars[0:3]) + vars[2]

        # The cost is stored is the first element of `fg`.
        # Any additions to the cost should be added to `cost`.
        cost = 0

        # The part of the cost based on the reference state.
        for t in range(0, N):
            cost += W_cte * np.power(vars[cte_start + t], 2)
            cost += W_yaw_err * np.power(vars[yaw_err_start + t], 2)
            cost += W_vel_err * np.power(vars[speed_err_start + t], 2)
        
        # Minimize the use of actuators.	
        for t in range(0, N-1):
            cost += W_steer_use * np.power(vars[steer_start + t], 2)
            cost += W_throttle_use * np.power(vars[throttle_start + t], 2)
            cost += W_brake_use * np.power(vars[brake_start + t], 2)
                        
        
        # Minimize the value gap between sequential actuations. (This is actually to guarantee a min "snap" trajectory)
        # We could try to use even deeper derivatives (min Jerk trajectories), but for now we can see how this performs.
        for t in range(0, N-2):
            cost += W_dSteer * np.power(vars[steer_start + t + 1] - vars[steer_start + t], 2)
            cost += W_dThrottle * np.power(vars[throttle_start + t + 1] - vars[throttle_start + t], 2)
            cost += W_dBrake * np.power(vars[brake_start + t + 1] - vars[brake_start + t], 2)
        
        return cost
        
        
    def gradient(self, vars):
        #
        # The callback for calculating the gradient of the Objective Function
        #       
        grad_func = nd.Gradient(self.objective)           
        return grad_func(vars)
        
        # grad = np.ones(len(vars))
        # for t in range(0, N):
        #     grad[cte_start + t] = W_cte * 2 * vars[cte_start + t]
        #     grad[yaw_err_start + t] = W_yaw_err * 2 * vars[yaw_err_start + t]
        #     grad[speed_err_start + t] = W_vel_err * 2 * vars[speed_err_start + t]
        # return grad


    def constraints(self, vars):
        #
        # The callback for calculating the constraints
        #
        #return np.array((np.prod(x), np.dot(x, x)))
        # The rest of the constraints
        # Before we start defining the constraints, we need to recall that, as a discrete model, all the constraints
        # at time "t+1" depend on the values at "t" AND also that for simplicity we put all the constraints of the form
        # XX(t+1) = F(X(t))
        # as
        # F(X(t)) - XX(t+1) = 0 or XX(t+1) - F(X(t)) = 0
        # Therefore, we will start collecting all the actual (t+1) and previous (t) values       
        a = self.coeffs[3]
        b = self.coeffs[2]
        c = self.coeffs[1]
        Lf = 2.67 # this is the length from the CoG (Our reference) to the FRONT CENTER AXLE
        
        constraints_ = np.zeros(self.n_constraints)

        # Constraints at time 0
        constraints_[x_start]         = vars[x_start]
        constraints_[y_start]         = vars[y_start]
        constraints_[yaw_start]       = vars[yaw_start]
        constraints_[v_start]         = vars[v_start]
        constraints_[cte_start]       = vars[cte_start]
        constraints_[yaw_err_start]   = vars[yaw_err_start]
        constraints_[speed_err_start] = vars[speed_err_start]

        for t in range(1, N):
            # X
            x_t_1 = vars[x_start + t - 1]
            x_t = vars[x_start + t]

            # Y
            y_t_1 = vars[y_start + t - 1]
            y_t = vars[y_start + t]
            
            # YAW / HEADING
            yaw_t_1 = vars[yaw_start + t - 1]
            yaw_t = vars[yaw_start + t]
            
            # SPEED / VELOCITY MAGNITUDE            
            v_t_1 = vars[v_start + t - 1]
            v_t = vars[v_start + t]
            
            # CTE
            cte_t_1 = vars[cte_start + t - 1]
            cte_t = vars[cte_start + t]
            
            # YAW ERROR
            yaw_err_t_1 = vars[yaw_err_start + t - 1]
            yaw_err_t = vars[yaw_err_start + t]

            # SPEED ERROR
            #speed_err_t_1 = vars[speed_err_start + t - 1]
            speed_err_t = vars[speed_err_start + t]
            
            # we are just interested in getting the previous accel (throttle) and steering
            # a_t_1 = vars[throttle_start + t - 1]
            if t > 1:
                v_t_2 = vars[v_start + t - 2]
                a_t_1 = (v_t_1 - v_t_2)/dt
            else:
                a_t_1 = 0.0

            steer_t_1 = vars[steer_start + t - 1]
            
                    
            f_t_1 = self.coeffs[0] + c * x_t_1 + b * np.power(x_t_1, 2) + a * np.power(x_t_1, 3)
            psides_t_1 = np.arctan(c + 2 * b * x_t_1 + 3 * a * np.power(x_t_1, 2))

            # Now we are ready to Setup the rest of the model constraints
            
            # constraints_[x_start + t]         = -x_t + (x_t_1 + v_t_1 * np.cos(yaw_t_1) * dt)
            # constraints_[y_start + t]         = -y_t + (y_t_1 + v_t_1 * np.sin(yaw_t_1) * dt)
            # constraints_[yaw_start + t]       = -yaw_t + (yaw_t_1 + ((v_t_1 / Lf) * steer_t_1 * dt))
            # constraints_[v_start + t]         = -v_t + (v_t_1 + a_t_1 * dt)
            # constraints_[cte_start + t]       = -cte_t + ((f_t_1 - y_t_1) + (v_t_1 * np.sin(yaw_err_t_1) * dt))
            # constraints_[yaw_err_start + t]   = -yaw_err_t - ((yaw_t_1 - psides_t_1) + ((v_t_1/Lf) * steer_t_1 * dt))
            # constraints_[speed_err_start + t] = -speed_err_t  + (self.speed_ref - v_t)

        
        return constraints_


    def jacobian(self, vars):
        #
        # The callback for calculating the Jacobian
        #
        J_func = nd.Jacobian(self.constraints)
        return J_func(vars) 
        #J = np.ones(self.n_constraints)

    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #
        global hs
        hs = coo_matrix(np.tril(np.ones((self.num_state_variables, self.num_state_variables))))
        return (hs.col, hs.row)


    def hessian(self, vars, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian: • Hessian of the Lagrangian function
        #
        pass
        

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print ("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
