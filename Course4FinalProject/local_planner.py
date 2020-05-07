###########################################################
#       Self-Driving Car Specialization - Coursera
#           Motion Planning for Self-Driving Cars
#           Created/Modified on: May 6, 2019
#               Author: Munir Jojo-Verge
###########################################################

#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Author: Ryan De Iaco
# Date: October 29, 2018

import numpy as np
import copy
import path_optimizer
import collision_checker
import velocity_planner
from math import sin, cos, pi, sqrt

class LocalPlanner:
    def __init__(self, num_paths, path_offset, circle_offsets, circle_radii, path_select_weight, time_gap, a_max, slow_speed, stop_line_buffer):
        self._num_paths = num_paths
        self._path_offset = path_offset
        self._path_optimizer = path_optimizer.PathOptimizer()
        self._collision_checker = collision_checker.CollisionChecker(circle_offsets, circle_radii, path_select_weight)
        self._velocity_planner = velocity_planner.VelocityPlanner(time_gap, a_max, slow_speed, stop_line_buffer)
        self._prev_best_path = []

    # Computes the goal state set from a given goal position. This is done by
    # laterally sampling offsets from the goal location along the direction
    # perpendicular to the goal yaw of the ego vehicle.
    # TODO Implement this function.
    # Input: goal_index - the index in the waypoints that corresponds to the goal state.
    #        goal_state - the desired final state in the global frame, of the form [x, y, velocity].
    #        waypoints - the list of waypoints to track in the global frame, of the form [[x1, y1], [x2, y2], ...]. 
    #        ego_state - the current ego vehicle state in the global frame, of the form [x, y, theta, velocity]        
    def get_goal_state_set(self, goal_index, goal_state, waypoints, ego_state):
        # Compute the final heading based on the next index.
        # If the goal index is the last in the set of waypoints, use
        # the previous index instead.
        # To do this, compute the delta_x and delta_y values between
        # consecutive waypoints, then use the np.arctan2() function.
        # TODO YOUR CODE HERE.
        if goal_index == len(waypoints)-1:
            p1 = goal_index - 1
            p2 = goal_index
        elif goal_index >= 0:
            p1 = goal_index
            p2 = goal_index + 1

        delta = np.subtract(waypoints[p2],waypoints[p1])
        heading = np.arctan2(delta[1],delta[0])

        # Compute the center goal state in the local frame using 
        # the ego state. The following code will transform the input
        # goal state to the ego vehicle's local frame.
        # The goal state will be of the form (x, y, v).
        goal_pos_local = copy.copy(goal_state)

        # Translate so the ego state is at the origin in the new frame.
        # This is done by subtracting the ego_state from the goal_state_local.
        # TODO YOUR CODE HERE.
        goal_pos_local[0] -= ego_state[0] 
        goal_pos_local[1] -= ego_state[1]

        # Rotate such that the ego state has zero heading in the new frame.
        # Recall that the general rotation matrix is [cos(theta) -sin(theta)
        #                                             sin(theta)  cos(theta)]
        # and that we are rotating by -ego_state[2] to ensure the ego vehicle's
        # current yaw corresponds to theta = 0 in the new local frame.
        # TODO YOUR CODE HERE
        theta = -ego_state[2]
        rotation_matrix = np.array([[cos(theta), -sin(theta)],
                                    [sin(theta),cos(theta)]])
        goal = rotation_matrix@np.transpose(goal_pos_local[0:2])

        # Compute the goal yaw in the local frame by subtracting off the 
        # current ego yaw from the heading variable.
        # TODO YOUR CODE HERE
        goal_t = heading + theta

        # Velocity is preserved after the transformation.
        goal_v = goal_state[2]

        # Keep the goal heading within [-pi, pi] so the optimizer behaves well.
        if goal_t > pi:
            goal_t -= 2*pi
        elif goal_t < -pi:
            goal_t += 2*pi

        # Compute and apply the offset for each path such that
        # all of the paths have the same heading of the goal state, 
        # but are laterally offset with respect to the goal heading.
        goal_state_set = []        

        for i in range(self._num_paths):
            offset = (i - self._num_paths // 2) * self._path_offset
            # Compute the projection of the lateral offset along the x
            # and y axis. To do this, multiply the offset by cos(goal_theta + pi/2)
            # and sin(goal_theta + pi/2), respectively.
            # TODO YOUR CODE HERE
            x_offset = offset * cos(goal_t + pi/2)
            y_offset = offset * sin(goal_t + pi/2)

            goal_state_set.append([goal[0] + x_offset, goal[1] + y_offset, goal_t, goal_v])
           
        return goal_state_set  
              
    # Plans the path set using polynomial spiral optimization to
    # each of the goal states.
    def plan_paths(self, goal_state_set):
        paths = []
        path_validity = []
        for goal_state in goal_state_set:
            path = self._path_optimizer.optimize_spiral(goal_state[0], goal_state[1], goal_state[2])
            if np.linalg.norm([path[0][-1] - goal_state[0], path[1][-1] - goal_state[1], path[2][-1] - goal_state[2]]) > 0.1:
                path_validity.append(False)
            else:
                paths.append(path)
                path_validity.append(True)


        return paths, path_validity

def transform_paths(paths, ego_state):
    transformed_paths = []
    for path in paths:
        x_transformed = []
        y_transformed = []
        t_transformed = []

        for i in range(len(path[0])):
            x_transformed.append(ego_state[0] + path[0][i]*cos(ego_state[2]) - path[1][i]*sin(ego_state[2]))
            y_transformed.append(ego_state[1] + path[0][i]*sin(ego_state[2]) + path[1][i]*cos(ego_state[2]))
            t_transformed.append(path[2][i] + ego_state[2])

        transformed_paths.append([x_transformed, y_transformed, t_transformed])

    return transformed_paths
