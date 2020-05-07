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
import scipy.spatial
from math import sin, cos, pi, sqrt

class CollisionChecker:
    def __init__(self, circle_offsets, circle_radii, weight):
        self._circle_offsets = circle_offsets
        self._circle_radii = circle_radii
        self._weight = weight

    # Takes in a set of paths and obstacles, and returns an array
    # of bools that says whether or not each path is collision free.
    # TODO Implement this function.
    # Input: paths - a list of paths, each path of the form [[x1, x2, ...], [y1, y2, ...], [theta1, theta2, ...]].
    #        obstacles - a list of obstacles, each obstacle represented by a list of occupied points of the form [[x1, y1], [x2, y2], ...]. 
    def collision_check(self, paths, obstacles):
        collision_check_array = np.zeros(len(paths), dtype=bool)
        for i in range(len(paths)):
            collision_free = True
            path = paths[i]

            # Iterate over the points in the path.
            for j in range(len(path[0])):
                # Compute the circle locations along this point in the path.
                # The circle offsets are given by self._circle_offsets.
                # The circle offsets need to placed at each point along the path,
                # with the offset rotated by the yaw of the vehicle.
                # Each path is of the form [[x_values], [y_values], [theta_values]],
                # where each of x_values, y_values, and theta_values are in sequential
                # order.

                # Thus, we need to compute:
                # circle_x = point_x + circle_offset*cos(yaw)
                # circle_y = point_y circle_offset*sin(yaw)
                # for each point along the path.
                # point_x is given by path[0][j], and point _y is given by path[1][j]. 
                # TODO YOUR CODE HERE. 
                circle_locations = np.zeros((len(self._circle_offsets), 2))                
                for c in range(len(self._circle_offsets)):
                    circle_locations[c, 0] = path[0][j] + self._circle_offsets[c] * np.cos(path[2][j])
                    circle_locations[c, 1] = path[1][j] + self._circle_offsets[c] * np.sin(path[2][j])

                # Assumes each obstacle is approximated by a collection of points
                # of the form [x, y].
                # Here, we will iterate through the obstacle points, and check if any of
                # the obstacle points lies within any of our circles.
                for k in range(len(obstacles)):
                    collision_dists = scipy.spatial.distance.cdist(obstacles[k], circle_locations)
                    collision_dists = np.subtract(collision_dists, self._circle_radii)
                    collision_free = collision_free and not np.any(collision_dists < 0)

                    if not collision_free:
                        break
                if not collision_free:
                    break

            collision_check_array[i] = collision_free

        return collision_check_array

    def logistic(self, x):
        '''
        A function that returns a value between 0 and 1 for x in the
        range[0, infinity] and -1 to 1 for x in the range[-infinity, infinity].

        Useful for cost functions.
        '''
        return 2.0 / (1 + np.exp(np.abs(x/10)))

    def calculate_path_score(self, path, goal_state):
        # So we have a path (multiple points) and a goal.
        # At this point the on;y thing i can think off is to compare how close you are to the GOAL at the end of the path.                          
        last_pt_idx = len(path[0]) - 1
        last_pt = np.array([path[0][last_pt_idx],path[1][last_pt_idx]])

        error = np.linalg.norm(np.subtract(last_pt, goal_state[:2]))
        
        score = self.logistic(error)

        return score
        
    def proximity_to_colliding_paths(self, path, colliding_path):
        assert len(path[0]) == len(colliding_path[0])
        error = np.linalg.norm(np.subtract(path, colliding_path)) 

        score = -1 * self.logistic(error)
        return score


    # Selects the best path in the path set, according to how closely
    # it follows the lane centerline, and how far away it is from other
    # paths that are in collision. 
    # Disqualifies paths that collide with obstacles from the selection
    # process.
    # collision_check_array contains True at index i if paths[i] is collision-free,
    # otherwise it contains False.
    # TODO Implement this function. 
    # Input: paths - a list of paths, each path of the form [[x1, x2, ...], [y1, y2, ...], [theta1, theta2, ...]].
    #        collision_check_array - a list of booleans that denote whether or not the corresponding path index is collision free.
    #        goal_state - a list denoting the centerline goal, in the form [x, y].
    def select_best_path_index(self, paths, collision_check_array, goal_state):
        best_index = None
        best_score = -float('Inf')
        for i in range(len(paths)):
            # Handle the case of collision-free paths.
            # TODO YOUR CODE HERE
            if collision_check_array[i]:
                # Compute the "distance from centerline" score.
                # The centerline goal is given by goal_state.
                # The exact choice of objective function is up to you.
                score = self.calculate_path_score(paths[i], goal_state)

                # Compute the "proximity to other colliding paths" score and
                # add it to the "distance from centerline" score.
                # The exact choice of objective function is up to you.
                for j in range(len(paths)):
                    if j == i:
                        continue
                    else:
                        if not collision_check_array[j]:
                            # TODO YOUR CODE HERE
                            score += self._weight * self.proximity_to_colliding_paths(paths[i], paths[j])

            # Handle the case of colliding paths.
            else:
                score = -float('Inf')
                
            if score > best_score:
                best_score = score
                best_index = i

        return best_index
