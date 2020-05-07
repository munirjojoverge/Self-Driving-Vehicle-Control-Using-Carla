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
from math import sin, cos, pi, sqrt
EPSILON = 0.001

class VelocityPlanner:
    def __init__(self, time_gap, a_max, slow_speed, stop_line_buffer):
        self._time_gap = time_gap
        self._a_max = a_max
        self._slow_speed = slow_speed
        self._stop_line_buffer = stop_line_buffer
        self._prev_trajectory = [[0.0, 0.0, 0.0]]

    # Computes an open loop speed estimate based on the previously planned
    # trajectory, and the timestep since the last planning cycle.
    def get_open_loop_speed(self, timestep):
        if len(self._prev_trajectory) == 1:
            return self._prev_trajectory[0][2] 
        
        # If simulation time step is zero, give the start of the trajectory as the
        # open loop estimate.
        if timestep < 1e-4:
            return self._prev_trajectory[0][2]

        for i in range(len(self._prev_trajectory)-1):
            distance_step = np.linalg.norm(np.subtract(self._prev_trajectory[i+1][0:2], self._prev_trajectory[i][0:2]))
            velocity = self._prev_trajectory[i][2]
            if velocity < EPSILON:
                time_delta = np.inf
            else:
                time_delta = distance_step / velocity
           
            # If time_delta exceeds the remaining time in our simulation timestep, 
            # interpolate between the velocity of the current step and the velocity
            # of the next step to estimate the open loop velocity.
            if time_delta > timestep:
                v1 = self._prev_trajectory[i][2]
                v2 = self._prev_trajectory[i+1][2]
                v_delta = v2 - v1
                interpolation_ratio = timestep / time_delta
                return v1 + interpolation_ratio * v_delta

            # Otherwise, keep checking.
            else:
                timestep -= time_delta

        # Simulation time step exceeded the length of the path, which means we have likely
        # stopped. Return the end velocity of the trajectory.
        return self._prev_trajectory[-1][2]

    # Takes a path, and computes a velocity profile to our desired speed.
    # decelerate_to_stop denotes whether or not we need to decelerate to a stop line,
    # follow_lead_vehicle denotes whether or not we need to follow a lead vehicle, with
    # state given by lead_car_state.
    # The order of precedence for handling these cases is stop sign handling, lead vehicle
    # handling, then nominal lane maintenance. In a real velocity planner you would need
    # to handle the coupling between these states, but for simplicity this project can be
    # implemented by isolating each case.
    # For all profiles, the required acceleration is given by self._a_max.
    # Recall that the path is of the form [x_points, y_points, t_points].
    def compute_velocity_profile(self, path, desired_speed, ego_state, closed_loop_speed, decelerate_to_stop, lead_car_state, follow_lead_vehicle):
        profile = []
        start_speed = ego_state[3]
        # Generate a trapezoidal profile to decelerate to stop.
        if decelerate_to_stop:
            profile = self.decelerate_profile(path, start_speed)

        # If we need to follow the lead vehicle, make sure we decelerate to its speed by
        # the time we reach the time gap point.
        elif follow_lead_vehicle:
            profile = self.follow_profile(path, start_speed, desired_speed, lead_car_state)

        # Otherwise, compute the profile to reach our desired speed.
        else:
            profile = self.nominal_profile(path, start_speed, desired_speed)

        # Interpolate between the zeroth state and the first state.
        # This prevents the myopic controller from getting stuck at the zeroth state.
        if len(profile) > 1:
            interpolated_state = [(profile[1][0] - profile[0][0]) * 0.1 + profile[0][0], (profile[1][1] - profile[0][1]) * 0.1 + profile[0][1], (profile[1][2] - profile[0][2]) * 0.1 + profile[0][2]]
            del profile[0]
            profile.insert(0, interpolated_state)

        # Save the planned profile for open loop speed estimation.
        self._prev_trajectory = profile

        return profile

    # Computes a profile for decelerating to stop.
    def decelerate_profile(self, path, start_speed): 
        profile = []
        slow_speed = self._slow_speed
        stop_line_buffer = self._stop_line_buffer

        # Using d = (v_f^2 - v_i^2) / (2 * a)
        brake_distance = calc_distance(slow_speed, 0, -self._a_max)
        decel_distance = calc_distance(start_speed, slow_speed, -self._a_max)

        path_length = 0.0
        for i in range(len(path[0])-1):
            path_length += np.linalg.norm([path[0][i+1] - path[0][i], path[1][i+1] - path[1][i]])

        stop_index = len(path[0]) - 1
        temp_dist = 0.0
        # Compute the index at which we should stop.
        while (stop_index > 0) and (temp_dist < stop_line_buffer):
            temp_dist += np.linalg.norm([path[0][stop_index] - path[0][stop_index-1], path[1][stop_index] - path[1][stop_index-1]])
            stop_index -= 1

        # If the brake distance exceeds the length of the path, then we cannot perform
        # a smooth deceleration and require a harder deceleration. Build the path up
        # in reverse to ensure we reach zero speed at the required time.
        if brake_distance + decel_distance + stop_line_buffer > path_length:
            speeds = []
            vf = 0.0
            for i in reversed(range(stop_index, len(path[0]))):
                speeds.insert(0, 0.0)
            for i in reversed(range(stop_index)):
                dist = np.linalg.norm([path[0][i+1] - path[0][i], path[1][i+1] - path[1][i]])
                vi = calc_final_speed(vf, -self._a_max, dist)
                if vi > start_speed:
                    vi = start_speed

                speeds.insert(0, vi)
                vf = vi

            for i in range(len(speeds)):
                profile.append([path[0][i], path[1][i], speeds[i]])
            
        else:
            brake_index = stop_index 
            temp_dist = 0.0
            # Compute the index at which to start braking down to zero.
            while (brake_index > 0) and (temp_dist < brake_distance):
                temp_dist += np.linalg.norm([path[0][brake_index] - path[0][brake_index-1], path[1][brake_index] - path[1][brake_index-1]])
                brake_index -= 1

            # Compute the index to stop decelerating to the slow speed.
            decel_index = 0
            temp_dist = 0.0
            while (decel_index < brake_index) and (temp_dist < decel_distance):
                temp_dist += np.linalg.norm([path[0][decel_index+1] - path[0][decel_index], path[1][decel_index+1] - path[1][decel_index]])
                decel_index += 1

            vi = start_speed
            for i in range(decel_index): 
                dist = np.linalg.norm([path[0][i+1] - path[0][i], path[1][i+1] - path[1][i]])
                vf = calc_final_speed(vi, -self._a_max, dist)
                if vf < slow_speed:
                    vf = slow_speed

                profile.append([path[0][i], path[1][i], vi])
                vi = vf

            for i in range(decel_index, brake_index):
                profile.append([path[0][i], path[1][i], vi])
                
            for i in range(brake_index, stop_index):
                dist = np.linalg.norm([path[0][i+1] - path[0][i], path[1][i+1] - path[1][i]])
                vf = calc_final_speed(vi, -self._a_max, dist)
                profile.append([path[0][i], path[1][i], vi])
                vi = vf

            for i in range(stop_index, len(path[0])):
                profile.append([path[0][i], path[1][i], 0.0])

        return profile

    # Computes a profile for following a lead vehicle..
    def follow_profile(self, path, start_speed, desired_speed, lead_car_state):
        profile = []
        # Find the closest point to the lead vehicle on our planned path.
        min_index = len(path[0]) - 1
        min_dist = float('Inf')
        for i in range(len(path)):
            dist = np.linalg.norm([path[0][i] - lead_car_state[0], path[1][i] - lead_car_state[1]])
            if dist < min_dist:
                min_dist = dist
                min_index = i

        # Compute the time gap point, assuming our velocity is held constant at the 
        # minimum of the desired speed and the ego vehicle's velocity,
        # from the closest point to the lead vehicle on our planned path.
        desired_speed = min(lead_car_state[2], desired_speed)
        ramp_end_index = min_index
        distance = min_dist
        distance_gap = desired_speed * self._time_gap
        while (ramp_end_index > 0) and (distance > distance_gap):
            distance += np.linalg.norm([path[0][ramp_end_index] - path[0][ramp_end_index-1], path[1][ramp_end_index] - path[1][ramp_end_index-1]])
            ramp_end_index -= 1

        # We now need to reach the ego vehicle's speed by the time we reach the time gap
        # point, ramp_end_index, which therefore is the end of our ramp velocity profile.
        if desired_speed < start_speed:
            decel_distance = calc_distance(start_speed, desired_speed, -self._a_max)
        else:
            decel_distance = calc_distance(start_speed, desired_speed, self._a_max)

        vi = start_speed
        for i in range(ramp_end_index + 1):
            dist = np.linalg.norm([path[0][i+1] - path[0][i], path[1][i+1] - path[1][i]])
            if desired_speed < start_speed:
                vf = calc_final_speed(vi, -self._a_max, dist)
            else:
                vf = calc_final_speed(vi, self._a_max, dist)

            profile.append([path[0][i], path[1][i], vi])
            vi = vf

        # Once we hit the time gap point, we need to be at the desired speed.
        # If we can't get there using a_max, do an abrupt change in the profile
        # to use the controller to decelerate more quickly.
        for i in range(ramp_end_index + 1, len(path[0])):
            profile.append([path[0][i], path[1][i], desired_speed])

        return profile

    # Computes a profile for nominal speed tracking.
    def nominal_profile(self, path, start_speed, desired_speed):
        profile = []
        if desired_speed < start_speed:
            accel_distance = calc_distance(start_speed, desired_speed, -self._a_max)
        else:
            accel_distance = calc_distance(start_speed, desired_speed, self._a_max)

        ramp_end_index = 0
        distance = 0.0
        while (ramp_end_index < len(path[0])-1) and (distance < accel_distance):
            distance += np.linalg.norm([path[0][ramp_end_index+1] - path[0][ramp_end_index], path[1][ramp_end_index+1] - path[1][ramp_end_index]])
            ramp_end_index += 1

        vi = start_speed
        for i in range(ramp_end_index):
            dist = np.linalg.norm([path[0][i+1] - path[0][i], path[1][i+1] - path[1][i]])
            if desired_speed < start_speed:
                vf = calc_final_speed(vi, -self._a_max, dist)
                if vf < desired_speed:
                    vf = desired_speed
            else:
                vf = calc_final_speed(vi, self._a_max, dist)
                if vf > desired_speed:
                    vf = desired_speed

            profile.append([path[0][i], path[1][i], vi])
            vi = vf

        for i in range(ramp_end_index+1, len(path[0])):
            profile.append([path[0][i], path[1][i], desired_speed])

        return profile


# Using d = (v_f^2 - v_i^2) / (2 * a), compute the distance
# required for a given acceleration/deceleration.
# TODO Implement this function.
# Inputs: v_i - the initial speed in m/s.
#         v_f - the final speed in m/s.
#         a - the acceleration in m/s^2.
def calc_distance(v_i, v_f, a):
    if np.abs(a) < EPSILON:
        d = np.inf
    else:
        d = (v_f**2 - v_i**2) / (2 * a)
    return d

# Using v_f = sqrt(v_i^2 + 2ad), compute the final speed for a given
# acceleration across a given distance, with initial speed v_i.
# Make sure to check the discriminant of the radical. If it is negative,
# return zero as the final speed.
# TODO Implement this function.
# Inputs: v_i - the initial speed in m/s.
#         v_f - the ginal speed in m/s.
#         a - the acceleration in m/s^2.
def calc_final_speed(v_i, a, d): 
    disc = v_i**2 + 2*a*d
    if disc <= 0:
        v_f = 0
    else:
        v_f = np.sqrt(v_i**2 + 2*a*d)
    return v_f

