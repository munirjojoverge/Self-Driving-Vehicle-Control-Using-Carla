###########################################################
#       Self-Driving Car Specialization - Coursera
#           Motion Planning for Self-Driving Cars
#           Created/Modified on: May 6, 2019
#               Author: Munir Jojo-Verge
###########################################################

#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('kp', 1.90)
        self.vars.create_var('ki', 0.001)
        self.vars.create_var('i_reset_tolerance', 0.05)
        self.vars.create_var('kd', 0.80)
        self.vars.create_var('kp_heading', 3.00)
        self.vars.create_var('k_speed_crosstrack', 3.00)
        self.vars.create_var('x_prev', 0.0)
        self.vars.create_var('y_prev', 0.0)
        self.vars.create_var('yaw_prev', 0.0)
        self.vars.create_var('v_prev', 0.0)
        self.vars.create_var('t_prev', 0.0)
        self.vars.create_var('v_error', 0.0)
        self.vars.create_var('v_error_prev', 0.0)
        self.vars.create_var('v_error_integral', 0.0)
        self.vars.create_var('trajectory_heading_prev', 0.0)
        
        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    v_desired       : Current desired speed (meters per second)
                    t               : Current time (seconds)
                    waypoints       : Current waypoints to track
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """

            self.vars.v_error           = v_desired - v
            self.vars.v_error_integral += self.vars.v_error
            v_error_rate_of_change      = (self.vars.v_error - self.vars.v_error_prev) /\
                                          (t - self.vars.t_prev)
            # reset integrator if close enough to desired velocity
            if (np.absolute(self.vars.v_error) < self.vars.i_reset_tolerance):
                self.vars.v_error_integral = 0

            throttle_output = self.vars.kp * self.vars.v_error +\
                              self.vars.ki * self.vars.v_error_integral +\
                              self.vars.kd * v_error_rate_of_change

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """

            # Find cross track error (point with closest distance)
            crosstrack_error = float("inf")
            crosstrack_vector = np.array([float("inf"), float("inf")])
            ce_idx = 0

            for wp_idx in range(len(waypoints)):
                vect_xy_to_wp = np.array([waypoints[wp_idx][0] - x, 
                                          waypoints[wp_idx][1] - y])
                dist_xy_to_wp = np.linalg.norm(vect_xy_to_wp)
                if (dist_xy_to_wp <= crosstrack_error):
                    crosstrack_error = dist_xy_to_wp
                    crosstrack_vector = vect_xy_to_wp
                    closest_idx = wp_idx

            # Find the waypoint 3.0m ahead of the car to track.
            # ce_idx can't be len(waypoitns)-1 due to heading calculation
            # below causing poor results.
            temp_dist = 0.0
            ce_idx = closest_idx
            for i in range(closest_idx+1, len(waypoints)-1):
                temp_dist += np.linalg.norm([waypoints[i][0] - waypoints[i-1][0], \
                    waypoints[i][1] - waypoints[i-1][1]])
                ce_idx = i

                if temp_dist > 4.0:
                    break

            # Compute the sign of the crosstrack error
            crosstrack_heading = np.arctan2(crosstrack_vector[1], 
                                            crosstrack_vector[0])
            crosstrack_heading_error = crosstrack_heading - yaw
            crosstrack_heading_error = \
                    (crosstrack_heading_error + self._pi) % \
                    self._2pi - self._pi

            crosstrack_sign = np.sign(crosstrack_heading_error)
    
            # Compute heading relative to trajectory (heading error)
            # First ensure that we are not at the last index. If we are,
            # flip back to the first index (loop the waypoints)
            if ce_idx < len(waypoints)-1:
                vect_wp0_to_wp1 = np.array(
                        [waypoints[ce_idx+1][0] - waypoints[ce_idx][0],
                         waypoints[ce_idx+1][1] - waypoints[ce_idx][1]])
                trajectory_heading = np.arctan2(vect_wp0_to_wp1[1], 
                                                vect_wp0_to_wp1[0])
            else:
                vect_wp0_to_wp1 = np.array(
                        [waypoints[0][0] - waypoints[-1][0],
                         waypoints[0][1] - waypoints[-1][1]])
                trajectory_heading = np.arctan2(vect_wp0_to_wp1[1], 
                                                vect_wp0_to_wp1[0])

            self.vars.trajectory_heading_prev = trajectory_heading

            heading_error = trajectory_heading - yaw
            heading_error = \
                    (heading_error + self._pi) % self._2pi - self._pi

            # Compute steering command based on error
            steer_output = heading_error + \
                    np.arctan(self.vars.kp_heading * \
                              crosstrack_sign * \
                              crosstrack_error / \
                              (v + self.vars.k_speed_crosstrack))

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.x_prev       = x
        self.vars.y_prev       = y
        self.vars.yaw_prev     = yaw
        self.vars.v_prev       = v
        self.vars.v_error_prev = self.vars.v_error
        self.vars.t_prev       = t
        
