###########################################################
#       Self-Driving Car Specialization - Coursera
#           Motion Planning for Self-Driving Cars
#           Created/Modified on: May 6, 2019
#               Author: Munir Jojo-Verge
###########################################################

#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA waypoint follower assessment client script.

A controller assessment to follow a given trajectory, where the trajectory
can be defined using way-points.

STARTING in a moment...
"""
from __future__ import print_function
from __future__ import division

# System level imports
import sys
import os
import argparse
import logging
import time
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import controller2d
import configparser 
import local_planner
import behavioural_planner

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import live_plotter as lv   # Custom live plotting library
from carla            import sensor
from carla.client     import make_carla_client, VehicleControl
from carla.settings   import CarlaSettings
from carla.tcp        import TCPConnectionError
from carla.controller import utils

"""
Configurable params
"""
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations
WAIT_TIME_BEFORE_START = 2.00   # game seconds
TOTAL_RUN_TIME         = 100.00 # game seconds
TOTAL_FRAME_BUFFER     = 300
NUM_PEDESTRIANS        = 0
NUM_VEHICLES           = 2
SEED_PEDESTRIANS       = 0
SEED_VEHICLES          = 0

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]

PLAYER_START_INDEX = 1
FIGSIZE_X_INCHES   = 8
FIGSIZE_Y_INCHES   = 8
PLOT_LEFT          = 0.1    # in fractions of figure width and height
PLOT_BOT           = 0.1
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8

# Planning Constants
NUM_PATHS = 7
BP_LOOKAHEAD_BASE = 8.0 # m
BP_LOOKAHEAD_TIME = 2.0 # s
PATH_OFFSET = 1.5 # m
CIRCLE_OFFSETS = [-1.0, 1.0, 3.0] # m
CIRCLE_RADII = [1.5, 1.5, 1.5] # m
PATH_SELECT_WEIGHT = 10
TIME_GAP = 1.0 # s
A_MAX = 1.5 # m/s^2
SLOW_SPEED = 2.0 # m/s
STOP_LINE_BUFFER = 3.5 # m
LEAD_VEHICLE_LOOKAHEAD = 20.0 # m

WAYPOINTS_FILENAME = 'course4_waypoints.txt'
DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0

CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/controller_output/'

# Course 4 specific params
C4_STOP_SIGN_FILE        = 'stop_sign_params.txt'
C4_STOP_SIGN_FENCELENGTH = 5        # m
C4_PARKED_CAR_FILE       = 'parked_vehicle_params.txt'

def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need.
    """
    settings = CarlaSettings()
    
    # There is no need for non-agent info requests if there are no pedestrians
    # or vehicles.
    get_non_player_agents_info = False
    if (NUM_PEDESTRIANS > 0 or NUM_VEHICLES > 0):
        get_non_player_agents_info = True

    # Base level settings
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=get_non_player_agents_info, 
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        SeedVehicles=SEED_VEHICLES,
        SeedPedestrians=SEED_PEDESTRIANS,
        WeatherId=SIMWEATHER,
        QualityLevel=args.quality_level)
    return settings

class Timer(object):
    """ Timer Class
    
    The steps are used to calculate FPS, while the lap or seconds since lap is
    used to compute elapsed time.
    """
    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) /\
                     self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def get_current_pose(measurement):
    """Obtains current x,y,yaw pose from the client measurements
    
    Obtains the current x,y, and yaw pose from the client measurements.

    Args:
        measurement: The CARLA client measurements (from read_data())

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x   = measurement.player_measurements.transform.location.x
    y   = measurement.player_measurements.transform.location.y
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

    return (x, y, yaw)

def get_start_pos(scene):
    """Obtains player start x,y, yaw pose from the scene
    
    Obtains the player x,y, and yaw pose from the scene.

    Args:
        scene: The CARLA scene object

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x = scene.player_start_spots[0].location.x
    y = scene.player_start_spots[0].location.y
    yaw = math.radians(scene.player_start_spots[0].rotation.yaw)

    return (x, y, yaw)

def get_player_collided_flag(measurement, 
                             prev_collision_vehicles, 
                             prev_collision_pedestrians,
                             prev_collision_other):
    """Obtains collision flag from player. Check if any of the three collision
    metrics (vehicles, pedestrians, others) from the player are true, if so the
    player has collided to something.

    Note: From the CARLA documentation:

    "Collisions are not annotated if the vehicle is not moving (<1km/h) to avoid
    annotating undesired collision due to mistakes in the AI of non-player
    agents."
    """
    player_meas = measurement.player_measurements
    current_collision_vehicles = player_meas.collision_vehicles
    current_collision_pedestrians = player_meas.collision_pedestrians
    current_collision_other = player_meas.collision_other

    collided_vehicles = current_collision_vehicles > prev_collision_vehicles
    collided_pedestrians = current_collision_pedestrians > \
                           prev_collision_pedestrians
    collided_other = current_collision_other > prev_collision_other

    return (collided_vehicles or collided_pedestrians or collided_other,
            current_collision_vehicles,
            current_collision_pedestrians,
            current_collision_other)

def send_control_command(client, throttle, steer, brake, 
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client.
    
    Send control command to CARLA client.

    Args:
        client: The CARLA client object
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """
    control = VehicleControl()
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    client.send_control(control)

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
    """Store the resulting plot.
    """
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

def write_trajectory_file(x_list, y_list, v_list, t_list, collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')

    with open(file_name, 'w') as trajectory_file: 
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f %r\n' %\
                                  (x_list[i], y_list[i], v_list[i], t_list[i],
                                   collided_list[i]))

def write_collisioncount_file(collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'collision_count.txt')

    with open(file_name, 'w') as collision_file: 
        collision_file.write(str(sum(collided_list)))

def exec_waypoint_nav_demo(args):
    """ Executes waypoint navigation demo.
    """

    with make_carla_client(args.host, args.port) as client:
        print('Carla client connected.')

        settings = make_carla_settings(args)

        # Now we load these settings into the server. The server replies
        # with a scene description containing the available start spots for
        # the player. Here we can provide a CarlaSettings object or a
        # CarlaSettings.ini file as string.
        scene = client.load_settings(settings)

        # Refer to the player start folder in the WorldOutliner to see the 
        # player start information
        player_start = PLAYER_START_INDEX

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)

        #############################################
        # Load Configurations
        #############################################
        config = configparser.ConfigParser()
        config.read(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))         
        demo_opt = config['Demo Parameters']

        # Get options
        enable_live_plot = demo_opt.get('live_plotting', 'true').capitalize()
        enable_live_plot = enable_live_plot == 'True'
        live_plot_period = float(demo_opt.get('live_plotting_period', 0))

        # Set options
            # live plot timer
        live_plot_timer = Timer(live_plot_period)

        #############################################
        # Load stop sign and parked vehicle parameters
        # Convert to input params for LP
        #############################################
        # Stop sign (X(m), Y(m), Z(m), Yaw(deg))
        stopsign_data = None
        stopsign_fences = []     # [x0, y0, x1, y1]
        with open(C4_STOP_SIGN_FILE, 'r') as stopsign_file:
            next(stopsign_file)  # skip header
            stopsign_reader = csv.reader(stopsign_file, 
                                         delimiter=',', 
                                         quoting=csv.QUOTE_NONNUMERIC)
            stopsign_data = list(stopsign_reader)
            # convert to rad
            for i in range(len(stopsign_data)):
                stopsign_data[i][3] = stopsign_data[i][3] * np.pi / 180.0 

        # obtain stop sign fence points for LP
        for i in range(len(stopsign_data)):
            x = stopsign_data[i][0]
            y = stopsign_data[i][1]
            z = stopsign_data[i][2]
            yaw = stopsign_data[i][3] + np.pi / 2.0  # add 90 degrees for fence
            spos = np.array([
                    [0, 0                       ],
                    [0, C4_STOP_SIGN_FENCELENGTH]])
            rotyaw = np.array([
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
            spos_shift = np.array([
                    [x, x],
                    [y, y]])
            spos = np.add(np.matmul(rotyaw, spos), spos_shift)
            stopsign_fences.append([spos[0,0], spos[1,0], spos[0,1], spos[1,1]])

        # Parked car(s) (X(m), Y(m), Z(m), Yaw(deg), RADX(m), RADY(m), RADZ(m))
        parkedcar_data = None
        parkedcar_box_pts = []      # [x,y]
        with open(C4_PARKED_CAR_FILE, 'r') as parkedcar_file:
            next(parkedcar_file)  # skip header
            parkedcar_reader = csv.reader(parkedcar_file, 
                                          delimiter=',', 
                                          quoting=csv.QUOTE_NONNUMERIC)
            parkedcar_data = list(parkedcar_reader)
            # convert to rad
            for i in range(len(parkedcar_data)):
                parkedcar_data[i][3] = parkedcar_data[i][3] * np.pi / 180.0 

        # obtain parked car(s) box points for LP
        for i in range(len(parkedcar_data)):
            x = parkedcar_data[i][0]
            y = parkedcar_data[i][1]
            z = parkedcar_data[i][2]
            yaw = parkedcar_data[i][3]
            xrad = parkedcar_data[i][4]
            yrad = parkedcar_data[i][5]
            zrad = parkedcar_data[i][6]
            cpos = np.array([
                    [-xrad, -xrad, -xrad, 0,    xrad, xrad, xrad,  0    ],
                    [-yrad, 0,     yrad,  yrad, yrad, 0,    -yrad, -yrad]])
            rotyaw = np.array([
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
            cpos_shift = np.array([
                    [x, x, x, x, x, x, x, x],
                    [y, y, y, y, y, y, y, y]])
            cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)
            for j in range(cpos.shape[1]):
                parkedcar_box_pts.append([cpos[0,j], cpos[1,j]])

        #############################################
        # Load Waypoints
        #############################################

        waypoints_file = WAYPOINTS_FILENAME
        with open(waypoints_file) as waypoints_file_handle:
            waypoints = list(csv.reader(waypoints_file_handle, 
                                        delimiter=',',
                                        quoting=csv.QUOTE_NONNUMERIC))

        #############################################
        # Controller 2D Class Declaration
        #############################################
        # Initialize the controller.
        controller = controller2d.Controller2D(waypoints)

        #############################################
        # Determine simulation average timestep (and total frames)
        #############################################
        # ensure at least one simulation is used
        num_iterations = ITER_FOR_SIM_TIMESTEP
        if (ITER_FOR_SIM_TIMESTEP < 1):
            num_iterations = 1

        measurement_data, sensor_data = client.read_data()
        sim_start_stamp = measurement_data.game_timestamp / 1000.0
        # send a control command to proceed to next iteration
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        sim_duration = 0
        for i in range(num_iterations):
            # gather current data
            measurement_data, sensor_data = client.read_data()
            # send a control command to proceed to next iteration
            send_control_command(client, throttle=0.0, steer=0, brake=1.0)
            # last stamp
            if i == num_iterations - 1:
                sim_duration = measurement_data.game_timestamp / 1000.0 - \
                               sim_start_stamp  
        
        SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
        print("SERVER SIMULATION STEP APPROXIMATION: " + \
              str(SIMULATION_TIME_STEP))
        TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) /\
                               SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

        #############################################
        # Frame-by-Frame Iteration and Initialization
        #############################################
        # store pose history starting from the start position
        measurement_data, sensor_data = client.read_data()
        start_timestamp = measurement_data.game_timestamp / 1000.0
        start_x, start_y, start_yaw = get_current_pose(measurement_data)
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        x_history             = [start_x]
        y_history             = [start_y]
        yaw_history           = [start_yaw]
        time_history          = [0]
        speed_history         = [0]
        collided_flag_history = [False]  # assume player starts off non-collided

        #############################################
        # Vehicle Trajectory Live Plotting Setup
        #############################################

        lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
        lp_1d = lv.LivePlotter(tk_title="Controls Feedback")
        
        # Add 2D position / trajectory plot
        trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
                title='Vehicle Trajectory',
                figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
                edgecolor="black",
                rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

        trajectory_fig.set_invert_x_axis() # because UE4 - left-handed coord sys
        trajectory_fig.set_axis_equal() 

        waypoints_np = np.array(waypoints)
        trajectory_fig.add_graph("waypoints", window_size=waypoints_np.shape[0],
                                 x0=waypoints_np[:,0], y0=waypoints_np[:,1],
                                 linestyle="-", marker="", color='g')

        trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                 x0=[start_x]*TOTAL_EPISODE_FRAMES, 
                                 y0=[start_y]*TOTAL_EPISODE_FRAMES,
                                 color=[1, 0.5, 0])

        trajectory_fig.add_graph("start_pos", window_size=1, 
                                 x0=[start_x], y0=[start_y],
                                 marker=11, color=[1, 0.5, 0], 
                                 markertext="Start", marker_text_offset=1)

        trajectory_fig.add_graph("end_pos", window_size=1, 
                                 x0=[waypoints_np[-1, 0]], 
                                 y0=[waypoints_np[-1, 1]],
                                 marker="D", color='r', 
                                 markertext="End", marker_text_offset=1)

        trajectory_fig.add_graph("car", window_size=1, 
                                 marker="s", color='b', markertext="Car",
                                 marker_text_offset=1)

        trajectory_fig.add_graph("leadcar", window_size=1, 
                                 marker="s", color='g', markertext="Lead Car",
                                 marker_text_offset=1)

        trajectory_fig.add_graph("stopsign", window_size=1,
                                 x0=[stopsign_fences[0][0]], y0=[stopsign_fences[0][1]],
                                 marker="H", color="r",
                                 markertext="Stop Sign", marker_text_offset=1)
        
        trajectory_fig.add_graph("stopsign_fence", window_size=1,
                                 x0=[stopsign_fences[0][0], stopsign_fences[0][2]],
                                 y0=[stopsign_fences[0][1], stopsign_fences[0][3]],
                                 color="r")

        # load parked car points
        parkedcar_box_pts_np = np.array(parkedcar_box_pts)
        trajectory_fig.add_graph("parkedcar_pts", window_size=parkedcar_box_pts_np.shape[0],
                                 x0=parkedcar_box_pts_np[:,0], y0=parkedcar_box_pts_np[:,1],
                                 linestyle="", marker="+", color='b')

        for i in range(NUM_PATHS):
            trajectory_fig.add_graph("local_path " + str(i), window_size=200,
                                     x0=None, y0=None, color=[0.0, 0.0, 1.0])

        # Add 1D speed profile updater
        forward_speed_fig =\
                lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
        forward_speed_fig.add_graph("forward_speed", 
                                    label="forward_speed", 
                                    window_size=TOTAL_EPISODE_FRAMES)
        forward_speed_fig.add_graph("reference_signal", 
                                    label="reference_Signal", 
                                    window_size=TOTAL_EPISODE_FRAMES)

        # Add control signals graph
        throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle")
        throttle_fig.add_graph("throttle", 
                              label="throttle", 
                              window_size=TOTAL_EPISODE_FRAMES)

        # Add control signals graph
        brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake")
        brake_fig.add_graph("brake", 
                              label="brake", 
                              window_size=TOTAL_EPISODE_FRAMES)

        # Add control signals graph
        steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer")
        steer_fig.add_graph("steer", 
                              label="steer", 
                              window_size=TOTAL_EPISODE_FRAMES)

        # live plotter is disabled, hide windows
        if not enable_live_plot:
            lp_traj._root.withdraw()
            lp_1d._root.withdraw()        

        # Initialize the planning variables.
        wp_goal_index = 0
        lp = local_planner.LocalPlanner(NUM_PATHS, PATH_OFFSET, CIRCLE_OFFSETS, CIRCLE_RADII, PATH_SELECT_WEIGHT, TIME_GAP, A_MAX, SLOW_SPEED, STOP_LINE_BUFFER)
        bp = behavioural_planner.BehaviouralPlanner(BP_LOOKAHEAD_BASE, stopsign_fences, LEAD_VEHICLE_LOOKAHEAD)

        # Initialize the current timestamp.
        current_timestamp = start_timestamp

        # Iterate through the frames until the end of the waypoints are reached.
        prev_collision_vehicles = 0
        prev_collision_pedestrians = 0
        prev_collision_other = 0
        reached_the_end = False
        skip_first_frame = True
        local_waypoints = None
        path_validity = np.zeros((NUM_PATHS, 1), dtype=bool)
        for frame in range(TOTAL_EPISODE_FRAMES):
            # Gather current measurement data.
            measurement_data, sensor_data = client.read_data()

            # Update pose and timestamp.
            prev_timestamp = current_timestamp
            current_timestamp = float(measurement_data.game_timestamp) / 1000.0
            current_x, current_y, current_yaw = \
                get_current_pose(measurement_data)
            current_speed = measurement_data.player_measurements.forward_speed

            # Wait for some initial time before starting the demo.
            if current_timestamp <= WAIT_TIME_BEFORE_START:
                send_control_command(client, throttle=0.0, steer=0, brake=1.0)
                continue
            
            # Store state history.
            x_history.append(current_x)
            y_history.append(current_y)
            yaw_history.append(current_yaw)
            speed_history.append(current_speed)
            time_history.append(current_timestamp) 

            collided_flag,\
            prev_collision_vehicles,\
            prev_collision_pedestrians,\
            prev_collision_other = get_player_collided_flag(measurement_data,
                                                 prev_collision_vehicles,
                                                 prev_collision_pedestrians,
                                                 prev_collision_other)
            collided_flag_history.append(collided_flag)

            # Obtain lead vehicle information.
            lead_car_pos = []
            lead_car_length = []
            lead_car_speed = []
            for agent in measurement_data.non_player_agents:
                agent_id = agent.id  # unique id of the agent
                if agent.HasField('vehicle'):
                    lead_car_pos.append(
                            [agent.vehicle.transform.location.x,
                             agent.vehicle.transform.location.y])
                    lead_car_length.append(agent.vehicle.bounding_box.extent.x)
                    lead_car_speed.append(agent.vehicle.forward_speed)

            if frame % 2 == 0:
                # TODO Once you have completed the prerequisite functions of each of these
                # lines, you can uncomment the code below to run the planner. Note that
                # functions lower in this block often require outputs from the functions
                # earlier in this block, so it may be easier to implement those first to
                # get a more intuitive flow of the planner.
                # In addition, some of these functions have already been implemented for you,
                # but it is useful for you to understand what each function is doing.
                # Before you uncomment a function, please take the time to take a look at
                # it and understand what is going on. It will also help inform you on the
                # flow of the planner, which in turn will help you implement the functions
                # flagged for you in the TODO's.

                # Compute open loop speed estimate.
                open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

                # Calculate the goal state set in the local frame for the local planner.
                # Current speed should be open loop for the velocity profile generation.
                ego_state = [current_x, current_y, current_yaw, open_loop_speed]

                # Set lookahead based on current speed.   
                bp.set_lookahead(BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * open_loop_speed)

                # Perform a state transition in the behavioural planner.    
                bp.transition_state(waypoints, ego_state, current_speed)

                # Check to see if we need to follow the lead vehicle.
                bp.check_for_lead_vehicle(ego_state, lead_car_pos[1])

                # Compute the goal state set from the behavioural planner's computed goal state.
                goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, waypoints, ego_state)

                # Calculate planned paths in the local frame.
                paths, path_validity = lp.plan_paths(goal_state_set)

                # Transform those paths back to the global frame.
                paths = local_planner.transform_paths(paths, ego_state)

                # Perform collision checking.
                collision_check_array = lp._collision_checker.collision_check(paths, [parkedcar_box_pts]) 

                # Compute the best local path.
                best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)
                # If no path was feasible, continue to follow the previous best path.
                if best_index == None:
                    best_path = lp._prev_best_path
                else:
                    best_path = paths[best_index]
                    lp._prev_best_path = best_path

                # Compute the velocity profile for the path, and compute the waypoints.
                # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
                # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
                desired_speed = bp._goal_state[2]
                lead_car_state = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
                decelerate_to_stop = bp._state == behavioural_planner.DECELERATE_TO_STOP
                local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state, current_speed, decelerate_to_stop, lead_car_state, bp._follow_lead_vehicle)

                # Update the controller waypoint path with the best local path.
                # This controller is similar to that developed in Course 1 of this
                # specialization.
                controller.update_waypoints(local_waypoints)                

            # Update the controller with the sensor values, then compute the control commands
            # for this iteration.
            if local_waypoints != None:
                controller.update_values(current_x, current_y, current_yaw, 
                                         current_speed,
                                         current_timestamp, frame)
                controller.update_controls()
                cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
                print('cmd_throttle, cmd_steer, cmd_brake: ', cmd_throttle, cmd_steer, cmd_brake)
            else:
                cmd_throttle = 0.0
                cmd_steer = 0.0
                cmd_brake = 0.0

            # Skip the first frame
            if skip_first_frame and frame == 0:
                pass
            elif local_waypoints == None:
                pass
            else:
                # Update plot
                trajectory_fig.roll("trajectory", current_x, current_y)
                trajectory_fig.roll("car", current_x, current_y)
                if lead_car_pos:
                    trajectory_fig.roll("leadcar", lead_car_pos[1][0],
                                        lead_car_pos[1][1])

                forward_speed_fig.roll("forward_speed", 
                                       current_timestamp, 
                                       current_speed)
                forward_speed_fig.roll("reference_signal", 
                                       current_timestamp, 
                                       controller._desired_speed)

                throttle_fig.roll("throttle", current_timestamp, cmd_throttle)
                brake_fig.roll("brake", current_timestamp, cmd_brake)
                steer_fig.roll("steer", current_timestamp, cmd_steer)

                if frame % 2 == 0:
                    path_counter = 0
                    for i in range(NUM_PATHS):
                        # If a path was invalid in the set, there is no path to plot.
                        if path_validity[i]:
                            # Colour paths according to collision checking.
                            if not collision_check_array[path_counter]:
                                colour = 'r'
                            elif i == best_index:
                                colour = 'k'
                            else:
                                colour = 'b'
                            trajectory_fig.update("local_path " + str(i), paths[path_counter][0], paths[path_counter][1], colour)
                            path_counter += 1
                        else:
                            trajectory_fig.update("local_path " + str(i), [ego_state[0]], [ego_state[1]], 'r')


                    if enable_live_plot and \
                       live_plot_timer.has_exceeded_lap_period():
                        lp_traj.refresh()
                        lp_1d.refresh()
                        live_plot_timer.lap()

            # Output controller command to CARLA server
            send_control_command(client,
                                 throttle=cmd_throttle,
                                 steer=cmd_steer,
                                 brake=cmd_brake)

            # Find if reached the end of waypoint
            dist_to_last_waypoint = np.linalg.norm(np.array([
                waypoints[-1][0] - current_x,
                waypoints[-1][1] - current_y]))
            if dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
                reached_the_end = True
            if reached_the_end:
                break

        # End of demo - stop vehicle and store outputs
        if reached_the_end:
            print("Reached the end of path. Writing to controller_output...")
        else:
            print("Exceeded assessment time. Writing to controller_output...")
        send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
        store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
        store_trajectory_plot(forward_speed_fig.fig, 'forward_speed.png')
        store_trajectory_plot(throttle_fig.fig, 'throttle_output.png')
        store_trajectory_plot(brake_fig.fig, 'brake_output.png')
        store_trajectory_plot(steer_fig.fig, 'steer_output.png')
        write_trajectory_file(x_history, y_history, speed_history, time_history,
                              collided_flag_history)
        write_collisioncount_file(collided_flag_history)

def main():
    """Main function.

    Args:
        -v, --verbose: print debug information
        --host: IP of the host server (default: localhost)
        -p, --port: TCP port to listen to (default: 2000)
        -a, --autopilot: enable autopilot
        -q, --quality-level: graphics quality level [Low or Epic]
        -i, --images-to-disk: save images to disk
        -c, --carla-settings: Path to CarlaSettings.ini file
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    args = argparser.parse_args()

    # Logging startup info
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    # Execute when server connection is established
    while True:
        try:
            exec_waypoint_nav_demo(args)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

