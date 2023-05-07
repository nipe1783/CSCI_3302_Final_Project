"""grocery controller."""

# Nov 2, 2022

from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard, Display
import math
import numpy as np
import copy
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import cv2
# from mapping import obstacle_detected_roam
from computer_vision import goal_detect
from planning import rrt_star, visualize_path, getPathSpace
from localization import position_gps, navigate
from manipulation import manipulate_to, ik_arm, get_position
from helper import delay
#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)
# target_pos = (0.0, 0.0, 0.35, 0.07, 0, -1, 2.29, -2.07, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)
torso_enc = robot.getDevice("torso_lift_joint_sensor")
torso_enc.enable(timestep)

# Enable wheel encoders (position sensors)
left_wheel = robot.getDevice("wheel_left_joint_sensor")
left_wheel.enable(timestep)
right_wheel = robot.getDevice("wheel_right_joint_sensor")
right_wheel.enable(timestep)

# Enable torso encoders (position sensors)
# torso_enc = robot.getDevice("torso_lift_joint")
# torso_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")

# Enable keyboard (optional)
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Environment dimesnions:
# Width: 16.1 m
# Height: 30 m
# Origin is in bottom right of the room.

world_width = 16.1 - 1.5
world_height = 30 - 1.5

# Odometry
pose_x     = world_height - 10
pose_y     = world_width/2
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

# queue that stores list of goal locations, format is [[x,y,z], onLeft]
goal_queue = []

# ------------------------------------------------------------------
# Robot Modes
# mode = "manual"
mode = "autonomous"
# state = "openGripper"
state = "start"
counter = 0
camera_on = True # Camera will need to be turned off at some stages to avoid duplicate objects

map_width = int(360 * (world_width/ world_height))
map_height = 360

world_to_map_width = map_width / world_width
world_to_map_height = map_height / world_height

width = 13 # using same conversion where 360 pixels = 12 meters. 30 pixels per meter.
robot_space = np.ones(shape=[width,width])
path_space = np.zeros(shape=[map_height,map_width], dtype= np.uint8) # A convolved map of the path the robot is on, used for collision checking
map = np.zeros(shape=[map_height,map_width]) # map of probable obstaces
seen = np.zeros(shape=[map_height,map_width], dtype='uint8') # map of points which the robot has seen, used in exploration to keep from going in circles

arm_joints =  [0, 0, target_pos[2], 0,  target_pos[3], target_pos[4], target_pos[5], target_pos[6], target_pos[7], 0.0, target_pos[8], 0, 0, 0]

arm_queue = []

waypoints = []
threshold = 0.3 # we can change this value for tuning of what is considered an obstacle.\

# angle = -1
# Main Loop
goal_point = ()

# arm_joints = ik_arm([0.8, 0, 0.8], arm_joints, target_orientation=np.array([[1,0,0],[0,-1,0],[0,0,-1]]), orientation_mode="all")
# robot_parts = manipulate_to(arm_joints, robot_parts)

while robot.step(timestep) != -1:

    # print(state)

    # GPS coardinates:
    pose_x, pose_y, pose_theta = position_gps(gps, compass, world_height, world_width)
    goal_point, goal_q = goal_detect(camera, pose_x, pose_y, 1.1+torso_enc.getValue(), pose_theta, goal_queue)
    if camera_on:
        goal_queue = goal_q
    if mode == "autonomous":
        if state == "start":
            counter, state = delay(10, state, "exploration", counter)
        if state == "exploration":
            if goal_queue:
            # if False:
                # goal_queue has some goals in it. Robot navigates to them
                # Occasionally, the robot will accidentally do into configuration space so backing up if a path is not
                nearest_goal_index = 0
                nearest_dist = math.dist((pose_x, pose_y), goal_queue[0][0][0:2])
                for i, goal in enumerate(goal_queue):
                    newDist = math.dist((pose_x, pose_y), goal[0][0:2])
                    if newDist < nearest_dist:
                        nearest_goal_index= i
                        nearest_dist= newDist
                nearest_goal = goal_queue.pop(i)
                goal_queue.append(nearest_goal)
                goal_location = nearest_goal[0]
                goal_z = goal_location[2]
                goal_xy = goal_location[0:2]
                goal_on_left = nearest_goal[1]
                nav_point = goal_xy
                # Check if goal is on left or right
                if goal_on_left:
                    nav_point[1] = nav_point[1]+0.6
                else:
                    nav_point[1] = nav_point[1]-0.6
                # Checks if goal is on top or bottom
                if goal_location[2] < 0.7:
                    arm_joints[2] = 0
                else:
                    arm_joints[2] = 0.35
                robot_parts["torso_lift_joint"].setPosition(arm_joints[2])
                # rrt* for navigation
                configuration_space = (convolve2d((map>=threshold).astype(int), robot_space, mode = "same") >= 1).astype(np.uint8)
                validity_check = lambda point: configuration_space[(int(point[0]* world_to_map_height), int(point[1]*world_to_map_width))] == 0
                bounds = np.array([[0, world_height],[0, world_width]])
                node_list, pathFound = rrt_star(bounds, validity_check, np.array([pose_x, pose_y]), np.array(nav_point), 300, 0.6, state_is_goal=goal_check)
                if not node_list:
                    # If the robot accidentally wanders into configuration space, it will back up so a path can be calculated
                    vL = -MAX_SPEED/3
                    vR = -MAX_SPEED/3
                if pathFound: 
                    waypoints = node_list[-1].getPath()
                    path_space = getPathSpace(waypoints, np.zeros(shape=[map_height,map_width]), robot_space, world_to_map_width, world_to_map_height)
                    # visualize_path(waypoints, configuration_space, pose_x, pose_y, world_to_map_width, world_to_map_height)
                    state = "navigation"
                    counter = 0
                    print("path to goal found")

            elif waypoints:
                # path following to explore unseen space
                if counter >= len(waypoints):
                    counter = 0
                    waypoints = []
                elif np.any(np.bitwise_and(path_space, map>=threshold)):
                    print("path invalid, recalculating")
                    counter = 0
                    waypoints = []
                elif math.dist((pose_x, pose_y), waypoints[counter]) < 0.3:
                    counter+=1
                else:
                    vL, vR= navigate(pose_x, pose_y, pose_theta, waypoints[counter])
            else:
                # go to unexplored territory
                # rrt* for exploration to unexplored space
                vL = -MAX_SPEED/4
                vR = -MAX_SPEED/4
                configuration_space = (convolve2d((map>=threshold).astype(int), robot_space, mode = "same") >= 1).astype(np.uint8)
                # configuration_space = np.add((convolve2d(seen, robot_space, mode = "same") < 1).astype(np.uint8), configuration_space)
                validity_check = lambda point: configuration_space[(int(point[0]* world_to_map_height), int(point[1]*world_to_map_width))] == 0
                goal_check = lambda point: (not seen[(int(point[0]* world_to_map_height), int(point[1]*world_to_map_width))]) and point[0] < 19
                bounds = np.array([[0, world_height],[0, world_width]])
                node_list, pathFound = rrt_star(bounds, validity_check, np.array([pose_x, pose_y]), None, 500, 1, state_is_goal=goal_check)
                if node_list: 
                    waypoints = node_list[-1].getPath()
                    path_space = getPathSpace(waypoints, np.zeros(shape=[map_height,map_width]), robot_space, world_to_map_width, world_to_map_height)
                    visualize_path(waypoints, configuration_space, pose_x, pose_y, world_to_map_width, world_to_map_height)
                    counter = 0

        elif state == "navigation":
            # print(waypoints)
            if np.any(np.bitwise_and(path_space, map>=threshold)):
                print("path invalid, recalculating")
                counter = 0
                waypoints = []
                state = "exploration"
            elif counter >= len(waypoints):
                state = "theta-docking"
                print("docking")
                counter = 0
                waypoints = []
            elif math.dist((pose_x, pose_y), waypoints[counter]) < 0.2:
                counter += 1
            else:
                vL, vR= navigate(pose_x, pose_y, pose_theta, waypoints[counter])

        elif state == "theta-docking":
            # print("docking")
            buffer = 0.03
            if goal_on_left:
                # rotate robot to 3pi/2
                if pose_theta > 3*math.pi/2 + buffer:
                    vL = MAX_SPEED/10
                    vR = -MAX_SPEED/10
                elif pose_theta < 3*math.pi/2 - buffer:
                    vL = -MAX_SPEED/10
                    vR = MAX_SPEED/10
                else:
                    vL = 0
                    vR = 0
                    state = "stabilize"
            else:
                # rotate robot to 3pi/2
                if pose_theta > math.pi/2 + buffer:
                    vL = MAX_SPEED/10
                    vR = -MAX_SPEED/10
                elif pose_theta < math.pi/2 - buffer:
                    vL = -MAX_SPEED/10
                    vR = MAX_SPEED/10
                else:
                    vL = 0
                    vR = 0
                    state = "stabilize"
        elif state == "stabilize":
            counter, state = delay(30, state, "openGripper", counter)

        elif state == "openGripper":
            robot_parts["gripper_left_finger_joint"].setPosition(0.045)
            robot_parts["gripper_right_finger_joint"].setPosition(0.045)
            if left_gripper_enc.getValue()>=0.044:
                state = "setArmToReady"

        elif state == "setArmToReady":
            if not goal_point:
                # object not found, the robot will forget this object and look for others
                vL = -MAX_SPEED
                vR = -MAX_SPEED
                goal_queue.pop(len(goal_queue)-1)
                state = "exploration"
                continue
            # angle = math.atan(goal_point[1]/goal_point[0])
            angle = 0
            # goal_point[0] += .07
            goal_point[1] += .015
            goal_point[2] = goal_location[2]-0.01
            print(goal_point)
            arm_queue = []
            print(get_position(arm_joints))
            # points = np.linspace(get_position(arm_joints), [-0.2,0,goal_point[2]], 5)
            # for i in points:
            #     # print(i)
            #     arm_queue.append(ik_arm(i,  arm_joints))
            points = np.linspace([0.1,goal_point[1],goal_point[2]], [goal_point[0]+0.2,goal_point[1],goal_point[2]],20)
            for i in points:
                # print(i)
                arm_queue.append(ik_arm(i,  arm_joints, angle=angle))
            state = "movingArmToReady"
            camera_on = False
        elif state == "movingArmToReady":
            if counter % 20 == 0:
                if len(arm_queue) > int(counter/20):
                    robot_parts = manipulate_to(arm_queue[int(counter/20)], robot_parts)
                else:
                    state = "closeGripper"
                    counter = -1
            counter += 1

        elif state == "closeGripper":
            vL = 0
            vR = 0
            robot_parts["gripper_left_finger_joint"].setPosition(0)
            robot_parts["gripper_right_finger_joint"].setPosition(0)
            counter, state = delay(40, state, "backOut", counter)   

        elif state == "backOut":
            vL= -MAX_SPEED/2
            vR= -MAX_SPEED/2
            counter, state = delay(40, state, "restore-height", counter)

        elif state == "restore-height":
            arm_joints[2] = 0.35
            robot_parts["torso_lift_joint"].setPosition(arm_joints[2])
            vL = 0
            vR = 0
            counter, state = delay(150, state, "setArmToBasket", counter)
        elif state == "setArmToBasket":
            vL = 0
            vR = 0
            arm_queue = []
            # arm_queue.append(ik_arm([0.25, 0.05, 0.6 + arm_joints[2] ], arm_joints, target_orientation=np.array([[1,0,0],[0,-1,0],[0,0,-1]]), orientation_mode="all"))
            arm_queue= np.linspace(arm_joints, ik_arm([0.2, 0.05, 0.4 + arm_joints[2] ], arm_joints, target_orientation=np.array([[1,0,0],[0,1,0],[0,0,1]]), orientation_mode="all"), 10)
            # Fixing arm_joint_7 angle:
            
            state = "movingArmToBasket"

        elif state == "movingArmToBasket":
            if counter % 10 == 0:
                if len(arm_queue) > int(counter/10):
                    robot_parts = manipulate_to(arm_queue[int(counter/10)], robot_parts)
                else:
                    state = "releaseObject"
                    vL = 0
                    vR = 0
                    counter = -1
            counter += 1

        elif state == "releaseObject":
            robot_parts["gripper_left_finger_joint"].setPosition(0.045)
            robot_parts["gripper_right_finger_joint"].setPosition(0.045)
            goal_queue.pop(len(goal_queue)-1)
            camera_on = True
            counter, state = delay(100, state, "stowArm", counter)
            state = "exploration"
    
    # Odometer coardinates:
    # pose_x, pose_y, pose_theta = odometer(pose_x, pose_y, pose_theta, vL, vR, timestep)

    # Debuggin Statements, uncomment to see more info, lag warning
    # print("pose_x: ", pose_x, " pose_y: ", pose_y, " pose_theta: ", pose_theta)
    # print("pose_x: %f pose_y: %f pose_theta: %f vL: %f, vR: %f, State: %s, Counter: %i" % (pose_x, pose_y, pose_theta, vL, vR, state, counter))
    
    # --------------
    # Lidar Mapping:
    # --------------
    
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    # This is to improve efficiency
    cos_pose_theta= math.cos(pose_theta)
    sin_pose_theta= math.sin(pose_theta)

    # (1-2/(4pi/3)*677/2-83) = 2=92
    number_of_readings = len(lidar_sensor_readings)

    robot_X_map = int(pose_x * world_to_map_height)
    robot_Y_map = int(pose_y * world_to_map_width)
    
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho >= LIDAR_SENSOR_MAX_RANGE:
            # This is just to draw a line to keep track of seen positions
            if (92 < i and i < number_of_readings-92):
                rx = math.cos(alpha)*LIDAR_SENSOR_MAX_RANGE
                ry = -math.sin(alpha)*LIDAR_SENSOR_MAX_RANGE

                # Convert detection from robot coordinates into world coordinates
                wx = cos_pose_theta * rx - sin_pose_theta * ry + pose_x
                wy = sin_pose_theta * rx + cos_pose_theta * ry + pose_y

                if wx >= world_height:
                    wx = world_height - .001
                elif wx <= 0:
                    wx = .001
                if  wy >= world_width:
                    wy = world_width - .001
                elif wy <= 0:
                    wy = .001

                mx = int(wx * world_to_map_height)
                my = int(wy * world_to_map_width)
                cv2.line(seen, (robot_Y_map, robot_X_map), (my, mx), 1, 1)
                display.setColor(int(0x777777))
                display.drawLine(robot_Y_map + 50, robot_X_map, my + 50, mx)
        else:

            rx = math.cos(alpha)*rho
            ry = -math.sin(alpha)*rho

            # Convert detection from robot coordinates into world coordinates
            wx = cos_pose_theta * rx - sin_pose_theta * ry + pose_x
            wy = sin_pose_theta * rx + cos_pose_theta * ry + pose_y

            if wx >= world_height:
                wx = world_height - .001
            elif wx <= 0:
                wx = .001
            if  wy >= world_width:
                wy = world_width - .001
            elif wy <= 0:
                wy = .001

            mx = int(wx * world_to_map_height)
            my = int(wy * world_to_map_width)
            if map[mx, my] < 1:
                map[mx, my] += 0.005
            # The statement below is to keep track of seen locations for the purpose of exploration
            if (92 < i and i < number_of_readings-92):
                cv2.line(seen, (robot_Y_map, robot_X_map), (my, mx), 1, 2)
                display.setColor(int(0x777777))
                display.drawLine(robot_Y_map + 50, robot_X_map, my + 50, mx)
            g = int(map[mx, my] * 255)
            display.setColor(g)
            display.drawPixel(my + 50,mx)

    display.setColor(int(0xFFFFFF))
    display.drawPixel(robot_Y_map + 50,robot_X_map)
    
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    
