"""grocery controller."""

# Nov 2, 2022

from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard, Display
import math
import numpy as np
import copy
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import ikpy
import ikpy.chain
import cv2
import heapq
from mapping import mode_manual, obstacle_detected_roam
from computer_vision import goal_detect, goal_locate, add_goal_state
from planning import rrt_star, visualize_path, getPathSpace
from localization import position_gps, navigate
from manipulation import manipulate_to, ik_arm, get_position
from helper import delay, find_nearest_goal
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


# Enable wheel encoders (position sensors)
left_wheel = robot.getDevice("wheel_left_joint_sensor")
left_wheel.enable(timestep)
right_wheel = robot.getDevice("wheel_right_joint_sensor")
right_wheel.enable(timestep)

# Enable torso encoders (position sensors)
torso_enc = robot.getDevice("torso_lift_joint_sensor")
torso_enc.enable(timestep)

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
# Origin is top left left of the room.

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

# queue that stores list of goal x,y,z locations
goal_queue = []

# ------------------------------------------------------------------
# Robot Modes
# mode = "manual"
mode = "autonomous"
# state = "openGripper"
state = "exploration"
counter = 0

map_width = int(360 * (world_width/ world_height))
map_height = 360

world_to_map_width = map_width / world_width
world_to_map_height = map_height / world_height

map = np.zeros(shape=[map_height,map_width])
seen = np.zeros(shape=[map_height,map_width], dtype='uint8')

heights = [0.575, 1.075]
active_links =  [False, False, False, False,  True, True, True, True, True, True, True, False, False]
arm_joints =  [0, 0, 0.35, 0,  0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 0, 0]
my_chain = ikpy.chain.Chain.from_urdf_file("arm.urdf", active_links_mask=active_links)
arm_queue = []
# curr_pose = my_chain.forward_kinematics([1] * 13)
# target_orientation = [0.5, -0.5, 2.0]
# target_position = [curr_pose[0][3], curr_pose[1][3], curr_pose[2][3]]

width = round(30*(AXLE_LENGTH)) # using same conversion where 360 pixels = 12 meters. 30 pixels per meter.
robot_space = np.ones(shape=[width,width])
waypoints = []
threshold = 0.3 # we can change this value for tuning of what is considered an obstacle.\

# angle = -1
# Main Loop

while robot.step(timestep) != -1:

    print(state)

    # GPS coardinates:
    pose_x, pose_y, pose_theta = position_gps(gps, compass, world_height, world_width)
    pose_z = 1.1+torso_enc.getValue()

    # Locate yellow blobs and return camera position, if blob is detected
    gx, gy, goal_detected = goal_detect(camera)

    if mode == "manual":
        mode_manual()

    elif mode == "autonomous":

        if state == "exploration":

            if goal_queue:
                # goal_queue has some goals in it. Robot navigates to them.

                # gathering goal location after blob is detected using webots API:
                goal_location = find_nearest_goal(pose_x, pose_y, goal_queue)
                goal_z = goal_location[2]
                goal_xy = goal_location[0:2]

                # setting robot torso to correct height:
                if goal_z < 0.85:
                    # goal on lower shelf
                    goal_shelf = "middle"
                    arm_joints[2] = 0
                else:
                    # goal on upper shelf
                    goal_shelf = "top"
                    arm_joints[2] = 0.35
                robot_parts["torso_lift_joint"].setPosition(arm_joints[2])

                # find location robot should navigate to:
                if pose_y < goal_xy[1]:
                    # goal is on right shelf robot should arrive in middle of isle.
                    rrt_goal = [goal_xy[0], goal_xy[1] - 1.5]
                else:
                    rrt_goal = [goal_xy[0], goal_xy[1] + 1.5]

                # compute RRT path:
                configuration_space = (convolve2d((map>=threshold).astype(int), robot_space, mode = "same") >= 1).astype(np.uint8)
                validity_check = lambda point: configuration_space[(int(point[0]* world_to_map_height), int(point[1]*world_to_map_width))] == 0 # check to see point is a valid space on map.
                goal_check = lambda point: math.dist(point, rrt_goal) < 0.1 # check to see if current point is within 0.8 meters of goal.
                bounds = np.array([[0, world_height],[0, world_width]])
                node_list, pathFound = rrt_star(bounds, validity_check, np.array([pose_x, pose_y]), np.array(rrt_goal), 1000, 0.2, state_is_goal=goal_check)
                if pathFound: 
                    waypoints = node_list[-1].getPath()
                    path_space = getPathSpace(waypoints, np.zeros(shape=[map_height,map_width]), robot_space, world_to_map_width, world_to_map_height)
                    visualize_path(waypoints, configuration_space, pose_x, pose_y, world_to_map_width, world_to_map_height)
                    state = "navigation"
                    counter = 0

            else:
                # Function to randomly explore map until goal is detected.
                turn_left, turn_right = obstacle_detected_roam(lidar)

                if turn_left and turn_right:
                    # Turn Left
                    vL = -MAX_SPEED/7
                    vR = MAX_SPEED/7
                elif turn_left:
                    # Turn Left
                    vL = -MAX_SPEED/7
                    vR = MAX_SPEED/7
                elif turn_right:
                    # Turn Right
                    vL = MAX_SPEED/7
                    vR = -MAX_SPEED/7
                else:
                    # Continue Forward
                    vL = MAX_SPEED/1.3
                    vR = MAX_SPEED/1.3
                if(goal_detected):
                    goal_queue = add_goal_state(camera, pose_x, pose_y, pose_z, pose_theta, goal_queue)

        elif state == "recalculate path":
            configuration_space = (convolve2d((map>=threshold).astype(int), robot_space, mode = "same") >= 1).astype(np.uint8)
            # plt.imshow(configuration_space)
            # plt.show()

            validity_check = lambda point: configuration_space[(int(point[0]* world_to_map_height), int(point[1]*world_to_map_width))] == 0
            goal_check = lambda point: seen[(int(point[0]* world_to_map_height), int(point[1]*world_to_map_width))] == 0

            bounds = np.array([[0, world_height],[0, world_width]])
            waypoints = rrt_star(bounds, validity_check, np.array([pose_x, pose_y]), np.array(goal_xy), 1000, 0.5, state_is_goal=goal_check)[-1].getPath()
            prevPoint = (int(pose_y * world_to_map_width), int(pose_x * world_to_map_height))
            for point in waypoints:
                point = (int(point[1] * world_to_map_width), int(point[0] * world_to_map_height))
                cv2.line(configuration_space, prevPoint, point, 1, 1)
                prevPoint = point
            plt.imshow(configuration_space)
            plt.show()
            state = "navigation"
            counter = 0

        elif state == "navigation":
            if np.any(np.bitwise_and(path_space, map>=threshold)):
                print("path invalid, recalculating")
                counter = 0
                waypoints = []
                state = "exploration"
            elif counter >= len(waypoints):
                state = "theta-docking"
                counter = 0
                waypoints = []
            elif math.dist((pose_x, pose_y), waypoints[counter]) < 0.3:
                counter += 1
            else:
                vL, vR= navigate(pose_x, pose_y, pose_theta, waypoints[counter])

        elif state == "theta-docking":

            buffer = 0.01
            if pose_x > goal_xy[0] + buffer:
                if pose_theta > (math.pi + buffer):
                    # rotate robot to pose_theta = pi
                    vL = MAX_SPEED/10
                    vR = -MAX_SPEED/10
                elif pose_theta < (math.pi - buffer):
                    # rotate robot to pose_theta = pi
                    vL = -MAX_SPEED/10
                    vR = MAX_SPEED/10
                else:
                    vL = 0
                    vR = 0
                    counter, state = delay(50, state, "x-docking", counter)
            elif pose_x < goal_xy[0] - buffer: 
                if pose_theta < (2*math.pi) and pose_theta > math.pi:
                    # rotate robot to pose_theta = 2pi
                    vL = -MAX_SPEED/10
                    vR = MAX_SPEED/10
                elif pose_theta > buffer and pose_theta < math.pi:
                    # rotate robot to pose_theta = 2pi
                    vL = MAX_SPEED/10
                    vR = -MAX_SPEED/10
                else:
                    vL = 0
                    vR = 0
                    counter, state = delay(50, state, "x-docking", counter)
            else:   
                    vL = 0
                    vR = 0
                    counter, state = delay(50, state, "re-theta-docking", counter)

        elif state == "x-docking":
            if pose_theta > (math.pi - math.pi/8) and pose_theta < (math.pi + math.pi/8):
                # robot is facing in -x direction.
                if pose_x > goal_xy[0] + buffer:
                    # move forward until goal.x == pose_x
                    if pose_theta > ( math.pi + buffer):
                        # turn right
                        vL = MAX_SPEED/5
                        vR = MAX_SPEED/10
                    elif pose_theta < ( math.pi + buffer):
                        # turn left
                        vL = MAX_SPEED/10
                        vR = MAX_SPEED/5
                    else:
                        # move forward
                        vL = MAX_SPEED/5
                        vR = MAX_SPEED/5
                else:
                    vL = 0
                    vR = 0
                    counter, state = delay(50, state, "re-theta-docking", counter)
            elif pose_theta > (2*math.pi - math.pi/8) or (pose_theta < math.pi/8):
                # robot is facing in +x direction.
                if pose_x < goal_xy[0] - buffer:
                    # move forward until goal.x == pose_x
                    if pose_theta < (buffer):
                        # turn right
                        vL = MAX_SPEED/5
                        vR = MAX_SPEED/10
                    elif pose_theta > ( 2*math.pi - buffer):
                        # turn left
                        vL = MAX_SPEED/10
                        vR = MAX_SPEED/5
                    else:
                        # move forward
                        vL = MAX_SPEED/5
                        vR = MAX_SPEED/5
                else:
                    vL = 0
                    vR = 0
                    counter, state = delay(50, state, "re-theta-docking", counter)
            else:
                state = "theta-docking"
        elif state == "re-theta-docking":
            buffer = 0.01
            if pose_y < goal_xy[1]:
                # robot needs to rotate to pi/2
                if pose_theta > (math.pi/2 + buffer) :
                    # rotate robot clockwise         
                    vL = MAX_SPEED/10
                    vR = -MAX_SPEED/10
                elif pose_theta < (math.pi/2 - buffer):
                    # rotate robot ccw
                    vL = -MAX_SPEED/10
                    vR = MAX_SPEED/10
                else:
                    vL = 0
                    vR = 0
                    goal_theta = math.pi/2
                    counter, state = delay(50, state, "orient-docking", counter)
            else:
                # robot needs to rotate to 3pi/2
                if pose_theta < (3*math.pi/2 - buffer) :
                    # # rotate robot clockwise
                    vL = -MAX_SPEED/10
                    vR = MAX_SPEED/10
                elif pose_theta > (3*math.pi/2 + buffer):
                    # rotate robot ccw
                    vL = MAX_SPEED/10
                    vR = -MAX_SPEED/10
                else:
                    vL = 0
                    vR = 0
                    goal_theta = math.pi * 3/2
                    counter, state = delay(50, state, "orient-docking", counter)

        elif state == "orient-docking":
            if gx == -1 and gy == -1 and lidar.getRangeImage()[333] > 1:
                vL = MAX_SPEED/20
                vR = MAX_SPEED/20
            elif gx == -1 and gy == -1:
                counter, state = delay(20, state, "exploration", counter)
                continue
            elif gx < 110:
                # robot rotate left
                vL = -MAX_SPEED/10
                vR = MAX_SPEED/20
            elif gx > 130:
                # robot rotate right
                vL = MAX_SPEED/20
                vR = -MAX_SPEED/10
            else:
                # go forward
                vL = 0
                vR = 0
                state = "approach"

        elif state == "approach":
            error_x = goal_xy[0] - pose_x
            if not (error_x > -.02 and error_x < 0.02):
                state = "orient-docking"
            if lidar.getRangeImage()[333] > 1:
                vL = MAX_SPEED/5
                vR = MAX_SPEED/5
            else:
                vL = 0
                vR = 0
                state = "theta-final"

        elif state == "theta-final":
            if pose_y > goal_xy[1]:
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
            goal_point, goal_orientation, position_on_camera = goal_locate(camera)
            angle = (123-position_on_camera[0])/120
            goal_point[0] -= .07
            goal_point[1] += 0.055
            if goal_shelf == "top":
                goal_point[2] = 1.03
            elif goal_shelf == "middle":
                goal_point[2] = 0.52
            position = robot_parts["arm_6_joint"].getTargetPosition()
            arm_queue = []
            points = np.linspace([0,0,1.05], goal_point)
            goal_point_prev = goal_point
            for i in points:
                arm_queue.append(ik_arm(i, my_chain, arm_joints, angle=angle))
            state = "movingArmToReady"

        elif state == "movingArmToReady":
            if counter % 10 == 0:
                if len(arm_queue) > int(counter/10):
                    robot_parts, arm_joints = manipulate_to(arm_queue[int(counter/10)], robot_parts)
                else:
                    state = "pause"
                    counter = -1
            counter += 1

        elif state == "pause":
            counter, state = delay(100, state, "check-wrist-position", counter)

        elif state =="check-wrist-position":
            # checks to see if wrist joint it lined up with cube center.
            wrist_position = get_position(arm_joints)
            goal_point, goal_orientation, position_on_camera = goal_locate(camera)
            goal_point[2] = 1.03
            goal_point[0] -= 0.07
            print("wrist pos: ", wrist_position, "goal_pose before: ", goal_point)
            if wrist_position[1] > goal_point[1]:
                goal_point[1] = wrist_position[1] - abs(goal_point[1] - wrist_position[1])/4
            else:
                goal_point[1] = wrist_position[1] + abs(goal_point[1] - wrist_position[1])/4

            print("wrist pos: ", wrist_position, "goal_pose after: ", goal_point)
            arm_queue = []
            angle = (123-position_on_camera[0])/120
            points = np.linspace(wrist_position, goal_point)
            for i in points:
                arm_queue.append(ik_arm(i, my_chain, arm_joints, angle=angle))
            state = "move-correct-arm"

        elif state == "move-correct-arm":
            if counter % 10 == 0:
                if len(arm_queue) > int(counter/10):
                    robot_parts, arm_joints = manipulate_to(arm_queue[int(counter/10)], robot_parts)
                else:
                    state = "pause2"
                    counter = -1
            counter += 1

        elif state == "pause2":
            counter, state = delay(100, state, "stop", counter)
        
        elif state == "stop":
            wrist_position = get_position(arm_joints)
            goal_point, goal_orientation, position_on_camera = goal_locate(camera)
            print("wrist pos: ", wrist_position, "goal_pos: ", goal_point)
            vL = 0
            vR = 0

        elif state == "closeGripper":
            vL = 0
            vR = 0
            robot_parts["gripper_left_finger_joint"].setPosition(0)
            robot_parts["gripper_right_finger_joint"].setPosition(0)
            counter, state = delay(40, state, "backOut", counter)   

        elif state == "backOut":
            vL= -MAX_SPEED/2
            vR= -MAX_SPEED/2
            counter, state = delay(100, state, "restore-height", counter)

        elif state == "restore-height":
            torso_position = robot_parts["torso_lift_joint"].getTargetPosition()
            if torso_position < 0.35:
                robot_parts["torso_lift_joint"].setPosition(torso_position + .01)
            vL = 0
            vR = 0
            counter, state = delay(100, state, "setArmToBasket", counter)
        elif state == "setArmToBasket":
            vL = 0
            vR = 0
            basket = ik_arm([0.25, 0.05, 0.4 + arm_joints[2] ], my_chain, arm_joints)
            arm_queue = np.linspace(arm_joints, basket, 30)
            # Fixing arm_joint_7 angle:
            part_1 = list(np.linspace(arm_queue[0][10], 2, 7))
            part_2 = list(np.linspace(0, -1, 8))
            part_3 = list(np.linspace(0, 0, 7))
            part_4 = list(np.linspace(0, -2, 8))
            arm_joint_7_points = np.array(part_1 + part_2 + part_3 + part_4)
            for i, queue in enumerate(arm_queue):
                queue[10] = arm_joint_7_points[i]
            state = "movingArmToBasket"

        elif state == "movingArmToBasket":
            if counter % 10 == 0:
                if len(arm_queue) > int(counter/10):
                    robot_parts, arm_joints = manipulate_to(arm_queue[int(counter/10)], robot_parts)
                    arm_7_position = robot_parts["arm_7_joint"].getTargetPosition()               
                else:
                    state = "releaseObject"
                    vL = 0
                    vR = 0
                    counter = -1
            counter += 1

        elif state == "releaseObject":
            robot_parts["gripper_left_finger_joint"].setPosition(0.045)
            robot_parts["gripper_right_finger_joint"].setPosition(0.045)
            robot_parts["arm_7_joint"].setPosition(0)
            counter, state = delay(100, state, "stowArm", counter)

        elif state == "stowArm":

            robot_parts, arm_joints = manipulate_to(ik_arm([0.0, -0.2, 1.6], my_chain, arm_joints), robot_parts)
            state = "exploration"
    
    # Odometer coardinates:
    # pose_x, pose_y, pose_theta = odometer(pose_x, pose_y, pose_theta, vL, vR, timestep)

    # print("pose_x: ", pose_x, " pose_y: ", pose_y, " pose_theta: ", pose_theta)
    # print("pose_x: %f pose_y: %f pose_theta: %f vL: %f, vR: %f State: %s, gx: %i, gy: %i" % (pose_x, pose_y, pose_theta, vL, vR, state, gx, gy))
    
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
            if (92 < i and i < number_of_readings-92):
                cv2.line(seen, (robot_Y_map, robot_X_map), (my, mx), 1, 1)
                display.setColor(int(0x777777))
                display.drawLine(robot_Y_map + 50, robot_X_map, my + 50, mx)
            g = int(map[mx, my] * 255)
            display.setColor(g)
            display.drawPixel(my + 50,mx)

    # display.setColor(int(0xFFFFFF))
    # display.drawPixel(robot_Y_map + 50,robot_X_map)
    
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    
