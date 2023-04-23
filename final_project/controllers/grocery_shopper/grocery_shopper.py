"""grocery controller."""

# Nov 2, 2022

from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard, Display
import math
import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import ikpy
import ikpy.chain
import ikpy.utils.plot as plot
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
import cv2
import heapq

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


# ------------------------------------------------------------------
# Robot Modes
# mode = "manual"
mode = "roam"
# state = "openGripper"
state = "exploration"
counter = 0

map_width = int(360 * (world_width/ world_height))
map_height = 360

world_to_map_width = map_width / world_width
world_to_map_height = map_height / world_height

map = np.zeros(shape=[map_height,map_width])

active_links =  [False, False, False, False,  True, True, True, True, True, True, True, False, False, False]
arm_joints =  [0, 0, 0.35, 0,  0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 0, 0, 0.045]
my_chain = ikpy.chain.Chain.from_urdf_file("arm.urdf", active_links_mask=active_links)
# curr_pose = my_chain.forward_kinematics([1] * 13)
# target_orientation = [0.5, -0.5, 2.0]
# target_position = [curr_pose[0][3], curr_pose[1][3], curr_pose[2][3]]

# ------------------------------------------------------------------
# Helper Functions

def delay(time, newState):
    global counter
    if (counter < time):
        counter += 1
    else:
        counter = 0
        global state
        state = newState
    if mode == "manual":
        vL, vR = mode_manual(vL, vR)

def odometer(pose_x, pose_y, pose_theta, vL, vR, timestep):

    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    return pose_x, pose_y, pose_theta

def position_gps(gps):
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]

    pose_x = -pose_x + (world_height / 2)
    pose_y = -pose_y + (world_width / 2)

    n = compass.getValues()
    rad = math.atan2(n[0], n[1]) + math.pi
    pose_theta = rad

    return pose_x, pose_y, pose_theta

def lidar_map(pose_x, pose_y, pose_theta, lidar):

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    point_cloud_sensor_reading = lidar.getPointCloud()
    point_cloud_sensor_reading = point_cloud_sensor_reading[83:len(point_cloud_sensor_reading)-83]

    for i, point in enumerate(point_cloud_sensor_reading):

        # x, y, z are relative to lidar point origin.
        rx = point.x
        ry = point.y
        rz = point.z
        rho = math.sqrt( rx** 2+ ry**2)

        alpha = lidar_offsets[i]

        # point location in world coa:
        wx = math.cos(pose_theta) * rx - math.sin(pose_theta) * ry + pose_x
        wy = math.sin(pose_theta) * rx + math.cos(pose_theta) * ry + pose_y

        if wx >= world_height:
            wx = world_height - .001
        elif wx <= 0:
            wx = .001
        if  wy >= world_width:
            wy = world_width - .001
        elif wy <= 0:
            wy = .001
        if abs(rho) < LIDAR_SENSOR_MAX_RANGE:
            mx = abs(int(wx * world_to_map_height))
            my = abs(int(wy * world_to_map_width))
            # print("wx: ", wx, " wy: ", wy, " mx: ", mx, " my: ", my)
            # You will eventually REPLACE the following lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            if map[mx, my] < 1:
                map[mx, my] += 0.005
            g = int(map[mx, my] * 255)
            # display.setColor(g*(256**2) + g*256 + g)
            display.setColor(int(0xFF0000))
            display.drawPixel(my + 50,mx)

configuration_space = np.zeros(shape=[360,360])
width = round(30*(AXLE_LENGTH)) # using same conversion where 360 pixels = 12 meters. 30 pixels per meter.
robot_space = np.ones(shape=[width,width])
threshold = 0.2 # we can change this value for tuning of what is considered an obstacle.

def configuration_map():
    configuration_space = convolve2d(map, robot_space, mode = "same")
    configuration_space = (configuration_space >= 1).astype(int)
    return configuration_space

def obstacle_detected_roam():

    # returns true if obstacle too close infront of robot
    turn_left = False
    turn_right = False
    point_cloud_sensor_reading = lidar.getPointCloud()
    point_cloud_sensor_reading = point_cloud_sensor_reading[83:len(point_cloud_sensor_reading)-83]
    point_center = point_cloud_sensor_reading[250]
    point_left = point_cloud_sensor_reading[200]
    point_right = point_cloud_sensor_reading[300]

    rho_center = math.sqrt( point_center.x** 2+ point_center.y**2)
    rho_left = math.sqrt( point_left.x** 2+ point_left.y**2)
    rho_right = math.sqrt( point_right.x** 2+ point_right.y**2)

    if(rho_right < 2):
        turn_left = True
    if(rho_left < 2):
        turn_right = True
    if(rho_center < 2):
        turn_right = True

    return turn_left, turn_right

def goal_detect():

    img = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15,200,200])
    upper_yellow = np.array([40,255,255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # apply Gaussian Blur
    smoothed = cv2.GaussianBlur(mask, (0,0), sigmaX=1.5, sigmaY=1.5, borderType = cv2.BORDER_DEFAULT)
    
    # Apply a morphological opening to remove noise and small objects
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

    # Find the contours of the remaining blobs in the image
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # CODE FOR SHOWING OBJECTS:
    
    # Draw the contours on a copy of the original image
    smoothed_copy = smoothed.copy()
    cv2.drawContours(smoothed, contours, -1, (0, 255, 0), 2)

    # Identify the center of the blob by calculating the centroid of the contour

    filtered_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            filtered_contours.append(c)

    for c in filtered_contours:
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(smoothed_copy, (cx, cy), 5, (0, 0, 255), -1)

    key = keyboard.getKey()
    while(keyboard.getKey() != -1): pass
    if key == ord('S'):
        plt.imshow(smoothed_copy)
        plt.show()

    if len(filtered_contours) > 0:
        # location of first goal detected
        c = contours[0]
        M = cv2.moments(c)

        gx = int(M['m10'] / M['m00'])
        gy = int(M['m01'] / M['m00'])
        return gx, gy, True
    else:
        return 0, 0, False

def goal_angle(gx):

    heading = True

    # rotate robot so goal is at center of image.
    if gx < 110:
        # robot rotate left
        vL = -MAX_SPEED/5
        vR = MAX_SPEED/15
    elif gx > 130:
        # robot rotate right
        vL = MAX_SPEED/15
        vR = -MAX_SPEED/5
    else:
        # go forward
        vL = 0
        vR = 0
        heading = False
    
    return vL, vR, heading


keyboard = robot.getKeyboard()
keyboard.enable(timestep)

def mode_manual(vL, vR):

    key = keyboard.getKey()
    while(keyboard.getKey() != -1): pass
    if key == keyboard.LEFT :
        vL = -MAX_SPEED
        vR = MAX_SPEED
    elif key == keyboard.RIGHT:
        vL = MAX_SPEED
        vR = -MAX_SPEED
    elif key == keyboard.UP:
        vL = MAX_SPEED
        vR = MAX_SPEED
    elif key == keyboard.DOWN:
        vL = -MAX_SPEED
        vR = -MAX_SPEED
    elif key == ord(' '):
        vL = 0
        vR = 0
    elif key == ord('S'):
        plt.imshow(configuration_space)
        plt.show()
    else: # slow down
        vL *= 0.75
        vR *= 0.75
    return vL, vR

def manipulate_to(target_position, target_orientation=None):
    """Use IK to calculate position and then deliver position to joints

        Parameters
        ----------
        target_position: array
            3 value array with desired position
        target_Orientation: numpy.array
            An optional 3x3 array for orientation at destination
        Output
        ---------
        Null
    """
    # Note: +x is forword
    target_frame = np.eye(4)
    target_frame[:3, 3] = target_position
    if target_orientation is not None:
        target_frame[:3, :3] = target_orientation
    # else:
        # target_frame[:3, :3] = [[0,1,0],[1,0,0],[0,0,1]]
    global arm_joints
    arm_joints = my_chain.inverse_kinematics_frame(target_frame, initial_position=arm_joints)
    print(target_position)
    print(arm_joints)
    robot_parts["torso_lift_joint"].setPosition(arm_joints[2])
    robot_parts["arm_1_joint"].setPosition(arm_joints[4])
    robot_parts["arm_2_joint"].setPosition(arm_joints[5])
    robot_parts["arm_3_joint"].setPosition(arm_joints[6])
    robot_parts["arm_4_joint"].setPosition(arm_joints[7])
    robot_parts["arm_5_joint"].setPosition(arm_joints[8])
    robot_parts["arm_6_joint"].setPosition(arm_joints[9])
    robot_parts["arm_7_joint"].setPosition(arm_joints[10])
    
# Test statement/sanity check
# manipulate_to([0.2, 0.3, 1.0])

# Main Loop
while robot.step(timestep) != -1:
    gx, gy, goal_detected = goal_detect()
    if mode == "roam":
        if state == "exploration":
            # Function to randomly explore map until goal is detected.
            # print("goal_detected: ", goal_detected)
            turn_left, turn_right = obstacle_detected_roam()

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
                state = "orient"
        # Could we replace with pathfinding from previous lab?
        elif state == "orient":
            vL, vR, heading_error = goal_angle(gx)
            if gx == 0 and gy == 0:
                delay(20, "exploration")
            else:
                counter = 0
            if not heading_error:
                state = "approach"
        elif state == "approach":
            if (not (gx == 0 and gy == 0)) and (gx < 110 or gx > 130):
                state = "orient"
            if lidar.getRangeImage()[333] > 1.3:
                vL = MAX_SPEED/5
                vR = MAX_SPEED/5
            else:
                vL = 0
                vR = 0
                state = "stabilize"
        elif state == "stabilize":
            delay(30, "openGripper")
        elif state == "openGripper":
            robot_parts["gripper_left_finger_joint"].setPosition(0.045)
            robot_parts["gripper_right_finger_joint"].setPosition(0.045)
            if left_gripper_enc.getValue()>=0.044:
                state = "setArmToReady"
        elif state == "setArmToReady":
            
            manipulate_to([1.2,
                           -0.25,
                           0.78 + arm_joints[2] ])
            state = "movingArmToReady"
        elif state == "movingArmToReady":
            delay(100, "forward")
        elif state == "forward":
            vL= MAX_SPEED/4
            vR= MAX_SPEED/4
            delay(100, "closeGripper")
        elif state == "closeGripper":
            vL = 0
            vR = 0
            robot_parts["gripper_left_finger_joint"].setPosition(0)
            robot_parts["gripper_right_finger_joint"].setPosition(0)
            delay(40, "lift")    
        elif state == "lift":
            manipulate_to([0.9,
                           -0.05,
                           0.82 + arm_joints[2] ])
            state = "backOut"
        elif state == "backOut":
            vL= -MAX_SPEED/2
            vR= -MAX_SPEED/2
            delay(100, "setArmToBasket")
        elif state == "setArmToBasket":
            vL = 0
            vR = 0
            manipulate_to([0.25,
                           0.05,
                           0.5 + arm_joints[2] ])
            state = "movingArmToBasket"
        elif state == "movingArmToBasket":
            delay(100, "releaseObject")
        elif state == "releaseObject":
            robot_parts["gripper_left_finger_joint"].setPosition(0.045)
            robot_parts["gripper_right_finger_joint"].setPosition(0.045)
            delay(20, "stowArm")
        elif state == "stowArm":

            manipulate_to([0.0, -0.2, 1.6])
            state = "exploration"
    
    # Odometer coardinates:
    # pose_x, pose_y, pose_theta = odometer(pose_x, pose_y, pose_theta, vL, vR, timestep)

    # GPS coardinates:
    pose_x, pose_y, pose_theta = position_gps(gps)
    # print("pose_x: ", pose_x, " pose_y: ", pose_y, " pose_theta: ", pose_theta)
    print("pose_x: %f pose_y: %f pose_theta: %f vL: %f, vR: %f State: %s, gx: %i, gy: %i" % (pose_x, pose_y, pose_theta, vL, vR, state, gx, gy))
    #Lidar Map:
    lidar_map(pose_x, pose_y, pose_theta, lidar)
        
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    
