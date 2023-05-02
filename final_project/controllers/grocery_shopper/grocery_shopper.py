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
from rrt import rrt_star
from mapping import mode_manual, lidar_map

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

item0 = [2.6202858632038093, 3.58914, 0.5748653025126261]
item1 = [11.547000914636712, -8.505688675100652, 0.03486531467979112]
item2 = [-1.3296063450130449, -4.05309, 0.5748653625599697]
item3 = [2.3409306731047352, -3.53294, 0.5748654067999813]
item4 = [-0.8647696076431266, 0.3606378757006146, 0.5748653700715028]
item5 = [-2.773024136796192, 0.3100000000000003, 0.5748653025126261]
item6 = [0.7772010699377632, 0.3700000000000001, 0.5748652878199019]
item7 = [-2.6650533601864694, 0.2, 1.0748653705132794]
item8 = [-3.26060345214566, 3.629999999999986, 1.0748653543083286]
item9 = [5.67997938546148, 7.16031, 1.0748654067999996]
item10 = [3.504540887533399, 3.58914, 1.074864218478985]
item11 = [3.4994646114633245, 7.16914, 1.07486535924194]
goal_list = [item0, item1, item2, item3, item4, item5, item6, item7, item8, item9, item10, item11]

for goal in goal_list:
    goal[0] = world_height/2  - goal[0]
    goal[1] = world_width/2 - goal[1] 


# ------------------------------------------------------------------
# Robot Modes
mode = "manual"
# mode = "testing"
# mode = "autonomous"
# state = "openGripper"
state = "exploration"
counter = 0

map_width = int(360 * (world_width/ world_height))
map_height = 360

world_to_map_width = map_width / world_width
world_to_map_height = map_height / world_height

map = np.zeros(shape=[map_height,map_width])

heights = [0.575, 1.075]
active_links =  [False, False, False, False,  True, True, True, True, True, True, True, False, False]
arm_joints =  [0, 0, 0.35, 0,  0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 0, 0]
my_chain = ikpy.chain.Chain.from_urdf_file("arm.urdf", active_links_mask=active_links)
arm_queue = []
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

def goal_map(goal_list):

    for goal in goal_list:

        wx = goal[0]
        wy = goal[1]
        mx = abs(int(wx * world_to_map_height))
        my = abs(int(wy * world_to_map_width))
        display.setColor(int(0xFFFF00))
        display.drawPixel(my + 50,mx)

def location_map(pose_x, pose_y):

    mx = abs(int(pose_x * world_to_map_height))
    my = abs(int(pose_y * world_to_map_width))
    display.setColor(int(0xFFFFFF))
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

    if len(filtered_contours) > 0:

        # location of first goal detected
        c = filtered_contours[0]
        M = cv2.moments(c)
        gx = int(M['m10'] / M['m00'])
        gy = int(M['m01'] / M['m00'])
        return gx, gy, True
    else:
        return -1, -1, False
    
def goal_state():
    yellow = [255.0, 255.0, 0.0]
    for object in camera.getRecognitionObjects():
        color = object.getColors()
        color[0] = color[0]*255
        color[1] = color[1]*255
        color[2] = color[2]*255
        # print(color[0], color[1], color[2])
        if (color[0] == yellow[0] and color[1] == yellow[1] and color[2] == yellow[2]):

            return object.getPosition(), object.getOrientation()

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

def ik_arm(target_position, target_orientation=None, orientation_mode = None, initial = None, angle = None):
    global arm_joints
    if initial is None:
        initial = arm_joints
    # else:
        # target_frame[:3, :3] = [[0,1,0],[1,0,0],[0,0,1]]
    if angle is not None:
        angle = math.pi/2-angle
        rotate = np.array([[1,0,0],
                            [0, math.cos(-math.pi/2), math.sin(-math.pi/2)],
                            [0, -math.sin(-math.pi/2), math.cos(-math.pi/2)]])
        orientation = np.array([[math.cos(angle), math.sin(angle), 0],
                                [-math.sin(angle), math.cos(angle), 0],
                                [0,0,1]])
        target_orientation = np.dot(orientation,rotate)
        return my_chain.inverse_kinematics(target_position,target_orientation=target_orientation, orientation_mode="all" , initial_position=initial)
    elif target_orientation is not None:
        # print("orientation given")
        if orientation_mode is None:
            orientation_mode = "all"
        return my_chain.inverse_kinematics(target_position,target_orientation=target_orientation, orientation_mode=orientation_mode , initial_position=initial)
    else:
        return my_chain.inverse_kinematics(target_position, initial_position=initial)


def manipulate_to(newPose):
    """Use IK to calculate position and then deliver position to joints

        Parameters
        ----------
        target_position: array
            3 value array with desired position
        target_Orientation: numpy.array
            An optional 3x3 array for orientation at destination
        ---------
        Returns
        Null
    """
    global arm_joints
    arm_joints = newPose
    # print(arm_joints)
    robot_parts["arm_1_joint"].setPosition(arm_joints[4])
    robot_parts["arm_2_joint"].setPosition(arm_joints[5])
    robot_parts["arm_3_joint"].setPosition(arm_joints[6])
    robot_parts["arm_4_joint"].setPosition(arm_joints[7])
    robot_parts["arm_5_joint"].setPosition(arm_joints[8])
    robot_parts["arm_6_joint"].setPosition(arm_joints[9])
    robot_parts["arm_7_joint"].setPosition(arm_joints[10])
    
# Test statement/sanity check

angle = -1
# Main Loop

while robot.step(timestep) != -1:
    gx, gy, goal_detected = goal_detect()
    if mode == "manual":
        vL, vR = mode_manual(keyboard, MAX_SPEED, vL, vR)
    elif mode == "testing":
        if counter % 10 == 0:
            angle += 0.05
            print(angle)
            
                                        
            manipulate_to(ik_arm([0.6, 0.5, 1.2], angle=angle))
        counter += 1
    elif mode == "autonomous":
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
                goal_location, goal_orientation = goal_state()
                state = "orient"
        # Could we replace with pathfinding from previous lab?
        elif state == "orient":
            if gx == -1 and gy == -1:
                delay(20, "exploration")
                continue
            else:
                counter = 0
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
                state = "approach"
        elif state == "approach":
            if (not (gx == -1 and gy == -1)) and (gx < 110 or gx > 130):
                state = "orient"
            if lidar.getRangeImage()[333] > 1:
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
            angle = (123-gx)/120.
            print(angle)
            arm_queue = []
            points = np.linspace([0.2*math.cos(angle), 0.2*math.sin(angle), 0.7 + arm_joints[2]], [math.cos(angle), math.sin(angle), 0.7 + arm_joints[2]], 15)
            for i in points:
                arm_queue.append(ik_arm(i, angle=angle))
            state = "movingArmToReady"
        elif state == "movingArmToReady":
            if counter % 10 == 0:
                if len(arm_queue) > int(counter/10):
                    manipulate_to(arm_queue[int(counter/10)])
                else:
                    state = "closeGripper"
                    counter = -1
            counter += 1
        elif state == "closeGripper":
            vL = 0
            vR = 0
            robot_parts["gripper_left_finger_joint"].setPosition(0)
            robot_parts["gripper_right_finger_joint"].setPosition(0)
            delay(40, "lift")    
        elif state == "lift":
            manipulate_to(ik_arm([0.9, -0.05, 0.82 + arm_joints[2] ]))
            state = "backOut"
        elif state == "backOut":
            vL= -MAX_SPEED/2
            vR= -MAX_SPEED/2
            delay(100, "setArmToBasket")
        elif state == "setArmToBasket":
            vL = 0
            vR = 0
            basket = ik_arm([0.25, 0.05, 0.4 + arm_joints[2] ])
            arm_queue = np.linspace(arm_joints, basket, 20)
            state = "movingArmToBasket"
        elif state == "movingArmToBasket":
            if counter % 5 == 0:
                if len(arm_queue) > int(counter/5):
                    manipulate_to(arm_queue[int(counter/5)])
                else:
                    state = "releaseObject"
                    counter = -1
            counter += 1
        elif state == "releaseObject":
            robot_parts["gripper_left_finger_joint"].setPosition(0.045)
            robot_parts["gripper_right_finger_joint"].setPosition(0.045)
            delay(20, "stowArm")
        elif state == "stowArm":

            manipulate_to(ik_arm([0.0, -0.2, 1.6]))
            state = "exploration"
    
    # Odometer coardinates:
    # pose_x, pose_y, pose_theta = odometer(pose_x, pose_y, pose_theta, vL, vR, timestep)

    # GPS coardinates:
    pose_x, pose_y, pose_theta = position_gps(gps)
    # print("pose_x: ", pose_x, " pose_y: ", pose_y, " pose_theta: ", pose_theta)
    # print("pose_x: %f pose_y: %f pose_theta: %f vL: %f, vR: %f State: %s, gx: %i, gy: %i" % (pose_x, pose_y, pose_theta, vL, vR, state, gx, gy))
    #Lidar Map:
    lidar_map(pose_x, pose_y, pose_theta, world_width, world_height, LIDAR_SENSOR_MAX_RANGE, world_to_map_height, world_to_map_width, lidar, map, display)
    goal_map(goal_list)
    location_map(pose_x, pose_y)
        
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    
