"""grocery controller."""

# Nov 2, 2022

from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard, Display
import math
import numpy as np
import ikpy.chain
import ikpy.utils.plot as plot
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

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

map_width = int(360 * (world_width/ world_height))
map_height = 360

world_to_map_width = map_width / world_width
world_to_map_height = map_height / world_height

map = None
map = np.zeros(shape=[map_height,map_width])

active_links =  [False, False, True, False,  True, True, True, True, True, True, True, False, False]
my_chain = ikpy.chain.Chain.from_urdf_file("arm.urdf", active_links_mask=active_links)
# curr_pose = my_chain.forward_kinematics([1] * 13)
# target_orientation = [0.5, -0.5, 2.0]
# target_position = [curr_pose[0][3], curr_pose[1][3], curr_pose[2][3]]

# ------------------------------------------------------------------
# Helper Functions

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
            display.drawPixel(my,mx)

# ------------------------------------------------------------------
# Robot Modes
mode = "manual"
state = "openGripper"
counter = 0

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
        pass
    else: # slow down
        vL *= 0.75
        vR *= 0.75
    return vL, vR

def manipulate_to(target_position, target_orientation=None):

    # Note: +x is forword
    target_frame = np.eye(4)
    target_frame[:3, 3] = target_position
    if target_orientation is not None:
        target_frame[:3, :3] = target_orientation
    # else:
    joints = my_chain.inverse_kinematics_frame(target_frame, initial_position=[0.2]*13)
    robot_parts["torso_lift_joint"].setPosition(joints[2])
    robot_parts["arm_1_joint"].setPosition(joints[4])
    robot_parts["arm_2_joint"].setPosition(joints[5])
    robot_parts["arm_3_joint"].setPosition(joints[6])
    robot_parts["arm_4_joint"].setPosition(joints[7])
    robot_parts["arm_5_joint"].setPosition(joints[8])
    robot_parts["arm_6_joint"].setPosition(joints[9])
    robot_parts["arm_7_joint"].setPosition(joints[10])

# Test statement/sanity check
# manipulate_to([0.2, 0.3, 1.0])

# Main Loop
while robot.step(timestep) != -1:
    

    if mode == "manual":
        vL, vR = mode_manual(vL, vR)
        
    
    # Odometer coardinates:
    # pose_x, pose_y, pose_theta = odometer(pose_x, pose_y, pose_theta, vL, vR, timestep)

    # GPS coardinates:
    pose_x, pose_y, pose_theta = position_gps(gps)
    # print("pose_x: ", pose_x, " pose_y: ", pose_y, " pose_theta: ", pose_theta)

    #Lidar Map:
    lidar_map(pose_x, pose_y, pose_theta, lidar)
    if state == "exploration":
        pass
    if state == "navigation":
        pass
    elif state == "openGripper":
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            state = "setArmToTarget"
    elif state == "setArmToTarget":

        manipulate_to([1,0,1.5], )
        state = "movingArmToTarget"
    elif state == "movingArmToTarget":
        if (counter < 100):
            counter += 1
        else:
            state = "closeGripper"
            counter = 0
    elif state == "closeGripper":
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            state = "backOut"
    elif state == "backOut":
        if (counter < 50):
            counter += 1
            vL= -MAX_SPEED/2
            vR= -MAX_SPEED/2
        else:
            state = "setArmToBasket"
            counter = 0
    elif state == "setArmToBasket":

        manipulate_to([0.3, 0, 0.5])
        state = "movingArmToBasket"
    elif state == "movingArmToBasket":
        if (counter < 100):
            counter += 1
        else:
            state = "releaseObject"
            counter = 0
    elif state == "releaseObject":
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            state = "stowArm"
    elif state == "stowArm":

        manipulate_to([0.2, -0.2, 1.6])
        state = "exploration"

    
        
        
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    
