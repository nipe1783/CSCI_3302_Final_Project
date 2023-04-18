"""grocery controller."""

# Nov 2, 2022

from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard, Display
import math
import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

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

map = np.zeros(shape=[map_height,map_width])



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
            display.drawPixel(my + 50,mx)

configuration_space = np.zeros(shape=[360,360])
width = round(30*(AXLE_LENGTH + .25)) # using same conversion where 360 pixels = 12 meters. 30 pixels per meter.
robot_space = np.ones(shape=[width,width])
threshold = 0.2 # we can change this value for tuning of what is considered an obstacle.

def configuration_map():
    configuration_space = convolve2d(map, robot_space, mode = "same")
    configuration_space = (configuration_space >= 1).astype(int)
    return configuration_space

def obstacle_detected():

    # returns true if obstacle too close infront of robot
    turn_left = False
    turn_right = False
    point_cloud_sensor_reading = lidar.getPointCloud()
    point_cloud_sensor_reading = point_cloud_sensor_reading[83:len(point_cloud_sensor_reading)-83]
    point_center = point_cloud_sensor_reading[250]
    point_left = point_cloud_sensor_reading[230]
    point_right = point_cloud_sensor_reading[270]

    rho_center = math.sqrt( point_center.x** 2+ point_center.y**2)
    rho_left = math.sqrt( point_left.x** 2+ point_left.y**2)
    rho_right = math.sqrt( point_right.x** 2+ point_right.y**2)

    if(rho_right < 2):
        turn_left = True
    if(rho_left < 2):
        turn_right = True
    if(rho_center < 1):
        turn_right = True

    return turn_left, turn_right

def roam(vL, vR):
    # Function to randomly explore map until goal is detected.
    turn_left, turn_right = obstacle_detected()

    if turn_left and turn_right:
        # Turn Right
        vL = MAX_SPEED/5
        vR = -MAX_SPEED/5
    elif turn_left:
        # Turn Left
        vL = -MAX_SPEED/5
        vR = MAX_SPEED/5
    elif turn_right:
        # Turn Right
        vL = MAX_SPEED/5
        vR = -MAX_SPEED/5
    else:
        # Continue Forward
        vL = MAX_SPEED
        vR = MAX_SPEED

    return vL, vR

# ------------------------------------------------------------------
# Robot Modes
mode = "manual"
mode = "roam"

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



gripper_status="closed"

# Main Loop
while robot.step(timestep) != -1:
    

    if mode == "manual":
        vL, vR = mode_manual(vL, vR)

    if mode == "roam":
        vL, vR = roam(vL, vR)
        
    
    # Odometer coardinates:
    # pose_x, pose_y, pose_theta = odometer(pose_x, pose_y, pose_theta, vL, vR, timestep)

    # GPS coardinates:
    pose_x, pose_y, pose_theta = position_gps(gps)
    # print("pose_x: ", pose_x, " pose_y: ", pose_y, " pose_theta: ", pose_theta)

    #Lidar Map:
    lidar_map(pose_x, pose_y, pose_theta, lidar)

    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            gripper_status="closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            gripper_status="open"
