"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
import heapq
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

# range = robot.getDevice('range-finder')
# range.enable(timestep)
# camera = robot.getDevice('camera')
# camera.enable(timestep)
# camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
# mode = 'manual' 
# Part 1.1: manual mode
# mode = 'planner'
mode = 'autonomous'
print(mode)



###################
#
# Planner
#
###################
if mode == 'planner':
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    start_w = (-8.437874, -4.906207) # (Pose_X, Pose_Y) in meters
    end_w = (-3, -4) # (Pose_X, Pose_Y) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    start = (abs(int(start_w[0]*30)), abs(int(start_w[1]*30))) # (x, y) in 360x360 map
    end = (abs(int(end_w[0] * 30)), abs(int(end_w[1] * 30))) # (x, y) in 360x360 map
    print(end)
    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    def path_planner(map, start, end):
        # '''
        # :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
        # :param start: A tuple of indices representing the start cell in the map
        # :param end: A tuple of indices representing the end cell in the map
        # :return: A list of tuples as a path from the given start to the given end in the given maze
        # '''
        # pass
        # A*
        # format: (estimate, (x,y), (previous_x, previous_y), distance)
        queue = [(math.dist(start, end), start, (-1,-1), 0)]
        heapq.heapify(queue)
        checked = []
        while(queue):
            queueItem = heapq.heappop(queue)
            (estimate, point, prev, distance) = queueItem
            if map[point[0], point[1]] == 0:
                # print(queueItem)
                if point == end:
                    print('tracing through ' + str(len(checked)) + " nodes")
                    checked.reverse()
                    path = [point]
                    node = prev
                    for i in checked:
                        if(i[1] == node):
                            if(i[1] == start):
                                path.reverse()
                                return path
                            path.append(node)
                            node = i[2]
                    path.reverse()
                    return path
                
                checked.append(queueItem)
                map[point[0], point[1]] = 1
                # print(point)
                nextSteps = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
                for i in nextSteps:
                    next = (point[0] + i[0], point[1] + i[1])
                    # print(next)
                    if next[0] >= 0 and next[0] < 360 and next[1] >= 0 and next[1] < 360 and map[next[0], next[1]] == 0:
                        newDistance = distance + math.sqrt(i[0]**2 + i[1]**2)
                        heapq.heappush(queue, (newDistance + math.dist(next, end), next, point, newDistance))
        print("Error: Path Not Found")
        return [end]
        
            
    # Part 2.1: Load map (map.npy) from disk and visualize it
    map = np.load("map.npy")
    
    plt.imshow(map)
    plt.show()

    # Part 2.2: Compute an approximation of the “configuration space”
    # mapFilter = np.ones(shape = [13, 13]) 
    # configSpace = convolve2d(map, mapFilter)
    # plt.imshow(configSpace)
    # plt.show()
    
    configuration_space = np.zeros(shape=[360,360])
    width = round(30*(AXLE_LENGTH + .25)) # using same conversion where 360 pixels = 12 meters. 30 pixels per meter.
    robot_space = np.ones(shape=[width,width])
    threshold = 0.2 # we can change this value for tuning of what is considered an obstacle.
    map[map < threshold] = 0
    map[map >= threshold] = 1
    configuration_space = convolve2d(map, robot_space, mode = "same")
    configuration_space = (configuration_space >= 1).astype(int)
    plt.imshow(configuration_space)
    plt.show()


    # Part 2.3 continuation: Call path_planner
    path = path_planner(configuration_space, start, end)
    # plt.imshow(configuration_space)
    # plt.show()

    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    waypoints = []
    for i in path:
        waypoints.append((-i[0]/30, -i[1]/30))
    np.save("path.npy", waypoints)
    for i in path:
        map[i] = 1
    # plt.imshow(configuration_space)
    # plt.show()
    plt.imshow(map)
    plt.show()
    print("path saved")

######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization


# Initialize your map data structure here as a 2D floating point array
map = np.zeros(shape=[360,360])
waypoints = []

if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    waypoints = np.load("path.npy") # Replace with code to load your path
    # print(waypoints)
state = 0 # use this to iterate through your path




while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    ################ v [Begin] Do not modify v ##################
    # Ground truth pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    
    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.
        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy =  math.sin(t)*rx + math.cos(t)*ry + pose_y

        ################ ^ [End] Do not modify ^ ##################
        # print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))
        if abs(wx) >= 12:
            wx = 11.999
        if abs(wy) >= 12:
            wy = 11.999
        if abs(rho) < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.
 
            # You will eventually REPLACE the following lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.
            if map[abs(int(wx*30)), abs(int(wy*30))] < 1:
                map[abs(int(wx*30)), abs(int(wy*30))] += 0.005
            g = int(map[abs(int(wx*30)), abs(int(wy*30))] *255)
            display.setColor(g*(256**2) + g*256 + g)
            display.drawPixel(360-abs(int(wx*30)),abs(int(wy*30)))

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(360-abs(int(pose_x*30)), abs(int(pose_y*30)))

    ###################
    #
    # Controller
    #
    ###################
    if mode == 'manual':
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
            # Part 1.4: Filter map and save to filesystem
            threshold_map = np.multiply(map > 0.5, 1)
            np.save("map.npy", threshold_map)
              
            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            print("Map loaded")
        else: # slow down
            vL *= 0.75
            vR *= 0.75
    else: # not manual mode
        # Part 3.2: Feedback controller
        #STEP 1: Calculate the error
        rho = math.dist((pose_x, pose_y), waypoints[state])
        alpha = (2 * math.pi + math.atan2(pose_x - waypoints[state][0], waypoints[state][1] - pose_y) - pose_theta) % (2 * math.pi)
        if alpha > math.pi:
            alpha = alpha - (2*math.pi)
        # print(pose_theta)
        #STEP 2: Controller
        dX = rho
        dTheta = 10*alpha

        #STEP 3: Compute wheelspeeds
        vL = dX - (dTheta*AXLE_LENGTH/2)
        vR = dX + (dTheta*AXLE_LENGTH/2)
        
        speed = MAX_SPEED * 0.2
        if abs(vL) > abs(vR):
            vR = vR/abs(vL) * speed
            if vL > 0:
                vL = speed
            else:
                vL = -speed
        else: 
            vL = vL/abs(vR) * speed
            if vR > 0:
                vR = speed
            else:
                vR = -speed
                
        if dX < 0.15:
            if state == len(waypoints)-1:
                vR = 0
                vL = 0
            else:
                state += 1
        # Normalize wheelspeed
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)
        print("dX: %f dTheta: %f Alpha: %f State: %i, (%f, %f)" % (dX, dTheta, alpha, state, waypoints[state][0], waypoints[state][1]))

        # Normalize wheelspeed
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)


    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("vL: %f vR: %f X: %f Z: %f Theta: %f" % (vL, vR, pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
    
while robot.step(timestep) != -1:
    pass
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    # this is to keep the controller running