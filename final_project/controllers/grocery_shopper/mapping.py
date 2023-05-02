import math
from scipy.signal import convolve2d

def mode_manual(keyboard, MAX_SPEED, vL, vR):
    # Manual control mode for robot. 
    '''
    Manual control mode for robot
    keyboard: robot keyboard object.
    MAX_Speed: robot max speed.
    vL: robot current left wheel velocity.
    cR. robot current right wheel velocity.
    returns: vL, vR
    '''
    key = keyboard.getKey()
    print(key)
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
    else:
        vL *= 0.75
        vR *= 0.75
    return vL, vR

def obstacle_detected_roam(lidar):

    '''
    autonmous navigation. robot randomly moves and avoids obstacles.

    lidar: robot lidar object.

    returns:
        turn_left: bool that is true when robot should turn left
        turn_right: bool that is true when robot should turn right
    '''

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


def lidar_map(pose_x, pose_y, pose_theta, world_width, world_height, LIDAR_SENSOR_MAX_RANGE, world_to_map_height, world_to_map_width, lidar, map, display):
    '''
        Displays obstacles on display. Converts lidar point data into map data.
        pose_x: env x position of robot.
        pose_y: env y position of robot.
        pose_theta: env theta position of robot.
        world_width: env width in meters.
        world_height: env height in meters.
        LIDAR_SENSOR_MAX_RANGE: max viable range of lidar in meters.
        world_to_map_height: conversion from env height to map height.
        world_to_map_width: conersino from env width to map width.
        lidar: robot lidar object.

        returns: map, display
    '''
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    point_cloud_sensor_reading = lidar.getPointCloud()
    point_cloud_sensor_reading = point_cloud_sensor_reading[83:len(point_cloud_sensor_reading)-83]

    for i, point in enumerate(point_cloud_sensor_reading):

        # x, y, z are relative to lidar point origin.
        rx = point.x
        ry = point.y
        rho = math.sqrt( rx** 2+ ry**2)
        
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
            if map[mx, my] < 1:
                map[mx, my] += 0.005
            g = int(map[mx, my] * 255)
            display.setColor(g*(256**2) + g*256 + g)
            display.setColor(g)
            display.drawPixel(my + 50,mx)

    return map, display

def configuration_map(robot_space):
    '''
    creates configuration space map like in lab 5.

    robot_space: matrix of ones with num of rows and colums that are the length of the robot in pixel dimension.
    '''
    configuration_space = convolve2d(map, robot_space, mode = "same")
    configuration_space = (configuration_space >= 1).astype(int)
    return configuration_space

def goal_map(goal_list, display, world_to_map_height, world_to_map_width):
    '''
    shows cubes on display. 

    goal_list: list of x,y coa for cubes.
    display: robot display object.
    world_to_map_height: conversion from env height to map height.
    world_to_map_width: conersino from env width to map width.
    
    '''
    for goal in goal_list:

        wx = goal[0]
        wy = goal[1]
        mx = abs(int(wx * world_to_map_height))
        my = abs(int(wy * world_to_map_width))
        display.setColor(int(0xFFFF00))
        display.drawPixel(my + 50,mx)

    return display

def location_map(pose_x, pose_y, world_to_map_height, world_to_map_width, display):

    '''
    shows robot trail on display.

    pose_x: env x position of robot.
    pose_y: env y position of robot.

    returns:
        display: robot display object
    '''

    mx = abs(int(pose_x * world_to_map_height))
    my = abs(int(pose_y * world_to_map_width))
    display.setColor(int(0xFFFFFF))
    display.drawPixel(my + 50,mx)

    return display