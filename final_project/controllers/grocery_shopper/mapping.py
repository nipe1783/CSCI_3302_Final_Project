import math

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