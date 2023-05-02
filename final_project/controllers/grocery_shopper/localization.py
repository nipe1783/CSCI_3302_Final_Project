import math

def odometer(pose_x, pose_y, pose_theta, vL, vR, timestep, MAX_SPEED, MAX_SPEED_MS, AXLE_LENGTH):

    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    return pose_x, pose_y, pose_theta

def position_gps(gps, compass, world_height, world_width):

    '''
    finds robots env position

    gps: robot gps object
    compass: robot compass object.
    world_height: env height in m
    world_width: env width in m

    returns: 
        pose_x: robot x position in m
        pose_y: robot y position in  m
        pose_theta: robot theta orientation in rad
    '''
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]

    pose_x = -pose_x + (world_height / 2)
    pose_y = -pose_y + (world_width / 2)

    n = compass.getValues()
    rad = math.atan2(n[0], n[1]) + math.pi
    pose_theta = rad

    return pose_x, pose_y, pose_theta