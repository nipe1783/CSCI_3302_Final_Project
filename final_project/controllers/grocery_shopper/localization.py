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

def navigate(pose_x, pose_y, pose_theta, goal, AXLE_LENGTH, MAX_SPEED):
    rho = math.dist((pose_x, pose_y), goal)
    alpha = (2 * math.pi + math.atan2(pose_x - goal[0], goal[1] - pose_y) - pose_theta) % (2 * math.pi)
    if alpha > math.pi:
        alpha = alpha - (2*math.pi)
    alpha *= 100

    print("rho: ", rho, " alpha: ", alpha)

    # vL = dX - (dTheta*AXLE_LENGTH/2)
    # vR = dX + (dTheta*AXLE_LENGTH/2)
    vL = rho - (alpha*AXLE_LENGTH/2)
    vR = rho + (alpha*AXLE_LENGTH/2)
    
    speed = min(MAX_SPEED * 0.25, rho*10)
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
    return vL, vR
    # speed = 0.8*MAX_SPEED * (abs(alpha/math.pi))+0.2
    # if alpha < -0.05:
    #     return speed, 0.5*speed
    # elif alpha > 0.05:
    #     return 0.5*speed, speed
    # # elif alpha < -0.1:
    # #     return min(MAX_SPEED * rho, MAX_SPEED)*0.1, 0
    # # elif alpha > 0.1:
    # #     return 0, min(MAX_SPEED * rho, MAX_SPEED)*0.1
    # else:
    #     return min(MAX_SPEED * rho, MAX_SPEED)*0.5, min(MAX_SPEED * rho, MAX_SPEED)*0.5