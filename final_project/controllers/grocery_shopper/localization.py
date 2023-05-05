import math

MAX_SPEED = 7.0  # [rad/s]
AXLE_LENGTH = 0.4044 # m
TURN_1_ANGLE = 0.4
TURN_2_ANGLE = 0.05


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

def navigate(pose_x, pose_y, pose_theta, goal, navState):
    rho = math.dist((pose_x, pose_y), goal)
    alpha = (2 * math.pi + math.atan2(goal[1] - pose_y, goal[0] - pose_x) - pose_theta) % (2 * math.pi)
    if alpha > math.pi:
        alpha = alpha - (2*math.pi)
    print("Current xy: ", (pose_x,pose_y), "pose_theta: ", pose_theta, "goal: ", goal, "rho: ", rho, " alpha: ", alpha, "navState: ", navState)
    
    # navStates: 0:initial, 1: startLeft, 2: startRight, 3: endLeft, 4: endRight, 5: advance
    if navState == 0:
    #     if alpha < 0.05:
    #         return MAX_SPEED/2, -MAX_SPEED/2, 1
    #     elif alpha > 0.05:
    #         return -MAX_SPEED/2, MAX_SPEED/2, 2
    #     else:
    #         return 0,0,5
    # elif navState == 1:
    #     if alpha < TURN_1_ANGLE:
    #         return 0.3*MAX_SPEED, -0.25 * MAX_SPEED, 1
    #     else:
    #         return 0,0,3
        
    # elif navState == 2:
    #     if alpha > -TURN_1_ANGLE:
    #         speed = MAX_SPEED * abs(alpha/math.pi)
    #         return -0.25 * MAX_SPEED, 0.3 * MAX_SPEED, 2
    #     else:
    #         return 0,0,4
        
    # elif navState == 3:
    #     if alpha > TURN_2_ANGLE:
    #         speed = MAX_SPEED * abs(alpha/math.pi)
    #         return 0.5*speed, 2*speed, 3
    #     else:
    #         return 0,0,5
        
    # elif navState == 4:
    #     if alpha < -TURN_2_ANGLE:
    #         speed = MAX_SPEED * abs(alpha/math.pi)
    #         return 2*speed, 0.5*speed, 4
    #     else:
    #         return 0,0,5
        
    # else:
        dTheta = 10*alpha
        vL = rho - (dTheta*AXLE_LENGTH/2)
        vR = rho + (dTheta*AXLE_LENGTH/2)
        speed = MAX_SPEED/2
        if abs(vL) > abs(vR):
            return speed, vR/abs(vL) * speed, 0
        else: 
            return vL/abs(vR) * speed, speed, 0


    # dTheta = 5*alpha

    # vL = rho - (dTheta*AXLE_LENGTH/2)
    # vR = rho + (dTheta*AXLE_LENGTH/2)
    # speed = MAX_SPEED/2
    # if abs(vL) > abs(vR):
    #     vR = vR/abs(vL) * speed
    #     vL = speed
    # else: 
    #     vL = vL/abs(vR) * speed
    #     vR = speed
    # return vL, vR

    # if alpha > 0.1:
    #     # turn right
    #     vL = MAX_SPEED/10
    #     vR = MAX_SPEED/3
    # elif alpha < -0.1:
    #     # turn left
    #     vL = MAX_SPEED/3
    #     vR = MAX_SPEED/10
    # else:
    #     # move forward
    #     vL = MAX_SPEED/3
    #     vR = MAX_SPEED/3
    # return vL, vR

def docking(goal, pose_x, pose_y, pose_theta, vL, vR, mode):
    '''
    once navigation is complete docking will center goal on robots screen and make the robot perpendicular to the shelf.

    returns:
        vL:
        vR:
        mode:
    '''
    

    # case 1: robot is facing forward to the left. pose_theta > pi and pose_theta < 3pi/2
    if pose_theta > math.pi and pose_theta < (3/2 * math.pi):

        # rotate robot to pose_theta = pi
        pass
    