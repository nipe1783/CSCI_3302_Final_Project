import math

MAX_SPEED = 7.0  # [rad/s]
AXLE_LENGTH = 0.4044 # m
TURN_1_ANGLE = 0.4
TURN_2_ANGLE = 0.05


def position_odometer(pose_x, pose_y, pose_theta, vL, vR, timestep, MAX_SPEED, MAX_SPEED_MS, AXLE_LENGTH, gps, compass, world_height, world_width):
    """Odometer from previous labs, will reset to gps values if the error is too big
    """
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # TA SAID THIS WAS ACCEPTABLE
    true_x, true_y, true_theta = position_gps(gps, compass, world_height, world_width)

    if abs(true_x - pose_x) > 0.001 or abs(true_y - pose_y) > 0.001 or abs(true_theta - pose_theta) > 0.001:
        pose_x = true_x
        pose_y = true_y
        pose_theta = true_theta

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

def navigate(pose_x, pose_y, pose_theta, goal):
    """Given a point(goal), this will return the vL and vR to navigate to it. Taken from a previous lab"""
    rho = math.dist((pose_x, pose_y), goal)
    alpha = (2 * math.pi + math.atan2(goal[1] - pose_y, goal[0] - pose_x) - pose_theta) % (2 * math.pi)
    if alpha > math.pi:
        alpha = alpha - (2*math.pi)
    dTheta = 10*alpha
    vL = rho - (dTheta*AXLE_LENGTH/2)
    vR = rho + (dTheta*AXLE_LENGTH/2)
    speed = MAX_SPEED/2
    if abs(vL) > abs(vR):
        return speed, vR/abs(vL) * speed
    else: 
        return vL/abs(vR) * speed, speed
    