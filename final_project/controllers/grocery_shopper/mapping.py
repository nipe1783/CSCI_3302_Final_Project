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