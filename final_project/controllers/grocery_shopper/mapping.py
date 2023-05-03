import math


# -------------------------------------------------------------------------
# The majority of mapping code is included in the main file under lidar_map
# -------------------------------------------------------------------------

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