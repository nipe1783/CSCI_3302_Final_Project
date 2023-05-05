import math

def delay(time, state, new_state, counter):
    '''
    helper function to make robot pause for certain amount of time

    time: amount of time robot should be paused in ms
    newState: string of robots new state
    counter: keeps track of how longer robot has been paused.
    '''
    if (counter < time):
        counter += 1
    else:
        counter = 0
        state = new_state

    return counter, state

def near(point, point_list):
    '''
    takes in [x,y,z] of point and list of points in point_list. 
    returns True if the new point is within 0.1 of all x,y,z directions
    '''
    for p in point_list:
        if abs(p[0] - point[0]) <= 0.2 and abs(p[1] - point[1]) <= 0.2 and abs(p[2] - point[2]) <= 0.2:
            return True
    return False

def same_color(color1, color2):
    if (color1[0]*255 == color2[0] and color1[1]*255 == color2[1] and color1[2]*255 == color2[2]):
        return True
    else:
        return False
    
def find_nearest_goal(pose_x,pose_y,goals):
    min_distance = float('inf')
    nearest_goal = None
    for i, goal in enumerate(goals):
        goal_dist = math.dist((pose_x, pose_y), goal[0:2])
        if goal_dist < min_distance:
            min_distance = goal_dist
            nearest_goal = i
    return goals[nearest_goal], nearest_goal