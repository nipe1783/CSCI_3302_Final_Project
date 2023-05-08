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
    
def find_nearest_goal_index(pose_x,pose_y,goal_queue):
    """
    Iterates through the goal queue and returns the index of the goal which is currently closest to the robot
    """
    min_dist = float('inf')
    # nearest_goal = None
    # for i, goal in enumerate(goals):
    #     goal_dist = math.dist((pose_x, pose_y), goal[0:2])
    #     if goal_dist < min_distance:
    #         min_distance = goal_dist
    #         nearest_goal = i
    # return goals.pop(nearest_goal)
    index = 0
    min_dist = float('inf')
    for i, goal in enumerate(goal_queue):
        new_dist = math.dist((pose_x, pose_y), goal[0][0:2])
        if new_dist < min_dist:
            index= i
            min_dist = new_dist
    return index