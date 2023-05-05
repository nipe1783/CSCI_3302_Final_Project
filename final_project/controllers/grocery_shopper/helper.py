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
        if abs(p[0] - point[0]) <= 0.1 and abs(p[1] - point[1]) <= 0.1 and abs(p[2] - point[2]) <= 0.1:
            return True
    return False

def same_color(color1, color2):
    color1[0] = color1[0]*255
    color1[1] = color1[1]*255
    color1[2] = color1[2]*255
    if (color1[0] == color2[0] and color1[1] == color2[1] and color1[2] == color2[2]):
        return True
    else:
        return False