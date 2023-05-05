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