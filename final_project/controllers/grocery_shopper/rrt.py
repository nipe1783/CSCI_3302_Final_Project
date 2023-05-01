import numpy as np
import matplotlib.pyplot as plt
import math
import random

###############################################################################
## Base Code
###############################################################################
class Node:
    """
    Node for RRT Algorithm. This is what you'll make your graph with!
    """
    def __init__(self, pt, parent=None):
        self.point = pt # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = [] # List of points along the way from the parent node (for edge's collision checking)
    def getCost(self):
        node = self
        cost = 0
        while node.parent is not None:
            cost += math.dist(node.point, node.parent.point)
            node = node.parent
        return cost
        

def get_random_valid_vertex(state_is_valid, bounds):
    '''
    Function that samples a random n-dimensional point which is valid (i.e. collision free and within the bounds)
    :param state_valid: The state validity function that returns a boolean
    :param bounds: The world bounds to sample points from
    :return: n-Dimensional point/state
    '''
    vertex = None
    while vertex is None: # Get starting vertex
        pt = np.random.rand(bounds.shape[0]) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
        if state_is_valid(pt):
            vertex = pt
    return vertex

###############################################################################
## END BASE CODE
###############################################################################

def get_nearest_vertex(node_list, q_point):
    '''
    Function that finds a node in node_list with closest node.point to query q_point
    :param node_list: List of Node objects
    :param q_point: n-dimensional array representing a point
    :return Node in node_list with closest node.point to query q_point
    '''

    # TODO: Your Code Here
    closestDistance = math.dist(q_point, node_list[0].point)
    closestNode = node_list[0]
    for i in node_list:
        distance = math.dist(q_point, i.point)
        if distance < closestDistance:
            closestNode = i
            closestDistance = distance
    return closestNode

def steer(from_point, to_point, delta_q):
    '''
    :param from_point: n-Dimensional array (point) where the path to "to_point" is originating from (e.g., [1.,2.])
    :param to_point: n-Dimensional array (point) indicating destination (e.g., [0., 0.])
    :param delta_q: Max path-length to cover, possibly resulting in changes to "to_point" (e.g., 0.2)
    :return path: Array of points leading from "from_point" to "to_point" (inclusive of endpoints)  (e.g., [ [1.,2.], [1., 1.], [0., 0.] ])
    '''
    # TODO: Figure out if you can use "to_point" as-is, or if you need to move it so that it's only delta_q distance away
    dist = math.dist(from_point, to_point)
    if delta_q < dist:
        to_point = from_point + (delta_q/dist) *(to_point - from_point)
        
    # TODO Use 
    # the np.linspace function to get 10 points along the path from "from_point" to "to_point"
    path = np.linspace(from_point, to_point, num = 10)
    return path

def check_path_valid(path, state_is_valid):
    '''
    Function that checks if a path (or edge that is made up of waypoints) is collision free or not
    :param path: A 1D array containing a few (10 in our case) n-dimensional points along an edge
    :param state_is_valid: Function that takes an n-dimensional point and checks if it is valid
    :return: Boolean based on whether the path is collision free or not
    '''
    for i in path: 
        if not state_is_valid(i):
            return False
    return True
    # TODO: Your Code Here
    raise NotImplementedError

def rrt_star(state_bounds, state_is_valid, starting_point, goal_point, k, delta_q, r=None, state_is_goal=None):
    '''
    TODO: Implement the RRT algorithm here, making use of the provided state_is_valid function.
    RRT algorithm.
    If goal_point is set, your implementation should return once a path to the goal has been found 
    (e.g., if q_new.point is within 1e-5 distance of goal_point), using k as an upper-bound for iterations. 
    If goal_point is None, it should build a graph without a goal and terminate after k iterations.

    :param state_bounds: matrix of min/max values for each dimension (e.g., [[0,1],[0,1]] for a 2D 1m by 1m square)
    :param state_is_valid: function that maps states (N-dimensional Real vectors) to a Boolean (indicating free vs. forbidden space)
    :param starting_point: Point within state_bounds to grow the RRT from
    :param goal_point: Point within state_bounds to target with the RRT. (OPTIONAL, can be None)
    :param k: Number of points to sample
    :param delta_q: Maximum distance allowed between vertices
    :param r: distance of nodes to check for cost in RRT*
    :returns List of RRT graph nodes
    '''
    node_list = []
    node_list.append(Node(starting_point, parent=None)) # Add Node at starting point with no parent
    if r is None or r > delta_q:
        r=delta_q
    for i in range(k):
        goalChecked = False
        point=[]
        if goal_point is not None and random.random() < 0.05 :
            point = goal_point
        else:
            goalChecked=True
        while True:
            if goalChecked:
                point = get_random_valid_vertex(state_is_valid, state_bounds)
            else:
                goalChecked=True
            closest = get_nearest_vertex(node_list, point)
            path = steer(closest.point, point, delta_q)
            point = path[-1]
            # 10 nodes in path
            if check_path_valid(path, state_is_valid):
                # replace node with lower cost node in radius r if it exists
                cost = closest.getCost()+math.dist(closest.point, point)
                for n in node_list:
                    if math.dist(n.point, point) < r:
                        newCost = n.getCost() + math.dist(n.point, point)
                        if newCost < cost:
                            newPath = steer(n.point, point, delta_q)
                            if check_path_valid(newPath, state_is_valid):
                                cost = newCost
                                path = newPath
                                closest = n
                # check nearby nodes if rewiring will produce lower cost
                node = Node(point, parent=closest)
                node.path_from_parent = path
                cost = node.getCost()
                for n in node_list:
                    if math.dist(n.point, point) < r:
                        newCost = cost + math.dist(n.point, point)
                        if newCost < n.getCost():
                            newPath = steer(n.point, point, delta_q)
                            if check_path_valid(newPath, state_is_valid):
                                n.parent = node
                                n.path_from_parent = newPath
                node_list.append(node)
                # check if goal is within reach
                # if goal_point is not None and math.dist(goal_point, path[-1]) < delta_q:
                #     path = steer(path[-1], goal_point, delta_q)
                #     if check_path_valid(path, state_is_valid):
                #         node = Node(goal_point, parent = node)
                #         node.path_from_parent = path
                #         node_list.append(node)
                #         return node_list
                if goal_point is not None and math.dist(goal_point, point) == 0:
                    return node_list
                if state_is_goal is not None:
                    if state_is_goal(point):
                        return node_list
                break
    # if goal_point == None:
    #     return node_list
    # else:
    #     return None
    
    
    # TODO: Your code here
    # TODO: Make sure to add every node you create onto node_list, and to set node.parent and node.path_from_parent for each
    print("No goal given or path not found")
    return node_list