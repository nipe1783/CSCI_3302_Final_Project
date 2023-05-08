import math
import numpy as np
import ikpy
import ikpy.chain

active_links =  [False, False, False, False,  True, True, True, True, True, True, True, False, False]
my_chain = ikpy.chain.Chain.from_urdf_file("arm.urdf", active_links_mask=active_links)

def manipulate_to(arm_joints, robot_parts):
    """Deliver a given position calculated by ik_arm to joints

        Parameters
        ----------
        arm_joints: array 14 elements
            an point in joint space for the robot to set the arm to
        robot_parts: dictionary
            the containter for all of the robot parts, for use in setting positions
        ---------
        Returns
        Null
    """
    # print(arm_joints)
    robot_parts["arm_1_joint"].setPosition(arm_joints[4])
    robot_parts["arm_2_joint"].setPosition(arm_joints[5])
    robot_parts["arm_3_joint"].setPosition(arm_joints[6])
    robot_parts["arm_4_joint"].setPosition(arm_joints[7])
    robot_parts["arm_5_joint"].setPosition(arm_joints[8])
    robot_parts["arm_6_joint"].setPosition(arm_joints[9])
    robot_parts["arm_7_joint"].setPosition(arm_joints[10])

    return robot_parts

def ik_arm(target_position, initial, target_orientation=None, orientation_mode = None, angle = None):
    """Use IK to calculate position
        Parameters
        ----------
        target_position:
            a point in robot space for the arm to move to
        initial: array 14 elements
            the initial position of the arm. This cannot be none because ikpy will default to all zeros which is not a valid initial position
        angle: float
            If given, it is the angle that is desired for the hand in robot space when it reaches its goal
        target_orientation: numpy.array
            An optional 3x3 array for orientation at destination, if given, will default to orientation mode "all"
        orientation_mode: string
            See ikpy for more details
        ---------
        Returns
        Null
    """
    if angle is not None:
        angle = math.pi/2-angle
        rotate = np.array([[1,0,0],
                            [0, math.cos(-math.pi/2), math.sin(-math.pi/2)],
                            [0, -math.sin(-math.pi/2), math.cos(-math.pi/2)]])
        orientation = np.array([[math.cos(angle), math.sin(angle), 0],
                                [-math.sin(angle), math.cos(angle), 0],
                                [0,0,1]])
        target_orientation = np.dot(orientation,rotate)
        return my_chain.inverse_kinematics(target_position,target_orientation=target_orientation, orientation_mode="all" , initial_position=initial)
    elif target_orientation is not None:
        # print("orientation given")
        if orientation_mode is None:
            orientation_mode = "all"
        return my_chain.inverse_kinematics(target_position,target_orientation=target_orientation, orientation_mode=orientation_mode , initial_position=initial)
    else:
        return my_chain.inverse_kinematics(target_position, initial_position=initial)
    
def get_position(pose_arm):
    'returns position of wrist in robot coa'
    frame = my_chain.forward_kinematics(pose_arm)
    pose = frame[:3, 3]
    return pose