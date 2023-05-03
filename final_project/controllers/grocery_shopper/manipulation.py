import math
import numpy as np
import ikpy

def manipulate_to(new_pose, robot_parts):
    """Use IK to calculate position and then deliver position to joints

        Parameters
        ----------
        target_position: array
            3 value array with desired position
        target_Orientation: numpy.array
            An optional 3x3 array for orientation at destination
        ---------
        Returns
        Null
    """
    arm_joints = new_pose
    # print(arm_joints)
    robot_parts["arm_1_joint"].setPosition(arm_joints[4])
    robot_parts["arm_2_joint"].setPosition(arm_joints[5])
    robot_parts["arm_3_joint"].setPosition(arm_joints[6])
    robot_parts["arm_4_joint"].setPosition(arm_joints[7])
    robot_parts["arm_5_joint"].setPosition(arm_joints[8])
    robot_parts["arm_6_joint"].setPosition(arm_joints[9])
    robot_parts["arm_7_joint"].setPosition(arm_joints[10])

    return robot_parts


def ik_arm(target_position, my_chain, arm_joints, target_orientation=None, orientation_mode = None, initial = None, angle = None):
    if initial is None:
        initial = arm_joints
    # else:
        # target_frame[:3, :3] = [[0,1,0],[1,0,0],[0,0,1]]
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