import math
import cv2
import numpy as np

LIDAR_SENSOR_MAX_RANGE = 5.5

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 1
params.thresholdStep = 1

params.filterByArea = True
params.minArea = 2

params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
params.minRepeatability = 3
params.filterByColor = False

detector = cv2.SimpleBlobDetector_create(params)

# image dimensions: 135 240

def goal_detect(camera, pose_x, pose_y, height, pose_theta, goal_queue):

    '''
    detects yellow blobs on camera's image. returns location of the centermost object in robot space if detected and updates the list of goals.
    parameters:
    camera: robot camera object
    pose_x, pose_y, height, pose_theta: the current location and orientation of the camer
    goal_queue: the queue of existing goals, passed in to check for similar goals
    returns: 
        object_pos: pose of the centermost object in robot space from the persepective of the camera
        goal_queue: updated queue of goals in the form [[x,y,z], onLeft] where on Left is if the object is on the blue side of the shelf
    '''

    img = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15,200,200])
    upper_yellow = np.array([40,255,255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    blobs = detector.detect(mask)
    recObjects = camera.getRecognitionObjects()
    blob_centerness = math.dist((0,0), (135, 240))
    blob_pt = ()
    object_pos = []
    for object in recObjects:
        pose = object.getPosition()
        # if math.dist((0,0), pose[:2]) > LIDAR_SENSOR_MAX_RANGE:
        #     continue
        for blob in blobs:
            # blob.pt is the center of the blob, this statement checks if the recognition api and camera are seeing the same thing
            if math.dist(object.getPositionOnImage(), blob.pt) < 3:
                wx =  math.cos(pose_theta)*pose[0] - math.sin(pose_theta)*pose[1] + pose_x
                wy =  math.sin(pose_theta)*pose[0] + math.cos(pose_theta)*pose[1] + pose_y
                wz = height+pose[2]
                # print("wx", wx, "wy", wy, "wz", wz)
                # The code below is because of the same reason why the lidar sensor had to be moved 0.5 m up.
                if wz < 0.85:
                    wz = 0.53
                else:
                    wz = 1.03
                goal = np.array([wx, wy, wz])
                if math.dist(blob.pt, (67, 120)) < blob_centerness:
                    object_pos = pose
                    blob_pt = blob.pt
                    blob_centerness = math.dist(blob.pt, (67, 120))
                # Checking if the robot has seen an object before
                goalNew = True
                for i, gl in enumerate(goal_queue):
                    if(math.dist(goal[:2], gl[0][:2])< 1.1) and abs(goal[2]-gl[0][2])< 0.1:
                        goalNew = False
                        goal_queue[i][0] = [0.5*(gl[0][0] + goal[0]), 0.5*(gl[0][1] + goal[1]), goal[2]]
                        # print(goal_queue[i])
                        break
                if goalNew:
                    # Note: additional variable is True if object is on the robot's left side ie left of shelf and false if on right
                    if wy < pose_y:
                        goal_queue.append([goal, True])
                    else:
                        goal_queue.append([goal, False])
                    # loggin
                    print("New goal found at: ", goal, " goals total: ", len(goal_queue))
    return object_pos, goal_queue, blob_pt