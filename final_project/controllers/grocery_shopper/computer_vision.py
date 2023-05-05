import math
import cv2
import numpy as np


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

def goal_detect(camera, pose_x, pose_y, height, pose_theta, goal_queue):

    '''
    detects yellow blob on camera. returns location of blob on image and if there is a blob detected.

    camera: robot camera object

    returns: 
        gx: blob x location on camera. 
        gy: blob y location on camera. 
        bool: if blob is detected.
    '''

    img = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15,200,200])
    upper_yellow = np.array([40,255,255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    blobs = detector.detect(mask)  

    recObjects = camera.getRecognitionObjects()
    for blob in blobs:
        for object in recObjects:
            if math.dist(object.getPositionOnImage(), blob.pt) < 3:
                pose = object.getPosition()
                wx =  math.cos(pose_theta)*pose[0] - math.sin(pose_theta)*pose[1] + pose_x
                wy =  math.sin(pose_theta)*pose[0] + math.cos(pose_theta)*pose[1] + pose_y
                wz = height+pose[2]
                if wz < 0.7:
                    wz = 0.575
                else:
                    wz = 1.075
                goal = [wx, wy, wz]
                goalNew = True
                for gl in goal_queue:
                    if(math.dist(goal, gl)):
                        goalNew = False
                        break
                if goalNew:
                    goal_queue.append(goal)
    if recObjects:
        recObject = recObjects[0]
        return recObject.getPosition(), goal_queue
    else:
        return (), goal_queue
    