import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


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
                if wz < 0.85:
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
    
def finger_detect(camera):

    '''
    takes in camera object and returns direction (left, right) that robot arm needs to move.
    '''
    yellow = [255.0, 255.0, 0.0]
    for object in camera.getRecognitionObjects():
        color = object.getColors()
        # print(color[0], color[1], color[2])
        if (color[0]*255 == yellow[0] and color[1]*255 == yellow[1] and color[2]*255 == yellow[2]):
            goal_camera_position = object.getPositionOnImage()


    img = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_grey = np.array([0,0,0])
    upper_grey = np.array([55,55,55])

    # create a mask for the pixels in the desired color range
    mask = cv2.inRange(hsv, lower_grey, upper_grey)

    # apply the mask to the original image
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    mask = cv2.inRange(hsv, lower_grey, upper_grey)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5  # adjust this value to change the minimum blob size
    params.maxArea = 100  # adjust this value to change the maximum blob size
    detector = cv2.SimpleBlobDetector_create(params)
    blobs = detector.detect(mask)

    # define the region of interest
    x1, x2, y1, y2 = 100, 150, 80, 125

    # define the minimum and maximum blob size (area)
    min_area = 1
    max_area = 100

    # loop through all the blobs and filter them based on the criteria
    for blob in blobs:
        x, y = blob.pt
        size = blob.size
        area = blob.size**2
        if (x >= x1 and x <= x2 and y >= y1 and y <= y2 and area >= min_area and area <= max_area):
            # draw a circle around the blob
            print("yes")
            cv2.circle(masked_img, (int(x), int(y)), int(size), (0, 255, 0), thickness=2)

    # display the masked image
    plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGRA2RGB))
    plt.show()


    