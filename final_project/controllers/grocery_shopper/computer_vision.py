import numpy as np
import cv2
import math

def goal_detect(camera):

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

    # apply Gaussian Blur
    smoothed = cv2.GaussianBlur(mask, (0,0), sigmaX=1.5, sigmaY=1.5, borderType = cv2.BORDER_DEFAULT)
    
    # Apply a morphological opening to remove noise and small objects
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

    # Find the contours of the remaining blobs in the image
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # CODE FOR SHOWING OBJECTS:
    
    # Draw the contours on a copy of the original image
    smoothed_copy = smoothed.copy()
    cv2.drawContours(smoothed, contours, -1, (0, 255, 0), 2)

    # Identify the center of the blob by calculating the centroid of the contour

    filtered_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            filtered_contours.append(c)

    for c in filtered_contours:
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(smoothed_copy, (cx, cy), 5, (0, 0, 255), -1)

    if len(filtered_contours) > 0:

        # location of first goal detected
        c = filtered_contours[0]
        M = cv2.moments(c)
        gx = int(M['m10'] / M['m00'])
        gy = int(M['m01'] / M['m00'])
        return gx, gy, True
    else:
        return -1, -1, False
    
def goal_state(camera, pose_x, pose_y, pose_theta):
    yellow = [255.0, 255.0, 0.0]
    for object in camera.getRecognitionObjects():
        color = object.getColors()
        color[0] = color[0]*255
        color[1] = color[1]*255
        color[2] = color[2]*255
        # print(color[0], color[1], color[2])
        if (color[0] == yellow[0] and color[1] == yellow[1] and color[2] == yellow[2]):
            pose = object.getPosition()
            wx =  math.cos(pose_theta)*pose[0] - math.sin(pose_theta)*pose[1] + pose_x
            wy =  math.sin(pose_theta)*pose[0] + math.cos(pose_theta)*pose[1] + pose_y
            return [wx, wy]