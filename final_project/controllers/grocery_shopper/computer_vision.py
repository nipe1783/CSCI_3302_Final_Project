import numpy as np
import cv2

def goal_detect():

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