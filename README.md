# CSCI_3302_Final_Project

Final Project for CSCI3302 Introduction to Robotics. An autonomous robot which seeks to autonomously explore the environment, then find, grab, and put desired objects (yellow cubes) in the robot's shopping baskets.

## Team Members

- Nicolas Perrault
- Sean Shi
- Christian Lee

## Techniques

- Computer Vision
    - using color filtering and blob detection in `cv2` to discern objects and then matching with objects in webot's recognition api to get objects relative positions to camera
- Localization
    - odometry with vL and vR
    - webots's api with compass and gps if odometry is too far off
- Mapping
    - Autonomous mapping of obstacles and known space using LiDAR
- Manipulation
    - IK using `ikpy`
    - arm file in `.\final_project\controllers\grocery_shopper\arm.urdf`
- Planning
    - RRT* (adapted from HW2)
    - Modifications to base RRT include path smoothing, returning if a path has been found, other user feedback, and a function parameter to check if a state is a goal as well as the original goal point 

