# CSCI_3302_Final_Project

Final Project for CSCI3302 Introduction to Robotics. An autonomous robot which seeks to autonomously explore the environment, then find, grab, and put desired objects (yellow cubes) in the robot's shopping baskets.

*Note: Much of the code for the sections listed below is under corresponding files other than the main grocer_shopper.py*

[Link to demonstration video](https://www.youtube.com/watch?v=xDbsR8DpBQw&ab_channel=NicPerrault)

## Team Members

- Nicolas Perrault
- Sean Shi
- Christian Lee

## Techniques

- Computer Vision
    - using color filtering and blob detection in `cv2` to discern objects and get centers and then matching with objects in webot's recognition api to get objects relative positions to camera
- Localization
    - odometry with vL and vR
    - webots's api with compass and gps for correction if odometry is too far off
- Mapping
    - Autonomous mapping of obstacles and known space using LiDAR
    - Originally used manual mapping, but this is no longer used
- Manipulation
    - IK using `ikpy`
    - arm file in `.\final_project\controllers\grocery_shopper\arm.urdf`
- Planning
    - RRT* (adapted from HW2)
    - Modifications to base RRT include path smoothing, returning if a path has been found, some feedback in the form of print statements, and a function parameter to check if a state is a goal as well as the original goal point.
    - If you want to see rrt running for the robot, uncomment visualize_path

