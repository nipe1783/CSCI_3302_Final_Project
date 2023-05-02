# CSCI_3302_Final_Project

Final Project for CSCI3302 Introduction to Robotics. An autonomous robot which seeks to autonomously explore the environment, then find, grab, and put desired objects (yellow cubes) in the robot's shopping baskets.

## Team Members

- Nicolas Perrault
- Sean Shi
- Christian Lee

## Techniques

- Computer Vision
    - various image processing techniques using `cv2` discern objects and webot's recognition api to get objects relative positions
- Localization
    - webots's api
- Mapping
    - Autonomous Lidar mapping
- Manipulation
    - IK using `ikpy`
    - arm file in `.\final_project\controllers\grocery_shopper\arm.urdf`
- Planning
    - RRT* (adapted from HW2)

