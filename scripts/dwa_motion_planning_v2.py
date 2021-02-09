#!/usr/bin/env python
from my_simulations.msg import uwbLocalization
from my_simulations.msg import velocityCtrl
from my_simulations.msg import planedPath
from my_simulations.msg import goalPos
from my_simulations.msg import odometryDualObs
from sensor_msgs.msg import LaserScan
from my_simulations.msg import lidarObstacles
from my_simulations.msg import mapInfor
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import time
import rospy
import libconf
import io
import rospkg
from lidar_simulate import *
from mappingFunc import *
from dwa_occ_avoid import DynamicWindowApproach
import pickle

rospack = rospkg.RosPack()
config_file = rospack.get_path('my_simulations') + '/config/system.cfg'
with io.open(config_file, 'r', encoding='utf-8') as f:
    config = libconf.load(f)
isSimulation = config.default.gazebo.simulation

if isSimulation:
    odoMsg = Odometry()
else:
    odoMsg = odometryDualObs()

uwbMsg = uwbLocalization()
goalMsg = goalPos()
rplidarMsg = LaserScan()
mappingMsg = mapInfor()

setNewGoal = False
goalInit = False
updateMapping = False
pre_goal = np.zeros((1,2))
oneMeterPixels = 100.0

def robot_rplidar_obs_msg_callback(msg):
    global rplidarMsg
    rplidarMsg = msg

def robot_odo_obs_msg_callback(msg):
    global odoMsg
    odoMsg = msg

def robot_uwb_obs_msg_callback(msg):
    global uwbMsg
    uwbMsg = msg

def robot_mapping_msg_callback(msg):
    global mappingMsg, updateMapping
    if updateMapping:
        mappingMsg = msg

def f(x, u, dt):
    F = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    B = np.array( [ [dt*np.cos(x[2]), 0], [dt*np.sin(x[2]), 0], [0, dt], [1, 0], [0, 1]])
    x = np.dot(F, x.reshape(5,1)) + np.dot(B, u)
    return x.flatten()

def robot_goal_obs_msg_callback(msg):
    global goalMsg, goalInit
    goalMsg = msg
    if (goalMsg.goal_posx < 0.1 and goalMsg.goal_posy < 0.1) or (np.isnan(goalMsg.goal_posx) or np.isnan(goalMsg.goal_posy)):
        goalInit = False
    else:
        goalInit = True

def publishVel(pub, velMsg, v_l, v_r, u):
    velMsg.timestamp = rospy.Time.now()
    velMsg.wheel_vel_left = v_l
    velMsg.wheel_vel_right = v_r
    if u is not None:
        velMsg.vt = u[0]
        velMsg.ot = u[1]
    else:
        velMsg.vt = 0.
        velMsg.ot = 0.
    pub.publish(velMsg)

def publishVelSimulate(pub, velMsg, u, curAngle):
    if u is not None:
        velMsg.linear.x = u[0]
        velMsg.angular.z = u[1]

        print "vel: ", velMsg.linear.x, " omega: ", velMsg.angular.z
    else:
        velMsg.linear.x = 0
        velMsg.angular.z = 0
    pub.publish(velMsg)

def pickRefPos(iter, curPos, refPath, minDist, flag):
    if flag == 1: #init
        cumDist = 0
    else:
        cumDist = np.sqrt((curPos[0]-refPath[iter,0])**2 + (curPos[1]-refPath[iter,1])**2 )
    # print refPath
    while cumDist < minDist and iter < refPath.shape[0]-1:
        dist = np.sqrt((refPath[iter+1, 0]-refPath[iter, 0])**2+(refPath[iter+1, 1]-refPath[iter, 1])**2)
        # print dist,
        cumDist += dist
        iter += 1
    refPos = refPath[iter, :]
    return refPos, iter

def dwaControl(iter, start_pos, followPath, minDist, myPos_1, curAngle, linearVel, angularVel, dwa, \
    obstacles, lidarData):

    #myPos_1 is in 5 cm
    if iter == 0:
        [refPos, iter] = pickRefPos(iter, start_pos, followPath, minDist, 1)
    else:
        # print(iter, followPath.shape[0]-1)
        if iter <= followPath.shape[0]-1:
            [refPos, iter] = pickRefPos(iter, myPos_1[0:2], followPath, minDist, 2)
    # print 'refPos:', refPos, ' iter:', iter, myPos_1[0:2], followPath[-1, :]
    robotState = np.zeros((5,1)).flatten()
    robotState[0:2] = myPos_1 #myPos_1 is in meter
    robotState[2] = curAngle
    # robotState[4] = (odoMsg.right_velocity - odoMsg.left_velocity)/(oneMeterPixels*fWheelDistance)
    # robotState[3] = (odoMsg.right_velocity + odoMsg.left_velocity)/(oneMeterPixels*2)
    robotState[4] = angularVel
    robotState[3] = linearVel

    [u, traj] = dwa.dwaMethod(robotState, refPos, obstacles, lidarData, curAngle)
    return u, iter, traj, refPos

def getLidarInfo(rplidarMsg, lidarIndex, theta, curPos, angleIter, \
    lidar_offsetx, lidar_offsety, lidarObs):
    ranges = np.array([rplidarMsg.ranges]).flatten()
    ranges = ranges[lidarIndex]
    angle_array = angleIter #+ uwbMsg.theta
    filter = np.logical_or(np.isinf(ranges), ranges<0.1)
    #prepare data for dwa
    lidarData = ranges.copy()  #lidardata in meter
    lidarData[filter] = 0.
    lidarData[lidarData > 50] = 0.

    ranges = ranges[np.logical_not(filter)]
    angle_array = angle_array[np.logical_not(filter)]
    # theta = theta - np.pi/2
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    off_x = np.cos(angle_array) * ranges + lidar_offsety # in m
    off_y = np.sin(angle_array) * ranges - lidar_offsetx

    obstaclesToRobot = np.vstack((off_x, off_y))
    obstacles = np.dot(R, obstaclesToRobot) + np.tile(np.array([[curPos[0]], [curPos[1]]]), \
        (1, obstaclesToRobot.shape[1])) # in global coordinate, 1cm

    lidarObs.obstacle_x = obstacles[0, :]
    lidarObs.obstacle_y = obstacles[1, :]
    obstacles = obstacles.T

    return lidarData, obstacles, lidarObs

def stopRobot(linearVel, angleVel, fWheelDistance, stopSpeed):

    stoppedFlag = False
    u = np.zeros((2, 1)).flatten()
    if np.abs(linearVel) < 0.1 and np.abs(angleVel) < (5 * np.pi / 180.):
        stoppedFlag = True
        v_l = 0.
        v_r = 0.
    else:
        linearVel = linearVel * stopSpeed
        angleVel = angleVel * stopSpeed
        v_l = linearVel - (fWheelDistance/2)*angleVel
        v_r = linearVel + (fWheelDistance/2)*angleVel
        u[0] = linearVel
        u[1] = angleVel
    return v_l, v_r, stoppedFlag, u

def listener():
    global setNewGoal, recPathFlag, uwbMsg, goalMsg, rplidarMsg, goalInit, config, isSimulation

    uwbTopic = config.default.uwb_localization.aMessageTopic
    goalTopic = config.default.robot_set_goal.aMessageTopic
    pathTopic = config.default.robot_planned_path.aMessageTopic
    velTopic = config.default.robot_vel_ctrl.aMessageTopic
    
    lidarTopic = config.default.hokuyolidar.aMessageTopic
    obstaclesTopic = config.default.robot_obstacles.aMessageTopic
    mappingTopic = config.default.lidarMapping.aMessageTopic


    oneMeterPixels = config.default.sRobotPara.oneMeterPixels
    lidar_offsetx = config.default.sRobotPara.lidar_offsetx
    lidar_offsety = config.default.sRobotPara.lidar_offsety
    fWheelDistance = config.default.sRobotPara.fWheelDistance
    env = config.default.exploration.env

    rospy.init_node('core_algorithm_model', anonymous=False)
    rospy.Subscriber(uwbTopic, uwbLocalization, robot_uwb_obs_msg_callback)
    rospy.Subscriber(goalTopic, goalPos, robot_goal_obs_msg_callback)
    if isSimulation:
        odoTopic = config.default.robot_encoder.simulateMessageTopic
        rospy.Subscriber(odoTopic, Odometry, robot_odo_obs_msg_callback)
    else:
        odoTopic = config.default.robot_encoder.realMessageTopic
        rospy.Subscriber(odoTopic, odometryDualObs, robot_odo_obs_msg_callback)

    rospy.Subscriber(lidarTopic, LaserScan, robot_rplidar_obs_msg_callback)
    rospy.Subscriber(mappingTopic, mapInfor, robot_mapping_msg_callback)
    pathPub = rospy.Publisher(pathTopic, planedPath, queue_size = 20)
    
    obsPub = rospy.Publisher(obstaclesTopic, lidarObstacles, queue_size = 20)
    velMsg = velocityCtrl()
    if isSimulation:
        controlCmdTopic = config.default.robot_vel_ctrl.aSimulateMessTopic
        velPub = rospy.Publisher(controlCmdTopic, Twist, queue_size = 20)
        statePub = rospy.Publisher(velTopic, velocityCtrl, queue_size = 20)
        velCmdMsg = Twist()
    else:
        velPub = rospy.Publisher(velTopic, velocityCtrl, queue_size = 20)
    pathMsg = planedPath()
    lidarObs = lidarObstacles()

    # parameter for path planning, the resolution is 10cm
    fps = 10.0
    deltaT = 1 / fps
    reach_flag = False

    #parameter for controller
    reach_dist = 2.5 # in meter, 1.5 for indoor, 2.5 for outdoor
    #parameter for dwa avoidance
    obstacleR = 1.5
    near_goal_dist = 4.5
    # Kinematic = np.array([0.6, 50.0*np.pi/180.0, 0.6, 30.0*np.pi/180.0, 0.015, 1*np.pi/180.0])
    # evalParamNomal = np.array([0.35, 0.25, 0.25, 0.]) #heading dist vel angular
    # evalParamNearGoal = np.array([1.5, 0.15, 0.1, 0.00])
    # evalParamNearObs = np.array([0., 0.2, 0.0, 0.])

    if env == 2:
        ## garden
        Kinematic = np.array([0.7, 50.0*np.pi/180.0, 0.4, 60.0*np.pi/180.0, 0.015, 2*np.pi/180.0])
        evalParamNomal = np.array([0.15, 0.2, 0.18, 0.12])
        evalParamNearGoal = np.array([0.4, 0.2, 0.1, 0.00])
        evalParamNearObs = np.array([0., 0.2, 0.0, 0.3])
    elif env == 2:
        ##workshop
        Kinematic = np.array([0.7, 50.0*np.pi/180.0, 0.6, 80.0*np.pi/180.0, 0.015, 2*np.pi/180.0])
        evalParamNomal = np.array([0.3, 0.2, 0.18, 0.2])
        evalParamNearGoal = np.array([0.5, 0.2, 0.1, 0.00])
        evalParamNearObs = np.array([0., 0.2, 0.0, 0.3])

    virtualAppTime = 3
    dwa = DynamicWindowApproach(evalParamNomal, Kinematic, obstacleR, deltaT, virtualAppTime, 60)

    if isSimulation:
        publishVelSimulate(velPub, velCmdMsg, None, 0)
        statePub.publish(velMsg)
    else:
        publishVel(velPub, velMsg, 0., 0., None)
    print("waiting for uwb initialization")
    while (uwbMsg.state != 2) and not rospy.is_shutdown():
        time.sleep(0.1)
    print("uwb initialized")
    # time.sleep(5)
    rate = rospy.Rate(fps)  # 10hz

    ## init the motion planning model
    goal_pos = np.zeros((1,2))
    start_pos = np.zeros((1,2))
    showPoints = 0
    iter = 0
    resolution = 5.0
    minDist = 3 #4.5m
    goingOn = False

    lidarIndex = np.arange(180, 900, 4).astype(int)
    angleIter = np.arange(-np.pi /2., np.pi/2., np.pi/180.)

    movingFlag = True
    stopCnt = 0
    stopFrame = 10
    stopDist = 1.1 # in m
    obsCtAngular = np.zeros((stopFrame, 1))
    threshold = 1.0
    robotStateIdx = 0 # 0 angular adjust 1 normal control, 2 stop, 3 truning
    v_turning = (50*np.pi/180) * fWheelDistance / 2
    turning_threshold = 16 * np.pi / 180
    stopSpeed = 0.5
    begin_pathplan = False
    myInitPos = np.array([uwbMsg.robotPosx, uwbMsg.robotPosy])

    ## initTheta = true for workshop, initTheta = false for garden
    if env == 1: ## garden
        initTheta = False
    elif env == 2: ## workshop
        initTheta = True

    while not rospy.is_shutdown():
        velMsg.state = 1 ## 1 for normal control, 2 for stop, 3 for reach destination, 4 for emgency stop, 5 for turning
        if mappingMsg.state != 0:
            curAngle = mappingMsg.pos[-1]
            mappingMsg.state = 0
        else:
            curAngle = uwbMsg.theta

        if curAngle > np.pi:
            curAngle = curAngle - 2*np.pi
        if curAngle < -np.pi:
            curAngle = curAngle + 2*np.pi
        # For changing between simulation and real robot. 
        if isSimulation:
            linearVel = np.sqrt(odoMsg.twist.twist.linear.x**2 + odoMsg.twist.twist.linear.x**2)
            angularVel = odoMsg.twist.twist.angular.z
        else:
            linearVel = (odoMsg.right_velocity + odoMsg.left_velocity)/(oneMeterPixels*2)
            angularVel = (odoMsg.right_velocity - odoMsg.left_velocity)/(oneMeterPixels*fWheelDistance)

        if uwbMsg.state == 3:
            [v_l, v_r, stoppedFlag, u] = stopRobot(linearVel, angularVel, fWheelDistance, stopSpeed)
            if stoppedFlag:
                velMsg.state = 2
                print 'some needed nodes is missing, waiting to reconect them!'
                goingOn = False
            
            if isSimulation:
                # u = -u
                publishVelSimulate(velPub, velCmdMsg, u, curAngle)
                statePub.publish(velMsg)
            else:
                publishVel(velPub, velMsg, v_l, v_r, u)
            rate.sleep()
        if goalMsg.stateFlag == 2: #set new goal
            v_l = 0.
            v_r = 0.
            velMsg.state = 2
            if isSimulation:
                publishVelSimulate(velPub, velCmdMsg, None, 0)
                statePub.publish(velMsg)
            else:
                publishVel(velPub, velMsg, v_l, v_r, None)
            goal_pos[0, 0] = goalMsg.goal_posx / oneMeterPixels * resolution
            goal_pos[0, 1] = goalMsg.goal_posy / oneMeterPixels * resolution
            start_pos[0, 0] = goalMsg.start_pos[0] / oneMeterPixels * resolution
            start_pos[0, 1] = goalMsg.start_pos[1] / oneMeterPixels * resolution
            pathLen = len(goalMsg.path_posx)
            followPath = np.zeros((pathLen, 2))
            followPath[:, 0] = np.array(goalMsg.path_posx) / oneMeterPixels * resolution # in meter
            followPath[:, 1] = np.array(goalMsg.path_posy) / oneMeterPixels * resolution
            if initTheta == False:
                initTheta = True
                target_theta = 0
                print 'INFO: target theta: ', target_theta
            else:
                target_theta = np.arctan2( goalMsg.path_posy[1] - goalMsg.start_pos[1], goalMsg.path_posx[1] - goalMsg.start_pos[0] )

            robotStateIdx = 0
            iter = 0
            goingOn = True
            goalMsg.stateFlag = 1
            showPoints = followPath.shape[0]
            goalLocation = goal_pos.flatten()
            mapOffset = goalMsg.mapOffset
            minDistToObs = 1000
            # print 'goal_pos:', goal_pos
            print 'begin to navigate the robot to the given destination!'
        elif goalMsg.stateFlag == 3: # emergency stopping
            [v_l, v_r, stoppedFlag, u] = stopRobot(linearVel, angularVel, fWheelDistance, stopSpeed)
            if stoppedFlag:
                goalMsg.stateFlag = 1
                velMsg.state = 4
                print 'received stop command, the robot will be stopped!'
            if isSimulation:
                # u = -u
                publishVelSimulate(velPub, velCmdMsg, u, curAngle)
                statePub.publish(velMsg)
            else:
                publishVel(velPub, velMsg, v_l, v_r, u)
            goingOn = False

        if goingOn == True: #normal processing
            if mappingMsg.state != 0:
                myPos_1 = np.array(mappingMsg.posEstMean) * resolution / oneMeterPixels # in global coordinate, m
                mappingMsg.state = 0
            else:
                myPos_1 = np.array([uwbMsg.robotPosx + mapOffset[0], uwbMsg.robotPosy + mapOffset[1]]) * resolution / oneMeterPixels# in global coordinate, m

            if robotStateIdx == 0: #turning
                diff_theta = curAngle - target_theta
                if diff_theta > np.pi:
                    diff_theta = - diff_theta + 2*np.pi
                if diff_theta < -np.pi:
                    diff_theta = diff_theta + 2*np.pi
                if np.abs(diff_theta) > turning_threshold:
                    w_v = -np.sign(diff_theta) * v_turning
                    velMsg.state = 5
                    up = [0, w_v]
                    if isSimulation:
                        publishVelSimulate(velPub, velCmdMsg, up, 0)
                        statePub.publish(velMsg)
                    else:
                        v_l = - w_v * fWheelDistance / 2
                        v_r =  w_v * fWheelDistance / 2
                        publishVel(velPub, velMsg, v_l, v_r, None)
                else:
                    # stoppedFlag = True
                    [v_l, v_r, stoppedFlag, u] = stopRobot(linearVel, angularVel, fWheelDistance, stopSpeed)
                    if stoppedFlag:
                        velMsg.state = 2
                        velMsg.intiMapping = 2
                        robotStateIdx = 1
                        print('state 0 -> state 1')
                    if isSimulation:
                        # u[0] = -u[0]
                        publishVelSimulate(velPub, velCmdMsg, u, curAngle)
                        statePub.publish(velMsg)
                    else:
                        publishVel(velPub, velMsg, v_l, v_r, u)
                rate.sleep()
                continue

            elif robotStateIdx == 1:
                #deal with lidar data
                if len(rplidarMsg.ranges) > 0:

                    [lidarData, obstacles, lidarObs] = getLidarInfo(rplidarMsg, lidarIndex, curAngle, \
                        myPos_1, angleIter, lidar_offsetx, lidar_offsety, lidarObs)
                    [stopFlag, minDistToObs] = dwa.stopCases(obstacles, myPos_1, stopDist)

                    if stopFlag == True:
                        stopSpeed_e = stopSpeed * 1.5
                        [v_l, v_r, stoppedFlag, u] = stopRobot(linearVel, angularVel, fWheelDistance, stopSpeed_e)
                        if stoppedFlag:
                            velMsg.state = 2
                            robotStateIdx = 2 # near the obstacles
                            print('state 1 -> state 2')
                            if isSimulation:
                                # u = -u
                                publishVelSimulate(velPub, velCmdMsg, u, curAngle)
                                statePub.publish(velMsg)
                            else:
                                publishVel(velPub, velMsg, v_l, v_r, u)                     
                        stopCnt = 0
                        rate.sleep()
                        continue
                    else:
                        [u, iter, traj, refPos] = dwaControl(iter, start_pos, followPath, minDist,\
                             myPos_1, curAngle, linearVel, angularVel, dwa, obstacles, lidarData)
                        v_f = u[0]
                        w_v = u[1]
                        v_l = v_f - (fWheelDistance/2)*w_v
                        v_r = v_f + (fWheelDistance/2)*w_v
                        error = np.linalg.norm(myPos_1 - goalLocation) # in m
                        #if the distance between reference and current position is larger than a threshold distance, then choose an optimal position as reference
                        print 'Infor: error ', error, ', reach_dist ', reach_dist
                        if error < reach_dist:# reach the destination and set the reach flag to True, which will then send zero speed cmd to wheelchair
                            print('reach the destination')
                            robotStateIdx = 6 # reach the destination
                            print('state 1 -> state 6')
                        elif error < near_goal_dist:
                            robotStateIdx = 5
                            print('state 1 -> state 5')
                            dwa.setEvalParam(evalParamNearGoal)
                            if isSimulation:
                                # u = -u
                                publishVelSimulate(velPub, velCmdMsg, u, curAngle)
                                statePub.publish(velMsg)
                            else:
                                publishVel(velPub, velMsg, v_l, v_r, u)       
                        else: # normal state
                            pathPub.publish(pathMsg)
                            if isSimulation:
                                # u = -u
                                publishVelSimulate(velPub, velCmdMsg, u, curAngle)
                                velMsg.vt = u[0]
                                velMsg.ot = u[1]
                                statePub.publish(velMsg)
                            else:
                                publishVel(velPub, velMsg, v_l, v_r, u)       
                        # all state will publish obstacles data
                        lidarObs.timestamp = rospy.Time.now()
                        lidarObs.traj_x = traj[0, :]
                        lidarObs.traj_y = traj[1, :]
                        lidarObs.refPos = refPos.flatten()
                        obsPub.publish(lidarObs)
                        rate.sleep()
                        continue
                else:
                    velMsg.state = 2
                    robotStateIdx = 7
                    print('state 1 -> state 7')
                    print('cannot receive lidar data')
                    rate.sleep()
                    continue
            elif robotStateIdx == 2: # stop
                if len(rplidarMsg.ranges) > 0:
                    [lidarData, obstacles, lidarObs] = getLidarInfo(rplidarMsg, lidarIndex,\
                         curAngle, myPos_1, angleIter, lidar_offsetx, lidar_offsety, lidarObs)
                    [stopFlag, minDistToObs] = dwa.stopCases(obstacles, myPos_1, stopDist)#to overcome the noise data from lidar
                    if stopFlag == True:
                        if stopCnt == stopFrame:
                            robotStateIdx = 3
                            stopCnt = 0
                            print('state 2 -> state 3')
                        else:
                            obsCtAngular[stopCnt, 0] = dwa.getObsMeanAngular(lidarData, stopDist) # record the angular information of obs to predict the moving state of robot
                            stopCnt += 1
                            print('state 2 -> state 2')
                    else:
                        robotStateIdx = 1
                        print('state 2 -> state 1')
                else:
                    robotStateIdx = 7
                    print('state 2 -> state 7')
                    print('cannot receive lidar data')
                lidarObs.timestamp = rospy.Time.now()
                obsPub.publish(lidarObs)
                velMsg.state = 2
                if isSimulation:
                    publishVelSimulate(velPub, velCmdMsg, None, 0)
                    statePub.publish(velMsg)
                else:
                    publishVel(velPub, velMsg, v_l, v_r, None)       
                rate.sleep()
                continue
            elif robotStateIdx == 3: ## static or dynamic obstacle
                movingFlag = dwa.isMoving(obsCtAngular, threshold)
                if movingFlag == True:
                    robotStateIdx = 2
                    print('state 3 -> state 2')
                else:
                    robotStateIdx = 4
                    dwa.setEvalParam(evalParamNearObs) # set para for near obstacles
                    print('state 3 -> state 4')
                rate.sleep()
                continue
            elif robotStateIdx == 4: # turning
                if len(rplidarMsg.ranges) > 0:
                    [lidarData, obstacles, lidarObs] = getLidarInfo(rplidarMsg, lidarIndex, \
                        curAngle, myPos_1, angleIter, lidar_offsetx, lidar_offsety, lidarObs)

                    [u, iter, traj, refPos] = dwaControl(iter, start_pos, followPath, minDist, \
                        myPos_1, curAngle, linearVel, angularVel, dwa, obstacles, lidarData)
                    w_v = np.sign(u[1]) * v_turning
                    v_l = - v_turning * fWheelDistance / 2
                    v_r =  v_turning * fWheelDistance / 2
                    [stopFlag, mDist] = dwa.stopCases(obstacles, myPos_1, stopDist+0.3)
                    if stopFlag == False:
                        [v_l, v_r, stoppedFlag, u] = stopRobot(linearVel, angularVel, fWheelDistance, stopSpeed)
                        if stoppedFlag:
                            robotStateIdx = 1 # near the obstacles
                            dwa.setEvalParam(evalParamNomal)
                            print('state 4 -> state 1')
                        if isSimulation:
                            # u = -u
                            publishVelSimulate(velPub, velCmdMsg, u, curAngle)
                            velMsg.vt = u[0]
                            velMsg.ot = u[1]
                            statePub.publish(velMsg)
                        else:
                            publishVel(velPub, velMsg, v_l, v_r, u)       
                    else:
                        if isSimulation:
                            # u = -u
                            publishVelSimulate(velPub, velCmdMsg, u, curAngle)
                            velMsg.vt = u[0]
                            velMsg.ot = u[1]
                            statePub.publish(velMsg)
                        else:
                            publishVel(velPub, velMsg, v_l, v_r, u)       
                        print('state 4 -> state 4')
                    lidarObs.traj_x = traj[0, :]
                    lidarObs.traj_y = traj[1, :]
                    lidarObs.refPos = refPos.flatten()
                else:
                    robotStateIdx = 7
                    print('state 4 -> state 7')
                    print('cannot receive lidar data')
                lidarObs.timestamp = rospy.Time.now()
                obsPub.publish(lidarObs)
                rate.sleep()
                continue
            elif robotStateIdx == 5:
                print('state 5 -> state 1')
                dwa.setEvalParam(evalParamNearGoal)
                robotStateIdx = 1
                continue
            elif robotStateIdx == 6:
                print('state 6 -> state 8')
                [v_l, v_r, stoppedFlag, u] = stopRobot(linearVel, angularVel, fWheelDistance, stopSpeed)
                if stoppedFlag:
                    robotStateIdx = 8 # near the obstacles
                    print('reach the destination!')
                    velMsg.state = 3
                if isSimulation:
                    # u = -u
                    publishVelSimulate(velPub, velCmdMsg, u, curAngle)
                    velMsg.vt = u[0]
                    velMsg.ot = u[1]
                    statePub.publish(velMsg)
                else:
                    publishVel(velPub, velMsg, v_l, v_r, u)       
                rate.sleep()
                continue
            elif robotStateIdx == 7:
                [v_l, v_r, stoppedFlag, u] = stopRobot(linearVel, angularVel, fWheelDistance, stopSpeed)
                if isSimulation:
                    # u = -u
                    publishVelSimulate(velPub, velCmdMsg, u, curAngle)
                    velMsg.vt = u[0]
                    velMsg.ot = u[1]
                    statePub.publish(velMsg)
                else:
                    publishVel(velPub, velMsg, v_l, v_r, u)       
                pathPub.publish(pathMsg)
                print('exception handler')
                rate.sleep()
                continue
            elif robotStateIdx == 8:
                print('waiting for new destination!')
                goingOn = False
                rate.sleep()
                continue
        rate.sleep()
if __name__ == '__main__':
    listener()
