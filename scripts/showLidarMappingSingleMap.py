#!/usr/bin/env python
from my_simulations.msg import uwbLocalization
from my_simulations.msg import mapInfor
from sensor_msgs.msg import LaserScan
from my_simulations.msg import lidarObstacles
from my_simulations.msg import goalPos
from my_simulations.msg import velocityCtrl

import matplotlib
import rospy
import numpy as np
import libconf
import rospkg
import time
import io, os
import copy
from bresenhamLine import bresenhamLine
import scipy.io as sio
from lidar_simulate import *
# matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt

uwbMsg = uwbLocalization()
mappingMsg = mapInfor()
obsMsg = lidarObstacles()
velMsg = velocityCtrl()

updateMapping = True
updateLidar = True
updateUWBLocalization = True
initUWB = False
setNewGoal = 0
lidar_offsetx = None
lidar_offsety = None

def probToLog(prob):
    return np.log(prob / (1.0 - prob))

def logToProb(prob):
    exp = np.exp(prob)
    return exp / (1 + exp)

def robot_uwb_obs_msg_callback(msg):
    global uwbMsg, initUWB, updateUWBLocalization
    if updateUWBLocalization:
        uwbMsg = msg
    if not initUWB and uwbMsg.state >= 1:
        initUWB = True

def robot_vel_msg_callback(msg):
    global velMsg
    velMsg = msg

def robot_obs_obs_msg_callback(msg):
    global obsMsg
    obsMsg = msg

def robot_mapping_msg_callback(msg):
    global mappingMsg, updateMapping
    if updateMapping:
        mappingMsg = msg

def transformToRobotCoordinate(lidarAngular, lidarRanges, resol):
    global lidar_offsetx, lidar_offsety
    off_x = np.cos(lidarAngular) * lidarRanges + lidar_offsety
    off_y = np.sin(lidarAngular) * lidarRanges - lidar_offsetx
    obstaclesToRobot = (np.vstack((off_x, off_y)) / resol * 100)
    return obstaclesToRobot

def transformToWorldCoordinate(posEst, obstaclesToRobot):
    theta = posEst[2]  # - np.pi/2
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    obstacles = np.dot(R, obstaclesToRobot) + np.tile(np.array([[posEst[0]], [posEst[1]]]), (1, obstaclesToRobot.shape[1]))
    return obstacles.T

def findMapBoundary(uwbMsg, mapOffset, offsetX, offsetY, mapSize):

    posEstx = offsetX + np.array(uwbMsg.posx).astype(int)
    posEsty = offsetY + np.array(uwbMsg.posy).astype(int)

    minX = int(np.min(posEstx) - mapOffset)
    minY = int(np.min(posEsty) - mapOffset)
    maxX = int(np.max(posEstx) + mapOffset)
    maxY = int(np.max(posEsty) + mapOffset)

    if minX < 0:
        minX = 0
    if minY < 0:
        minY = 0
    if maxX >= mapSize[0]:
        maxX = mapSize[0]
    if maxY >= mapSize[1]:
        maxY = mapSize[1]

    return minX, minY, maxX, maxY

def doPathPlan(subMap, start_pos, goal_pos, extendL, minTorDist, scale, sampleSpeed, deltaT, curAngle, pathPlanTimeout):

    path_plan = RRT_PathPlanInit(subMap, start_pos, goal_pos, extendL, minTorDist, scale, False)
    optPath = path_plan.RRT_PathPlan(pathPlanTimeout)
    optPath = path_plan.PathSampling(optPath, 0, 2, 2*scale, 0, 0)
    optPath = path_plan.PathOptimal(optPath, extendL, minTorDist)
    followPath = path_plan.PathSampling(optPath, sampleSpeed, 1, 1, deltaT, curAngle)
    newGoal = np.zeros((1, 2))
    newGoal[0, 0] = followPath[-2, 0]
    newGoal[0, 1] = followPath[-2, 1]
    return followPath[:-1, :2], newGoal

def chooseNextBestGoal(subMap, pathPlanMap, posEstx, posEsty, oFFset, currentNode, idRecord, start_pos, extendL, minTorDist,\
     scale, sampleSpeed, deltaT, curAngle, pathPlanTimeout, initPath):

    maxScore = -0.001
    goal_pos = np.zeros((1, 2))
    passedNode = -1
    finalPath = None
    # print("idRecord:", idRecord)
    # print("posEstx:", posEstx)
    # print("posEsty:", posEsty)
    for i in np.arange(len(idRecord) - 2):
        idx = idRecord[i + 2]
        if currentNode is not None and idx == currentNode:
            continue
        goal_pos[0, 0] = posEstx[i]
        goal_pos[0, 1] = posEsty[i]
        goal_pos = goal_pos - oFFset
        [plannedPath, goalPos] = doPathPlan(pathPlanMap, start_pos, goal_pos, extendL, minTorDist, scale, sampleSpeed, deltaT, curAngle, pathPlanTimeout)

        entropySum = 0
        regionPiexl = 5
        entryCnt = 0
        print 'INFO: currently evaluate ID ', idx
        for j in np.arange(plannedPath.shape[0]):
            point = plannedPath[j, :]
            # if j == plannedPath.shape[0] - 1:
            #     regionPiexl = regionPiexl * 4
            subsubMap = subMap[int(point[1]-regionPiexl):int(point[1]+regionPiexl), int(point[0]-regionPiexl):int(point[0]+regionPiexl)]

            # print subsubMap
            entropy = subsubMap * np.log(subsubMap + 1e-10) + (1 - subsubMap) * np.log((1 - subsubMap) + 1e-10)
            entropySum += - np.sum(entropy.flatten())
            entryCnt += subsubMap.shape[0] * subsubMap.shape[1]
        print("entrpySum:", entropySum)
        if initPath == True:
            entropySum = entropySum / (entryCnt)
        print("entrpy:", entropySum, plannedPath.shape[0])
        if entropySum > maxScore:
            maxScore = entropySum
            finalPath = plannedPath.copy()
            nextRegion = idx
            bestGoal = goalPos

    return bestGoal, finalPath, nextRegion, maxScore

def CollisionDetection(maps, p1, p2, extend, minDist):
    mapSz = maps.shape
    start = np.floor(p2)
    goal = np.floor(p1)

    if goal[0] != start[0]:
        theta = np.arctan(np.abs( (goal[1]-start[1])/(goal[0]-start[0]) ))
    else:
        theta = np.pi/2
    offX = extend * np.sin(theta)
    offY = extend * np.cos(theta)
    # offX = extend
    # offY = extend
    polyx = np.array([start[0]-offX, goal[0]-offX, goal[0]+offX, start[0]+offX, start[0]-offX]).astype(int)
    polyy = np.array([start[1]+offY, goal[1]+offY, goal[1]-offY, start[1]-offY, start[1]+offY]).astype(int)

    lu = np.array([np.min(polyx), np.min(polyy)])
    rb = np.array([np.max(polyx), np.max(polyy)])
    lu[lu<0] = 0
    if rb[0] > mapSz[1]:
        rb[0] = mapSz[1]
    if rb[1] > mapSz[0]:
        rb[1] = mapSz[0]
    outerMap = maps[lu[1]:rb[1], lu[0]:rb[0]]
    [pointY, pointX] = np.where(outerMap == 1)

    pointX = pointX + lu[0]
    pointY = pointY + lu[1]
    hasObsInter = False
    hasObsPoint = False
    v_b = goal[:2] - start[:2]
    n_v_b = (v_b[0] ** 2 + v_b[1] ** 2)
    if len(pointX) >= 1:
        v_a_x = pointX - start[0]
        v_a_y = pointY - start[1]
        a_dot_b = ( v_a_x * v_b[0] + v_a_y * v_b[1] ) / n_v_b
        v_e_x = v_a_x - a_dot_b * v_b[0]
        v_e_y = v_a_y - a_dot_b * v_b[1]
        distMat = np.sqrt(v_e_x ** 2 + v_e_y ** 2)
        filter = distMat < extend
        if np.sum(filter) > 0:
            hasObsInter = True
    # newBound = np.array([goal[0]-minDist, goal[1]-minDist, goal[0]+minDist, goal[1]+minDist]).astype(int)
    # newBound[newBound < 0] = 0
    # if newBound[3] > mapSz[1]:
    #     newBound[3] = mapSz[1]
    # if newBound[2] > mapSz[0]:
    #     newBound[2] = mapSz[0]
    # pointRect = maps[newBound[1]:newBound[3], newBound[0]:newBound[2]]
    # if np.sum(pointRect[:]) > 1:
    #     hasObsPoint = True
    hasObsPoint = False
    if hasObsInter or hasObsPoint:
        iscollision = True
    else:
        iscollision = False
    return iscollision

def checkPathReplan(subMap, plannedPath, extend, minDist):
    path_replan = False
    idx = 0
    n = plannedPath.shape[0]

    distMat = np.sqrt( (plannedPath[idx+1:, 0] - plannedPath[idx:-1, 0]) ** 2 + (plannedPath[idx+1:, 1] - plannedPath[idx:-1, 1]) ** 2 )
    interDist = 0
    i = idx
    p1 = plannedPath[0, :]
    goalPos = plannedPath[-1, :]
    while i < n-1 and i >= idx:
        interDist += distMat[i-idx]
        if interDist > 20: #100cm
            interDist = 0
            p2 = plannedPath[i, :2].flatten()
            distToGoal = np.sqrt( (goalPos[0] - p2[0]) ** 2 + (goalPos[1] - p2[1]) ** 2)
            if distToGoal > 30: #150cm
                path_replan = CollisionDetection(subMap, p1, p2, extend, minDist)
                p1 = p2
                if path_replan == True:
                    break
            else:
                break
        i += 1
    return path_replan

def GenerateTrajectory(x, theta, vt, ot, iterCnt, deltaT):

    idx = np.arange(iterCnt)
    omega = theta + deltaT * ot * idx
    omega[omega > np.pi] = omega[omega > np.pi] - 2*np.pi
    omega[omega < -np.pi] = omega[omega < -np.pi] + 2 * np.pi
    sin_o = np.sin(omega)
    cos_o = np.cos(omega)
    traj = np.zeros((2, iterCnt))
    traj[0,:] = x[0] + np.cumsum(deltaT * vt * cos_o)
    traj[1,:] = x[1] + np.cumsum(deltaT * vt * sin_o)

    return traj

def publishGoal(goalPub, goalMsg, goal_pos, start_pos, plannedPath, oFFset, offsetX, offsetY):

    start_pos = start_pos + oFFset
    goalMsg.goal_posx = goal_pos[0, 0] + oFFset[0, 0]
    goalMsg.goal_posy = goal_pos[0, 1] + oFFset[0, 1]
    goalMsg.start_pos = start_pos.flatten()
    goalMsg.path_posx = plannedPath[:, 0] + oFFset[0, 0]
    goalMsg.path_posy = plannedPath[:, 1] + oFFset[0, 1]
    goalMsg.mapOffset = np.array([offsetX, offsetY]).flatten()
    goalMsg.stateFlag = 2
    goalPub.publish(goalMsg)

def checkScanLevel(maps, goal, exploredThreshold, mapOffset):
    mapOffset = int(20)
    left1 = int(goal[1]) - mapOffset
    left2 = int(goal[0]) - mapOffset
    right1 = int(goal[1]) + mapOffset
    right2 = int(goal[0]) + mapOffset
    if left1 < 0:
        left1 = 0
    if left2 < 0:
        left2 = 0
    if right1 >= maps.shape[0]:
        right1 = maps.shape[0] - 1
    if right2 >= maps.shape[1]:
        right2 = maps.shape[1] - 1

    submap = maps[ left1:right1, left2:right2 ].flatten()
    score = np.mean(np.abs(submap - 0.5))
    if score < exploredThreshold:
        return True
    else:
        return False

def CheckExplorationState(subMap, adapThres, frames, regionCnt, preEntropy, maxPathEntopyScore):

    if frames > regionCnt:
        adapThres = adapThres * np.exp(frames - regionCnt)
    entropy = subMap * np.log(subMap)
    entropySum = - np.sum(entropy.flatten())

    state = False
    if (entropySum < adapThres or maxPathEntopyScore < 0.12):
        state = True
    # print 'INFO: current threshold is ', adapThres
    return state, entropySum

def listener():
    global mappingMsg, updateMapping, uwbMsg, velMsg, setNewGoal,initUWB, obsMsg

    rospack = rospkg.RosPack()
    config_file = rospack.get_path('my_simulations') + '/config/system.cfg'
    with io.open(config_file, 'r', encoding='utf-8') as f:
        config = libconf.load(f)

    # load topic information
    mappingTopic = config.default.lidarMapping.aMessageTopic
    uwbTopic = config.default.uwb_localization.aMessageTopic
    pathTopic = config.default.robot_planned_path.aMessageTopic
    goalTopic = config.default.robot_set_goal.aMessageTopic
    obstaclesTopic = config.default.robot_obstacles.aMessageTopic
    velTopic = config.default.robot_vel_ctrl.aMessageTopic

    oneMeterPixels = config.default.sRobotPara.oneMeterPixels
    recordIdx = config.default.dataRecord.recordIdx
    drawPath = config.default.dataRecord.drawPath
    exploredThreshold = config.default.exploration.exploredThreshold
    env = config.default.exploration.env
    wait_time = config.default.exploration.waittime
    manualy = config.default.exploration.manually
    mapSize = config.default.exploration.mapSize
    mOffset = config.default.exploration.mapOffset
    pfCnt = config.default.pfParam.pfCnt
    pfParam = config.default.pfParam
    envName = config.default.dataRecord.envName
    # initialize ros
    rospy.init_node('uwbLidarMappingShowing', anonymous=False)
    rospy.Subscriber(mappingTopic, mapInfor, robot_mapping_msg_callback)
    rospy.Subscriber(uwbTopic, uwbLocalization, robot_uwb_obs_msg_callback)
    rospy.Subscriber(obstaclesTopic, lidarObstacles, robot_obs_obs_msg_callback )
    rospy.Subscriber(velTopic, velocityCtrl, robot_vel_msg_callback )
    goalPub = rospy.Publisher(goalTopic, goalPos, queue_size = 20)
    goalMsg = goalPos()

    # laod robot basic Configuration
    lidar_offsetx = config.default.sRobotPara.lidar_offsetx
    lidar_offsety = config.default.sRobotPara.lidar_offsety

    #init the map
    resolution = config.default.pfParam.resol # in cm
    maxHeight = int(mapSize * 100 / resolution)
    maxWidth = int(mapSize * 100 / resolution)
    mapMatrix5 = np.zeros((maxWidth, maxHeight))
    mapMatrixIndex5 = np.zeros((maxWidth, maxHeight), dtype = np.int32)

    mapMatrixShow = np.zeros((maxWidth, maxHeight))
    mapMatrixIndexShow = np.zeros((maxWidth, maxHeight), dtype = np.int32)

    #init parameters for updating map
    occupiedThreshold = 0.7
    freeThreshold = 0.3
    updateSetFree = probToLog(0.46)
    updateSetOccupied = probToLog(0.9)

    #draw maps
    plt.ion()
    fig = plt.figure()
    # plt.show(block=False)
    ax = fig.add_subplot(111)
    # fig.canvas.mpl_connect('button_press_event', on_button_press)

    if drawPath == True:
        fig2 = plt.figure(2)
        ax1 = fig2.add_subplot(111)

    rate = rospy.Rate(10)  # 10hz
    currMarkFreeIndex = 0
    currMarkOccIndex = 0
    currUpdateIndex = 0
    ac_pos_record_x = None
    ac_pos_record_y = None
    label_record = None
    ac_pos_record = None

    deltaT = 0.1
    iterCnt = int((9.0 / deltaT))

    # pathPlannedPath = os.getenv("HOME") + '/Slam_test/UWB_Lidar_Slam/semi_auto_explore/' + envName + "/"
    pathPlannedPath = os.getenv("HOME") + '/Slam_test/share_asus/auto_explore/' + envName + "/"
    if not os.path.isdir(pathPlannedPath):
        os.makedirs(pathPlannedPath)
    feedback = True
    offsetX = mOffset * 20
    offsetY = mOffset * 20
    mapOffset = 100
    biddedArea = 30
    preResult = None
    occupiedPosx = None
    trajectory = None
    record_traj = None

    # parameter for path planning
    scale = 2.0
    extendL = 10 * scale
    minTorDist = 8 * scale
    sampleSpeed = 0.5 * oneMeterPixels
    deltaT = 0.1
    pathPlanTimeout = 2.0
    start_pos = np.zeros((1, 2))
    subMap = None
    plannedPath = None
    exploredRegion = 0
    preEntropy = 1000000
    adapThres = 70
    bestGoal = None
    endExploration = False

    cntIdx = 30
    saveIdx = 1

    regionCnt = 5
    passedNode = None

    # print 'INFO: waiting for the initialization of UWB!'
    # while not rospy.is_shutdown() and uwbMsg.state == 1:
    #     time.sleep(0.1)
    # print 'INFO: UWB initialized!'

    UWBLiDARinit = False
    saveScreenShotIdx = 0
    nodePosInit = False
    waitForRecording = 1

    nextRegion = None

    maxPathEntopyScore = 1000

    path_record = None
    path_len_array = []
    path_record_id = 1

    prePathLen = None
    start_time = time.time()
    pastTraj = None
    mapInitlized = False

    # minX = 0
    minY = None
    initPath = False
    print("INFO: wait for uwb initalization")
    while not initUWB and not rospy.is_shutdown():
        time.sleep(0.1)

    while not rospy.is_shutdown() and mappingMsg.state != 5:
        # mappingMsg.state =: 1, map initialization is finished but did not find the optimal scan matching, 2, indicate this is the first frame of map initialization
        # 3, find the optimal point
        mapState = mappingMsg.state
        updateMapping = False
        posEstMean = np.array(mappingMsg.posEstMean)
        posEst = np.array(mappingMsg.pos)
        updateMapping = True
        if mapState == 3: #show the optimized map
            posCnt = (len(posEst)-1)
            theta = posEst[-1]
            posEstx = posEst[np.arange(2, posCnt, 2)].astype(int)
            posEsty = posEst[np.arange(3, posCnt, 2)].astype(int)
        else:## if lidar uwb fusion is not ready, then show the data collect from uwb slam
            posEstx = offsetX + np.array(uwbMsg.posx).astype(int)
            posEsty = offsetY + np.array(uwbMsg.posy).astype(int)
            if nodePosInit is False and uwbMsg.state == 2:
                posEstx_node = posEstx.copy()
                posEsty_node = posEsty.copy()
                idRecord = np.array(uwbMsg.node_id)
                nodePosInit = True

                ## find the boundary of the exploration region
                [minX, minY, maxX, maxY] = findMapBoundary(uwbMsg, mapOffset, offsetX, offsetY, mapMatrix5.shape)
                oFFset = np.array([[minX, minY]])

            robotPosx = offsetX + uwbMsg.robotPosx
            robotPosy = offsetY + uwbMsg.robotPosy
            posEstMean = np.array([robotPosx, robotPosy])
            theta = uwbMsg.theta

        if env == 1: #garden
            # maps = logToProb(mapMatrix5[450:1500, 500:1300])
            maps = logToProb(mapMatrix5)
        elif env == 2: #workshop
            # maps = logToProb(mapMatrix5[700:1200, 700:1100])
            maps = logToProb(mapMatrix5)
        mapsShow = logToProb(mapMatrixShow)

        if uwbMsg.state == 2:
            if setNewGoal == 2: # emergency stopping
                setNewGoal = 0
                goalMsg.stateFlag = 3
                goalPub.publish(goalMsg)
                rate.sleep()
                continue
            if uwbMsg.state == 3:
                print 'INFO: some needed nodes are missing, waiting for reconecting them!'
                rate.sleep()
                continue
            
            if mapState == 3 or mapState == 2:#update the map
                ## draw the grid map
                occupiedPosx = copy.copy(mappingMsg.occupiedPosx)
                occupiedPosy = copy.copy(mappingMsg.occupiedPosy)

                occupiedPosx1 = copy.copy(mappingMsg.occupiedPosx1)
                occupiedPosy1 = copy.copy(mappingMsg.occupiedPosy1)
                updateMapping = True
                currMarkOccIndex = currUpdateIndex + 2
                currMarkFreeIndex = currUpdateIndex + 1
                begin_pos = np.array(posEst[:2], dtype = np.int32)
                end_pos = np.ones((len(occupiedPosx), 2), dtype = np.int32)
                end_pos[:, 0] = occupiedPosx
                end_pos[:, 1] = occupiedPosy
                flag = bresenhamLine(mapMatrixShow, mapMatrixIndexShow, begin_pos, end_pos, updateSetFree, updateSetOccupied, currMarkFreeIndex, currMarkOccIndex)
                
                end_pos = np.ones((len(occupiedPosx1), 2), dtype = np.int32)
                end_pos[:, 0] = occupiedPosx1
                end_pos[:, 1] = occupiedPosy1
                flag = bresenhamLine(mapMatrix5, mapMatrixIndex5, begin_pos, end_pos, updateSetFree, updateSetOccupied, currMarkFreeIndex, currMarkOccIndex)
                currUpdateIndex += 3

            if minY is None:
                rate.sleep()
                continue
            subMap = maps[minY:maxY, minX:maxX].copy()
            start_pos[0, 0] = posEstMean[0] - oFFset[0, 0]#in 0.05m
            start_pos[0, 1] = posEstMean[1] - oFFset[0, 1]
            pathPlanMap = subMap.copy()
            filt = pathPlanMap>occupiedThreshold
            pathPlanMap[filt] = 1
            pathPlanMap[np.logical_not(filt)] = 0

            # print(posEstMean)
            if preResult is None:
                preResult = posEstMean.copy()
                trajectory = np.vstack((preResult, preResult))
                if record_traj is None:
                    record_traj = trajectory.copy()
                else:
                    record_traj = np.vstack((record_traj, preResult))
            elif np.sqrt( (preResult[0]-posEstMean[0])**2 + (preResult[1]-posEstMean[1])**2 ) > 0.1:
                preResult = posEstMean.copy()
                trajectory = np.vstack((trajectory, preResult))
                record_traj = np.vstack((record_traj, preResult))

        if manualy == False:
            # print("INFO: mappingMsg.state:", mapState)
            # if mapState == 2:
            #     mapInitlized = True

            # if mapInitlized == True:
            #     waitForRecording += 1
                
            # if waitForRecording > wait_time and endExploration == False:
            if waitForRecording > wait_time and endExploration == False:
                
                if uwbMsg.state == 2: ## received correct UWB data
                    [state, preEntropy] = CheckExplorationState(subMap, adapThres, exploredRegion, regionCnt, preEntropy, maxPathEntopyScore)
                    # print 'INFO: current entropy of the map is ', preEntropy
                    # print 'INFO: the entropy for the current planned path is ', maxPathEntopyScore
                    if state == True: ## the environment is well explored
                        goalMsg.stateFlag = 3 # finish the exploration
                        goalPub.publish(goalMsg)
                        endExploration = True
                        print 'INFO: finish the mapping process, good job!'
                    else:
                        if (velMsg.state == 3 or exploredRegion == 0): ## reach the new destination
                            [bestGoal, plannedPath, nextRegion, maxPathEntopyScore] = chooseNextBestGoal(subMap, pathPlanMap, posEstx, posEsty, \
                                 oFFset, nextRegion, idRecord, start_pos, extendL, minTorDist, scale, \
                                     sampleSpeed, deltaT, theta, pathPlanTimeout, initPath)
                            if initPath == False:
                                initPath = True
                            exploredRegion += 1
                            print 'INFO: go to region defined by node ', nextRegion, ' max entorpy:', maxPathEntopyScore
                            publishGoal(goalPub, goalMsg, bestGoal, start_pos, plannedPath, oFFset, offsetX, offsetY)

                            ## record the intermidiate data involvin with planned path
                            if path_record is None:
                                path_record = plannedPath[:, :2].copy()
                            else:
                                path_record = np.vstack((path_record, plannedPath[:, :2]))
                            path_len_array.append(plannedPath.shape[0])

                            print 'INFO: stored the length of planned path: ', path_len_array
                            sio.savemat(pathPlannedPath + str(recordIdx) + '_' + str(path_record_id) + '_map_for_pathplan', {'map':maps, 'path':plannedPath, 'offset':oFFset, 'time': time.time() - start_time})
                            path_record_id += 1

                        elif velMsg.state == 4: ##emgency stop, need to replan a new path
                            print 'INFO: replan a new path to bypass the obstalces!'
                            pathReplan_end = False
                            angleThres = 80 * np.pi / 180.0
                            preAngle = theta
                            if preAngle > np.pi:
                                preAngle = preAngle - 2 * np.pi
                            elif preAngle < -np.pi:
                                preAngle = preAngle + 2 * np.pi

                            newAngleThres = angleThres
                            loopCnt = 1
                            while pathReplan_end == False:
                                [plannedPath, goal] = doPathPlan(pathPlanMap, start_pos, bestGoal, extendL, minTorDist, scale, sampleSpeed, deltaT, theta, pathPlanTimeout)
                                # distMat = np.sqrt( (plannedPath[1:, 0] - plannedPath[:-1, 0]) ** 2 + (plannedPath[1:, 1] - plannedPath[:-1, 1]) ** 2 )
                                # curPathLen = np.sum(distMat.flatten())
                                angle = np.arctan2(plannedPath[1, 1] - plannedPath[0, 1], plannedPath[1, 0] - plannedPath[0, 0])

                                diff = angle - preAngle
                                if diff > np.pi:
                                    diff = 2 * np.pi - diff
                                elif diff < -np.pi:
                                    diff = 2 * np.pi + diff

                                print 'INFO: newAngleThres ', newAngleThres * 180 / np.pi, 'difference ', diff * 180 / np.pi
                                if np.abs(diff) < newAngleThres:
                                    pathReplan_end = True
                          
                                loopCnt += 1
                                if loopCnt > 10:
                                    newAngleThres = newAngleThres * 1.2
                                    loopCnt = 0

                            publishGoal(goalPub, goalMsg, bestGoal, start_pos, plannedPath, oFFset, offsetX, offsetY)

                            ## record intermidiate data involve with pathplan
                            if path_record is None:
                                path_record = plannedPath[:, :2].copy()
                            else:
                                path_record = np.vstack((path_record, plannedPath[:, :2]))
                            path_len_array.append(plannedPath.shape[0])
                            print 'INFO: stored the length of planned path: ', path_len_array
                            sio.savemat(pathPlannedPath + str(recordIdx) + '_' + str(path_record_id) + '_map_for_pathplan', {'map':maps, 'path':plannedPath, 'offset':oFFset, 'time': time.time() - start_time})
                            path_record_id += 1
                        else:
                            if plannedPath is not None:
                                extendLCheck = 5 * scale
                                minTorDistCheck = 5 * scale
                                path_replan = checkPathReplan(pathPlanMap, plannedPath, extendLCheck, minTorDistCheck)
                                if path_replan == True:
                                    print 'INFO: the path is blocked by obstacles, need to replan!'
                                    goalMsg.stateFlag = 3 # emergent stop the robot
                                    goalPub.publish(goalMsg)
                        if drawPath == True:
                            ax1.cla()
                            ax1.imshow(pathPlanMap)
                            if bestGoal is not None:
                                ax1.plot(bestGoal[0, 0] - minX, bestGoal[0, 1] - minY, 'ob')
                            ax1.plot(start_pos[0, 0] - minX, start_pos[0, 1] - minY, '*g')
                            ax1.plot(plannedPath[:, 0], plannedPath[:, 1], '-r')
                            ax1.set_xlim([0, pathPlanMap.shape[0]])# to change the direction of the coodinate in showing image
                            ax1.set_ylim([0, pathPlanMap.shape[1]]) # for the workshop
                            plt.savefig(pathPlannedPath + str(recordIdx) + '_planned_path_' + str(saveScreenShotIdx) + '.jpg')
                            saveScreenShotIdx += 1
                    # cntIdx += 1
                    # if cntIdx > 100:
                    #     sio.savemat(pathPlannedPath + str(recordIdx) + '_' + str(saveIdx) + '_map_1', {'map':maps})
                    #     saveIdx += 1
                    #     cntIdx = 1
            else:
                waitForRecording += 1

        mappingMsg.state = 0
        # draw maps
        ax.cla()
        ax.imshow(mapsShow, cmap='gray')
        # ax.imshow(maps)
        dx = 30 * np.cos(theta)
        dy = 30 * np.sin(theta)

        print "------------------"
        print posEstx
        print posEsty
        print "++++++++++++++++++++"

        ax.plot(posEstx, posEsty, 'oy')
        ax.plot(posEstMean[0], posEstMean[1], 'ob')
        ax.arrow(posEstMean[0], posEstMean[1], dx, dy, head_width=5, head_length=5, fc='b', ec='b')

        if manualy == False:
            if plannedPath is not None and endExploration == False:
                ax.plot(bestGoal[0, 0] + minX, bestGoal[0, 1] + minY, 'og')
                ax.plot(plannedPath[:, 0] + minX, plannedPath[:, 1] + minY, '-r')
        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], '-b')
            # ax.plot(record_traj[:, 0], record_traj[:, 1], '-b')

        # if occupiedPosx is not None:
        #     ax.plot(occupiedPosx, occupiedPosy, '.r')
        #     occupiedPosx = None

        if len(mappingMsg.particles) > 0:
            if pfParam.visualizeParticle == True:
                # print("INFOR: particle count, ", pfParam.pRefineCnt, "received particles count, ", len(mappingMsg.particles))
                particles = np.array(mappingMsg.particles).reshape(pfParam.pRefineCnt, 3)
            else:
                particles = np.array(mappingMsg.particles).reshape(pfParam.pfCnt, 3)
                particles[:, 0] = particles[:, 0] * 20 + offsetX
                particles[:, 1] = particles[:, 1] * 20 + offsetY
            ax.plot(particles[:, 0], particles[:, 1], '.c')
            mappingMsg.particles = []
            
            rectifypos = np.array(mappingMsg.rectifypos)
            if pfParam.visualizeParticle == False:
                rectifypos[0] = rectifypos[0] * 20 + offsetX
                rectifypos[1] = rectifypos[1] * 20 + offsetY 
            ax.plot(rectifypos[0], rectifypos[1], '.r')

            if mappingMsg.state == 3:
                dx = 30 * np.cos(rectifypos[-2])
                dy = 30 * np.sin(rectifypos[-2])
                ax.arrow(rectifypos[0], rectifypos[1], dx, dy, head_width=5, head_length=5, fc='r', ec='r')

            # print("INFO: rectifypos,", rectifypos)

        # if pastTraj is None:
        #     pastTraj = np.array([rectifypos[0], rectifypos[1]]).reshape(2,1)
        # else:
        #     pastTraj = np.hstack((pastTraj, np.array([posEst[0], posEst[1]]).reshape(2,1)))
        # # print result.shape, np.array([posEst[0], posEst[1]]).shape
        # plt.plot(pastTraj[0, :], pastTraj[1, :], '-r')

        if manualy == False:
            traj = GenerateTrajectory(posEstMean*resolution/oneMeterPixels, theta, velMsg.vt, velMsg.ot, iterCnt, deltaT)
            traj = (traj / resolution) * oneMeterPixels
            ax.plot(traj[0, :], traj[1, :], '-g', linewidth=3.0)

        obstacle_x = np.array(obsMsg.obstacle_x) * oneMeterPixels / resolution # in global coordinate 5cm
        obstacle_y = np.array(obsMsg.obstacle_y) * oneMeterPixels / resolution
        ax.plot(obstacle_x, obstacle_y, '.r')

        if manualy == False:
            if len(obsMsg.refPos) == 2:
                ax.plot(obsMsg.refPos[0] * oneMeterPixels / resolution, obsMsg.refPos[1] * oneMeterPixels / resolution, '*b')

        # if env == 1: #garden
        #     ax.set_xlim([450, 1500])# to change the direction of the coodinate in showing image
        #     ax.set_ylim([500, 1300])# garden outside the lab
        # elif env == 2: #workshop
        #     ax.set_xlim([700, 1200])# to change the direction of the coodinate in showing image
        #     ax.set_ylim([700, 1100]) # for the workshop

        if env == 1: #garden
            ax.set_xlim([450, 1500])# to change the direction of the coodinate in showing image
            ax.set_ylim([500, 1300])# garden outside the lab
        elif env == 2: #workshop
            ax.set_xlim([0, mapSize * 20])# to change the direction of the coodinate in showing image
            ax.set_ylim([0, mapSize * 20]) # for the workshop
            # ax.set_xlim([700, 1200])# to change the direction of the coodinate in showing image
            # ax.set_ylim([700, 1100]) # for the workshop

        plt.pause(0.000001)
        rate.sleep()

    end_time = time.time() - start_time
    # sio.savemat(pathPlannedPath + str(recordIdx) + '_moved_traj', {'trajectory':record_traj, 'path_record':path_record, 'path_len_array':path_len_array, 'offset': oFFset, 'time': end_time})

    print 'INFO: elapsed ', end_time, ' to explore the environment!'
if __name__ == '__main__':
    listener()
