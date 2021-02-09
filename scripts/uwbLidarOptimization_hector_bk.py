#!/usr/bin/env python
from autoExploration.msg import uwbLocalization
from autoExploration.msg import keyboardInput
from autoExploration.msg import mapInfor
from sensor_msgs.msg import LaserScan
from autoExploration.msg import velocityCtrl

import matplotlib.pyplot as plt
import rospy
import numpy as np
from bresenhamLine import bresenhamLine
from linearCombination import linearCombination
import scipy.misc
import scipy.io as sio
import itertools
from numpy.linalg import matrix_rank

import libconf
import rospkg
import time
import io, os
import copy
import pickle

def probToLog(prob):
    return np.log(prob / (1.0 - prob))

def logToProb(prob):
    exp = np.exp(prob)
    return exp / (1 + exp)

initUWB = False
uwbMsg = uwbLocalization()
lidarData = LaserScan()
keyInput = keyboardInput()
velMsg = velocityCtrl()

updateUWBLocalization = True
updateLidar = True
lidar_offsetx = None
lidar_offsety = None
obstacles = None
baseTheta = np.pi / 180
resolution = 5.0

lidarIndex = np.arange(0, 900).astype(int)
angleIter = np.arange(-np.pi/4.*3, np.pi/2., 0.25*np.pi/180.)

def robot_uwb_obs_msg_callback(msg):
    global uwbMsg, initUWB, updateUWBLocalization
    if updateUWBLocalization:
        uwbMsg = msg
    if not initUWB and uwbMsg.state == 2 and uwbMsg.posx != 0:
        initUWB = True

def robot_keyinput_msg_callback(msg):
    global keyInput
    keyInput = msg

def robot_lidar_msg_callback(msg):
    global lidarData, updateLidar
    if updateLidar == True:
        lidarData = msg

def robot_vel_msg_callback(msg):
    global velMsg
    velMsg = msg

def interpMapValueWithDerivaties(mapMatrix, obstacles, scale):

    if obstacles.shape[1] != 2:
        return None, None, None, None
    filt1 = np.logical_and(obstacles[:, 0] < scale-1, obstacles[:, 1] < scale-1)
    filt2 = np.logical_and(obstacles[:, 0] > 0, obstacles[:, 1] > 0)
    filt = np.logical_and(filt1, filt2)

    obstacles = obstacles[filt, :]
    if obstacles.shape[0] < 1:
        return None, None, None, None

    lbPoints = np.trunc(obstacles).astype(int)
    rpPoints = lbPoints + 1

    # P00, P01, P10, P11
    mapValue = logToProb(np.array([mapMatrix[lbPoints[:, 1], lbPoints[:, 0]], mapMatrix[rpPoints[:, 1], lbPoints[:, 0]] \
        ,mapMatrix[lbPoints[:, 1], rpPoints[:, 0]], mapMatrix[rpPoints[:, 1], rpPoints[:, 0]]]).T)

    factors = obstacles - lbPoints
    factInv = 1 - factors

    dx1 = mapValue[:, 0] - mapValue[:, 1] # P00-P01
    dx2 = mapValue[:, 2] - mapValue[:, 3] # P10-P11

    dy1 = mapValue[:, 0] - mapValue[:, 2] # P00-P10
    dy2 = mapValue[:, 1] - mapValue[:, 3] # P01-P11

    deltaMx = -(factInv[:, 1] * dy1 + factors[:, 1] * dy2)
    deltaMy = -(factors[:, 0] * dx2 + factInv[:, 0] * dx1)
    mP = factors[:, 1] * ( factors[:, 0] * mapValue[:, 3] + factInv[:, 0] * mapValue[:, 2] ) + \
        factInv[:, 1] * ( factors[:, 0] * mapValue[:, 1] + factInv[:, 0] * mapValue[:, 0] )

    return mP, deltaMx, deltaMy, filt

def Jacob(posEst, R, mobileId, fixConstrain, axisRemoveId):

    nodes = R.shape[0]
    srn_all = list(np.arange(nodes))
    srn = list(itertools.combinations(srn_all, 2))
    H = np.zeros(((nodes-1)*2+1, len(srn)))
    G = np.zeros(((nodes-1)*2+1, len(srn)))
    theta = posEst[-1]
    ccos = np.cos(theta)
    ssin = np.sin(theta)
    for i in np.arange(len(srn)):
        ia = srn[i][0]
        ib = srn[i][1]

        if R[ia, ib] == 0.:
            continue
        if (ia == mobileId[0] or ia == mobileId[1]) or (ib == mobileId[0] or ib == mobileId[1]):

            if (ia == mobileId[0] and ib == mobileId[1]) or (ia == mobileId[1] and ib == mobileId[0]):
                continue

            iaH = (ia - 1) * 2
            ibH = (ib - 1) * 2
            xi = posEst[iaH]
            yi = posEst[iaH+1]
            xj = posEst[ibH]
            yj = posEst[ibH+1]
            if ia == mobileId[0]: # ia left node
                iaH = 0
                xi = posEst[iaH] - fixConstrain * ssin
                yi = posEst[iaH+1] + fixConstrain * ccos
                dx = ((-ccos) * (xi - xj) - ssin * (yi - yj)) * fixConstrain
            elif ia == mobileId[1]: # ia right node
                iaH = 0
                xi = posEst[iaH] + fixConstrain * ssin
                yi = posEst[iaH+1] - fixConstrain * ccos
                dx = (ccos * (xi - xj) + ssin * (yi - yj)) * fixConstrain
            elif ib == mobileId[0]: # ib left node
                ibH = 0
                xj = posEst[ibH] - fixConstrain * ssin
                yj = posEst[ibH+1] + fixConstrain * ccos
                dx = ((-ccos) * (xi - xj) - ssin * (yi - yj)) * fixConstrain
            elif ib == mobileId[1]: # ib right node
                ibH = 0
                xj = posEst[ibH] + fixConstrain * ssin
                yj = posEst[ibH+1] - fixConstrain * ccos
                dx = (ccos * (xi - xj) + ssin * (yi - yj)) * fixConstrain

            d = ( (xi - xj)**2 + (yi - yj)**2 )**0.5
            e = (d - R[ia, ib]) / d
            H[-1, i] = dx / d
        else:
            iaH = (ia - 1) * 2
            ibH = (ib - 1) * 2
            xi = posEst[iaH]
            yi = posEst[iaH+1]
            xj = posEst[ibH]
            yj = posEst[ibH+1]
            d = ( (xi - xj)**2 + (yi - yj)**2 )**0.5
            e = (d - R[ia, ib]) / d

        H[iaH, i] = (xi - xj) / d
        H[iaH + 1, i] = (yi - yj) / d
        H[ibH, i] = (xj - xi) / d
        H[ibH + 1, i] = (yj - yi) / d

        G[iaH, i] = (xi - xj) * e
        G[iaH + 1, i] = (yi - yj) * e
        G[ibH, i] = (xj - xi) * e
        G[ibH + 1, i] = (yj - yi) * e

    H = np.delete(H, np.s_[axisRemoveId], axis=0)
    H_full = np.dot(H, H.T)
    G = np.delete(G, np.s_[axisRemoveId], axis=0)
    G_full = np.sum(G, axis = 1).reshape((nodes-1)*2-2, 1)
    return H_full, G_full

def estmateTransformation(mapMatrix, obstaclesToRobot, posEst, mobileId, R, scale, resol, beta, axisRemoveId, fixConstrain):

    theta = posEst[-1]# - np.pi/2
    Ro = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    obstacles = np.dot(Ro, obstaclesToRobot) + np.tile(np.array([[posEst[0]], [posEst[1]]]), (1, obstaclesToRobot.shape[1]))
    obstaclesToRobot = obstaclesToRobot.T
    obstacles = obstacles.T

    dim = len(posEst) - 3
    [mP, deltaMx, deltaMy, filt] = interpMapValueWithDerivaties(mapMatrix, obstacles, scale)
    if mP is None:
        return np.zeros((dim + 3, 1)).flatten(), obstacles

    H = np.zeros((len(deltaMx), dim, dim))
    obstaclesToRobot = obstaclesToRobot[filt, :]
    C = deltaMx * ( - np.sin(theta) * obstaclesToRobot[:, 0] - np.cos(theta) * obstaclesToRobot[:, 1] ) + \
        deltaMy * ( np.cos(theta) * obstaclesToRobot[:, 0] - np.sin(theta) * obstaclesToRobot[:, 1] )
    AA = deltaMx * deltaMx
    BB = deltaMy * deltaMy
    CC = C * C
    AB = deltaMx * deltaMy
    AC = deltaMx * C
    BC = deltaMy * C

    H[:, 0 , 0] = AA
    H[:, 0 , 1] = AB
    H[:, 0 , -1] = AC
    H[:, 1 , 0] = AB
    H[:, 1 , 1] = BB
    H[:, 1 , -1] = BC
    H[:, -1 , 0] = AC
    H[:, -1 , 1] = BC
    H[:, -1 , -1] = CC
    H = np.sum(H, axis = 0)

    deltaMdelataS = (np.array([deltaMx, deltaMy, C]) * ( 1 - mP )).T
    deltaMdelataS = np.sum(deltaMdelataS, axis = 0)
    S_full = np.zeros((dim, 1))

    S_full[:2, 0] = deltaMdelataS[:2]
    S_full[-1, 0] = deltaMdelataS[2]

    [J_full, G_full] = Jacob(posEst, R / resol, mobileId, fixConstrain / resol, axisRemoveId)
    H = H - beta*J_full

    if matrix_rank(H) < dim:
        return np.zeros((dim+3, 1)).flatten(), obstacles

    HIverse = np.linalg.inv(H)
    deltaEpsilon = np.dot(HIverse, S_full + beta*G_full).flatten()
    epsilon = insertDataToArray(dim, deltaEpsilon, axisRemoveId)
    return epsilon, obstacles

def insertDataToArray(dim, deltaEpsilon, axisRemoveId):
    tmpEpsilon = deltaEpsilon
    epsilon = np.zeros((dim+3, 1)).flatten()
    idx = 1
    for i in np.arange(len(axisRemoveId)):
        if axisRemoveId[i] >= dim + 3:
            print 'INFO: insert index is out of the bound'
            break
        if axisRemoveId[i] > dim:
            epsilon[0:dim] = deltaEpsilon
        else:
            epsilon[0:axisRemoveId[i]] = tmpEpsilon[0:axisRemoveId[i]]
            epsilon[axisRemoveId[i]+1:idx+dim] = tmpEpsilon[axisRemoveId[i]:]
            tmpEpsilon = epsilon[:idx+dim]
            idx += 1
    epsilon[axisRemoveId] = 0.
    return epsilon

def estimateTransformationMultiple(mapMatrix, lidarRanges, lidarAngular, times, posEst, mobileId, R, resol, thresold, scale, beta, axisRemoveId, fixConstrain):

    global lidar_offsetx, lidar_offsety

    off_x = np.cos(lidarAngular) * lidarRanges + lidar_offsety
    off_y = np.sin(lidarAngular) * lidarRanges - lidar_offsetx
    obstaclesToRobot = np.vstack((off_x, off_y)) / resol * 100

    updateFlag = False
    posReturn = posEst.copy()
    minScore = 100.

    for i in np.arange(times):

        [est, obstacles] = estmateTransformation(mapMatrix, obstaclesToRobot, posEst, mobileId, R, scale, resol, beta, axisRemoveId, fixConstrain)
        if (posEst[0] + est[0]) >= scale or (posEst[1] + est[1]) >= scale or (posEst[0] + est[0]) < 0 \
         or (posEst[1] + est[1]) < 0:#% or np.abs(est[-1]) > np.pi / 6:
            continue
        if np.sum(est[:-1] * (resol / 100.) > 1.) > 1:
            continue

        posEst = posEst + est
        theta = posEst[-1]# - np.pi/2

        Ro = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        obstacles = np.dot(Ro, obstaclesToRobot) + np.tile(np.array([[posEst[0]], [posEst[1]]]), (1, obstaclesToRobot.shape[1]))
        obstacles = obstacles.T
        if thresold is not None:
            obstaclesTmp = obstacles.astype(int)
            filt1 = np.logical_and(obstaclesTmp[:, 0] < scale, obstaclesTmp[:, 1] < scale)
            filt2 = np.logical_and(obstaclesTmp[:, 0] > 0, obstaclesTmp[:, 1] > 0)
            filt = np.logical_and(filt1, filt2)
            obstaclesTmp = obstaclesTmp[filt, :]
            if obstaclesTmp.shape[0] < 1:
                continue
            Prob = logToProb(mapMatrix[obstaclesTmp[:, 1], obstaclesTmp[:, 0]])
            freeCnt = np.sum(Prob < 0.5)
            occCnt = np.sum(Prob > 0.5)
            score = freeCnt / float(occCnt + 1.)
            #scoreValid = (occCnt + freeCnt) / float(obstaclesTmp.shape[0])
            scoreValid = (occCnt) / float(obstaclesTmp.shape[0])
            if (score < minScore and scoreValid > 0.35): #or (score < minScore and scoreValid > 0.2 and score < 0.1):
                minScore = score
                posReturn = posEst.copy()

    if minScore < thresold:
        updateFlag = True

    if thresold is None:
        return posEst, updateFlag
    else:
        return posReturn, updateFlag

def mapRepMultiMap(mapMatrix5, mapMatrix10, mapMatrix20, mapMatrix30, lidarRanges, lidarAngular, posEst, mobileId, R, threshold, beta, axisRemoveId, fixConstrain):

    for i in np.arange(1):
        posEst[:-1] = posEst[:-1] / 10.
        [posEst, updateFlag] = estimateTransformationMultiple(mapMatrix30, lidarRanges, lidarAngular, 10, posEst, mobileId, R, 50., None, 160, beta, axisRemoveId, fixConstrain)

        posEst[:-1] = posEst[:-1] * 2.5
        [posEst, updateFlag] = estimateTransformationMultiple(mapMatrix20, lidarRanges, lidarAngular, 10, posEst, mobileId, R, 20., None, 400, beta, axisRemoveId, fixConstrain)

        posEst[:-1] = posEst[:-1] * 2.
        [posEst, updateFlag] = estimateTransformationMultiple(mapMatrix10, lidarRanges, lidarAngular, 10, posEst, mobileId, R, 10., None, 800, beta, axisRemoveId, fixConstrain)

        posEst[:-1] = posEst[:-1] * 2.
        [posEst, updateFlag] = estimateTransformationMultiple(mapMatrix5, lidarRanges, lidarAngular, 5, posEst, mobileId, R, 5., threshold, 1600, beta, axisRemoveId, fixConstrain)

    # if updateFlag:
    #     print 'threshold:', threshold
    return posEst, updateFlag

def transformToRobotCoordinate(lidarAngular, lidarRanges, resol):

    global lidar_offsetx, lidar_offsety

    off_x = np.cos(lidarAngular) * lidarRanges + lidar_offsety
    off_y = np.sin(lidarAngular) * lidarRanges - lidar_offsetx
    obstaclesToRobot = (np.vstack((off_x, off_y)) / resol * 100)

    return obstaclesToRobot

def transformToWorldCoordinate(posEst, obstaclesToRobot):

    theta = posEst[-1]
    Ro = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    obstacles = np.dot(Ro, obstaclesToRobot) + np.tile(np.array([[posEst[0]], [posEst[1]]]), (1, obstaclesToRobot.shape[1]))

    return obstacles.T

def createParticlesGaussian(posEst, pfCnt, stateNoiseStd):

    particles = np.zeros((pfCnt, len(posEst)))
    particles[:, 0] = posEst[0] + (stateNoiseStd[0] * np.random.randn(pfCnt, 1)).flatten()
    particles[:, 1] = posEst[1] + (stateNoiseStd[1] * np.random.randn(pfCnt, 1)).flatten()
    particles[:, 2] = posEst[2] + (stateNoiseStd[2] * np.random.randn(pfCnt, 1)).flatten()
    return particles

def updateMapMatrix(mapMatrix5, mapMatrixIndex5, obstacles, posEst, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied):

    end_pos = np.ones((obstacles.shape[0], 2), dtype = np.int32)
    obstaclesTmp = (obstacles + 0.5).astype(np.int32)
    start_pos = posEst[:2].astype(np.int32)
    end_pos[:, 0] = obstaclesTmp[:, 0]
    end_pos[:, 1] = obstaclesTmp[:, 1]
    flag = bresenhamLine(mapMatrix5, mapMatrixIndex5, start_pos, end_pos, updateSetFree, updateSetOccupied, currMarkFreeIndex, currMarkOccIndex)
    return mapMatrix5, mapMatrixIndex5

def listener():
    global initUWB, uwbMsg, updateUWBLocalization, lidar_offsetx, lidar_offsety, updateLidar, lidarData, resolution, velMsg

    rospack = rospkg.RosPack()
    config_file = rospack.get_path('autoExploration') + '/config/system.cfg'
    with io.open(config_file, 'r', encoding='utf-8') as f:
        config = libconf.load(f)

    # load topic information
    uwbTopic = config.default.uwb_localization.aMessageTopic
    lidarTopic = config.default.rplidar.aMessageTopic
    mappingTopic = config.default.lidarMapping.aMessageTopic
    keyInputTopic = config.default.robot_keyboardCmd.aMessageTopic
    velTopic = config.default.robot_vel_ctrl.aMessageTopic

    fixConstrain = config.default.sRobotPara.fixConstrain
    recordIdx = config.default.dataRecord.recordIdx
    env = config.default.exploration.env
    fixConstrain = fixConstrain / 2.0

    # initialize ros
    rospy.init_node('uwbLidarMapping', anonymous=False)
    rospy.Subscriber(uwbTopic, uwbLocalization, robot_uwb_obs_msg_callback)
    rospy.Subscriber('/scan', LaserScan, robot_lidar_msg_callback)
    rospy.Subscriber(keyInputTopic, keyboardInput, robot_keyinput_msg_callback)
    rospy.Subscriber(velTopic, velocityCtrl, robot_vel_msg_callback )
    mappingPub = rospy.Publisher(mappingTopic, mapInfor, queue_size = 20)
    mappinginforMsg = mapInfor()

    #init the map
    anchorsPosX = np.array(config.default.uwb_localization.anchorPosX) * resolution / 100.
    anchorsPosY = np.array(config.default.uwb_localization.anchorPosY) * resolution / 100.
    resol = 0.05
    maxHeight = 80
    maxWidth = 80
    mapMatrix5 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)))
    mapMatrixIndex5 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)), dtype = np.int32)
    resol = 0.1
    mapMatrix10 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)))
    mapMatrixIndex10 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)), dtype = np.int32)
    resol = 0.2
    mapMatrix20 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)))
    mapMatrixIndex20 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)), dtype = np.int32)
    resol = 0.5
    mapMatrix30 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)))
    mapMatrixIndex30 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)), dtype = np.int32)
    #init parameters for updating map
    lidar_offsetx = config.default.sRobotPara.lidar_offsetx
    lidar_offsety = config.default.sRobotPara.lidar_offsety
    occupiedThreshold = 0.7
    freeThreshold = 0.3
    updateSetFree = probToLog(0.4)
    updateSetOccupied = probToLog(0.9)
    currMarkFreeIndex = 0
    currMarkOccIndex = 0
    currUpdateIndex = 0

    # some temp parameters for initialization
    initTheta = False
    initMap = False
    initUWB = False
    initTimes = 0
    lidarNoise = 0.03

    homePath = os.getenv("HOME") + '/Slam_test/UWB_Lidar_Slam/semi_auto_explore/'
    if not os.path.isdir(homePath):
        os.makedirs(homePath)

    while initUWB == False and not rospy.is_shutdown():
        time.sleep(0.1)
    prePos = np.array([uwbMsg.robotPosx, uwbMsg.robotPosy])

    rate = rospy.Rate(10)  # 10hz
    # homePath = os.getenv("HOME") + '/Dropbox/share_asus/Slam_test/'
    recordCnt = 0
    thresold = 0.2
    lidarCnt = 0
    updateCnt = 0
    beta = 0.01
    feedback = True
    consective_feedback = False
    recordMapIdx = 0

    while not rospy.is_shutdown() and (keyInput.key_input != 99 and keyInput.key_input != 67):
        mappinginforMsg.timestamp = rospy.Time.now()
        mappinginforMsg.state = 1
        start_time1 = time.time()

        if uwbMsg.state == 3:
            print 'INFO: some needed nodes is missing, waiting to reconect them!'
            rate.sleep()
        if initTheta == False:
            start_time = time.time()
            if env == 2:
                ## for workshop initialization
                if np.sqrt( (prePos[0]-uwbMsg.robotPosx)**2 + (uwbMsg.robotPosy - prePos[1])**2 ) * resolution < 60:
                    start_time = time.time()
                    rate.sleep()
                    continue
            elif env == 1:
                # if np.sqrt( (prePos[0]-uwbMsg.robotPosx)**2 + (uwbMsg.robotPosy - prePos[1])**2 ) * resolution < 60:
                #     start_time = time.time()
                #     rate.sleep()
                #     continue
                ## for garden initialization
                print 'INFO: heading ', np.abs(uwbMsg.theta), ' taget heading ', 0 * np.pi / 180.0
                if velMsg.intiMapping != 2:
                    start_time = time.time()
                    rate.sleep()
                    continue
                # diff = uwbMsg.theta - 90 * np.pi / 180.0
                # if diff > np.pi:
                #     diff = 2 * np.pi - diff
                # elif diff < -np.pi:
                #     diff = 2 * np.pi + diff
                # if np.abs(diff) > 2 * np.pi / 180.0:
                #     start_time = time.time()
                #     rate.sleep()
                #     continue
            updateUWBLocalization = False
            theta = uwbMsg.theta
            posx = 800.
            posy = 800.
            offsetx = posx - uwbMsg.posx[uwbMsg.axisID[0]-2] ## the reason for minusing 2 is that the order of axisID is reordered at the package of ekf_uwb_localization, the reorder is done by move the id of mobile nodes to the mst front of the ID array. While the posx and posy here only record the anchors' position. So the index for the ID of axis node will all be reduced 2.
            offsety = posy - uwbMsg.posy[uwbMsg.axisID[0]-2]

            u_posx = np.array(uwbMsg.posx).flatten() + offsetx
            u_posy = np.array(uwbMsg.posy).flatten() + offsety

            anchorCnt = len(uwbMsg.posx) + 2
            anchorLen = (anchorCnt - 1) * 2
            posEst = np.zeros((anchorLen + 1, 1)).flatten()
            posEst[0] = offsetx + uwbMsg.robotPosx
            posEst[1] = offsety + uwbMsg.robotPosy
            posEst[-1] = 0#np.pi
            posEst[np.arange(2, anchorLen, 2)] = u_posx + offsetx
            posEst[np.arange(3, anchorLen, 2)] = u_posy + offsety
            axisId = copy.copy(uwbMsg.axisID)
            mobileId = copy.copy(uwbMsg.mobileID)
            R = np.array(copy.copy(uwbMsg.R)).reshape(anchorCnt, anchorCnt) * 100.
            updateUWBLocalization = True
            axisRemoveId = np.array([(axisId[0]-1)*2, (axisId[0]-1)*2+1, (axisId[1]-1)*2+1])#similar with the minus 2 at the previous processing, the minus 1 here is because the position of mobile nodes from ekf_uwb_localization is consisted of two positions, however, when fusing with Lidar, only the center of these two nodes will be estimated. So the posEst will only contain the center position of two mobile nodes, so we need to minus 1 here.
            initTheta = True
            time.sleep(1)

        if initMap:
            updateUWBLocalization = False
            anchorCnt = len(uwbMsg.posx) + 2
            anchorLen = (anchorCnt - 1) * 2
            posEst = np.zeros((anchorLen + 1, 1)).flatten()
            u_posx = np.array(uwbMsg.posx).flatten()
            u_posy = np.array(uwbMsg.posy).flatten()
            posEst[np.arange(2, anchorLen, 2)] = u_posx + offsetx
            posEst[np.arange(3, anchorLen, 2)] = u_posy + offsety
            posEst[0] = uwbMsg.robotPosx + offsetx
            posEst[1] = uwbMsg.robotPosy + offsety
            posEst[-1] = uwbMsg.theta
            R = np.array(copy.copy(uwbMsg.R)).reshape(anchorCnt, anchorCnt) * 100.
            node_id = copy.copy(np.array(uwbMsg.node_id))
            axisRemoveId = copy.copy(uwbMsg.axisID)
            mobileId = copy.copy(uwbMsg.mobileID)
            updateUWBLocalization = True
            axisRemoveId = np.array([(axisRemoveId[0]-1)*2, (axisRemoveId[0]-1)*2+1, (axisRemoveId[1]-1)*2 + 1])
        vel = np.sqrt(uwbMsg.vx**2 + uwbMsg.vy**2)
        # if vel < 0.1 or velMsg.state == 5:
        # if vel < 0.1:
        #     rate.sleep()
        #     continue

        updateLidar = False
        ranges = np.array(copy.copy(lidarData.ranges))
        updateLidar = True

        if len(ranges) == 0: # check if there has received the lidar data
            rate.sleep()
            continue
        lidarCnt += 1

        try:
            lidarRanges = ranges[lidarIndex]
            filt = np.logical_and(lidarRanges > 0.05, lidarRanges < 30)
            lidarRanges = lidarRanges[filt]
            lidarAngular = angleIter[filt]
        except:
            continue

        if len(lidarRanges) == 0: # check if the lidar data is valid
            rate.sleep()
            continue

        mappinginforMsg.pos = posEst.copy()
        mappinginforMsg.posEstMean = posEst[:2].copy()
        mappinginforMsg.rectifypos = []
        mappinginforMsg.occupiedPosx = []
        mappinginforMsg.occupiedPosy = []
        if initMap == False:

            initMap = True
            tileLen = 10
            ranges = np.tile(lidarRanges, (1, tileLen)).flatten() + (lidarNoise * np.random.randn(len(lidarRanges)*tileLen, 1)).flatten()
            angular = np.tile(lidarAngular, (1, tileLen)).flatten()

            obstaclesToRobot = transformToRobotCoordinate(angular, ranges, resolution)
            obstacles = transformToWorldCoordinate(posEst, obstaclesToRobot)

            currMarkOccIndex = currUpdateIndex + 2
            currMarkFreeIndex = currUpdateIndex + 1
            [mapMatrix5, mapMatrixIndex5] = updateMapMatrix(mapMatrix5, mapMatrixIndex5, obstacles, posEst, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)

            obstaclesTmp = obstacles/2
            posEst1 = posEst / 2
            [mapMatrix10, mapMatrixIndex10] = updateMapMatrix(mapMatrix10, mapMatrixIndex10, obstaclesTmp, posEst1, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)

            obstaclesTmp = obstacles/4
            posEst1 = posEst / 4
            [mapMatrix20, mapMatrixIndex20] = updateMapMatrix(mapMatrix20, mapMatrixIndex20, obstaclesTmp, posEst1, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)

            obstaclesTmp = obstacles/10
            posEst1 = posEst / 10
            [mapMatrix30, mapMatrixIndex30] = updateMapMatrix(mapMatrix30, mapMatrixIndex30, obstaclesTmp, posEst1, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)
            currUpdateIndex += 3

            mappinginforMsg.occupiedPosx = obstacles[:, 0]
            mappinginforMsg.occupiedPosy = obstacles[:, 1]
            mappinginforMsg.state = 5
            mappingPub.publish(mappinginforMsg)
            rate.sleep()
            print('INFO: map updated!')
            continue

        updateFlag = False
        [posEst, updateFlag] = mapRepMultiMap(mapMatrix5, mapMatrix10, mapMatrix20, mapMatrix30, lidarRanges, lidarAngular, posEst, mobileId, R, thresold, beta, axisRemoveId, fixConstrain)
        mappinginforMsg.state = 3
        if updateFlag == True:# and vel > 3:

            updateCnt += 1
            tileLen = 5
            ranges = np.tile(lidarRanges, (1, tileLen)).flatten() + (lidarNoise * np.random.randn(len(lidarRanges)*tileLen, 1)).flatten()
            angular = np.tile(lidarAngular, (1, tileLen)).flatten()

            obstaclesToRobot = transformToRobotCoordinate(angular, ranges, resolution)
            obstacles = transformToWorldCoordinate(posEst, obstaclesToRobot)

            currMarkOccIndex = currUpdateIndex + 2
            currMarkFreeIndex = currUpdateIndex + 1
            [mapMatrix5, mapMatrixIndex5] = updateMapMatrix(mapMatrix5, mapMatrixIndex5, obstacles, posEst, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)

            obstaclesTmp = obstacles/2
            posEst1 = posEst / 2
            [mapMatrix10, mapMatrixIndex10] = updateMapMatrix(mapMatrix10, mapMatrixIndex10, obstaclesTmp, posEst1, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)

            obstaclesTmp = obstacles/4
            posEst1 = posEst / 4
            [mapMatrix20, mapMatrixIndex20] = updateMapMatrix(mapMatrix20, mapMatrixIndex20, obstaclesTmp, posEst1, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)

            obstaclesTmp = obstacles/10
            posEst1 = posEst / 10
            [mapMatrix30, mapMatrixIndex30] = updateMapMatrix(mapMatrix30, mapMatrixIndex30, obstaclesTmp, posEst1, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)
            currUpdateIndex += 3

            mappinginforMsg.occupiedPosx = obstacles[:, 0]
            mappinginforMsg.occupiedPosy = obstacles[:, 1]
            mappinginforMsg.state = 5

            stateCnt = len(posEst)
            mappinginforMsg.pos = posEst.copy()
            mappinginforMsg.posEstMean = posEst[:2].copy()
            if feedback:
                mappinginforMsg.state = 2
                rectPos = posEst.copy()
                x_idx = np.arange(0, stateCnt-1, 2)
                rectPos[x_idx] = rectPos[x_idx] - offsetx
                rectPos[x_idx + 1] = rectPos[x_idx + 1] - offsety
                mappinginforMsg.rectifypos = rectPos.flatten()
                mappinginforMsg.id = node_id

        # recordCnt += 1
        # if recordCnt >= 50:
        #     recordCnt = 0
        #     mapMatrix = logToProb(mapMatrix5)
        #     sio.savemat(homePath + str(recordIdx) + '_map_record', {'lidarCnt':lidarCnt, \
        #     'updateCnt': updateCnt, 'data':mapMatrix, 'time':time.time() - start_time, 'offsetx':offsetx, 'offsety': offsety, \
        #     'beta': beta, 'feedback': feedback, 'consective_feedback': consective_feedback, 'thresold': thresold, 'lidarNoise': lidarNoise})
        mappingPub.publish(mappinginforMsg)
        rate.sleep()
    mapMatrix = logToProb(mapMatrix5)
    offset = 10
    index = np.where(mapMatrix > 0.5)
    index = np.array(index).T
    maxX = np.max(index[:, 0]) + offset
    minX = np.min(index[:, 0]) - offset
    maxY = np.max(index[:, 1]) + offset
    minY = np.min(index[:, 1]) - offset

    scipy.misc.imsave(homePath + str(recordIdx) + "_final_map.jpg", mapMatrix)
    sio.savemat(homePath + str(recordIdx) + '_map_record', {'lidarCnt':lidarCnt, \
    'updateCnt': updateCnt, 'data':mapMatrix, 'time':time.time() - start_time, 'offsetx':offsetx, 'offsety': offsety, \
    'beta': beta, 'feedback': feedback, 'consective_feedback': consective_feedback, 'thresold': thresold, 'lidarNoise': lidarNoise})

    cnt = 0
    mappinginforMsg.state = 4 # the mapping process end
    while cnt < 10: # keep publishing the message in 2 seconds to ensure that other node can receive it
        mappingPub.publish(mappinginforMsg)
        time.sleep(0.1)
        cnt += 1
    print 'INFO: end mapping'
if __name__ == '__main__':
    listener()
