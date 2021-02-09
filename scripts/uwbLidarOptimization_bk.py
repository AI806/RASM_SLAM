#!/usr/bin/env python
from autoExploration.msg import uwbLocalization
from autoExploration.msg import mapInfor
from sensor_msgs.msg import LaserScan
from autoExploration.msg import velocityCtrl
from autoExploration.msg import odometryDualObs

import matplotlib.pyplot as plt
import rospy
import numpy as np
from bresenhamLine import bresenhamLine
from linearCombination import linearCombination
import scipy.misc
import scipy.io as sio
import itertools
from numpy.linalg import matrix_rank
from ParticleFilterClass import ParticleFilter

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
velMsg = velocityCtrl()
odoMsg = odometryDualObs()

updateUWBLocalization = True
updateLidar = True
lidar_offsetx = None
lidar_offsety = None
obstacles = None
baseTheta = np.pi / 180   


lidarIndex = np.arange(0, 900).astype(int)
angleIter = np.arange(-np.pi/4.*3, np.pi/2., 0.25*np.pi/180.)

# offsetPi = 5*np.pi/180.0

# lidarIndex = np.arange(20, 880).astype(int)
# angleIter = np.arange(-np.pi/4.*3 + offsetPi, np.pi/2.0 - offsetPi, 0.25*np.pi/180.)

def robot_uwb_obs_msg_callback(msg):
    global uwbMsg, initUWB, updateUWBLocalization
    if updateUWBLocalization:
        uwbMsg = msg
    if not initUWB and uwbMsg.state == 2 and uwbMsg.posx != 0:
        initUWB = True


def robot_lidar_msg_callback(msg):
    global lidarData, updateLidar
    if updateLidar == True:
        lidarData = msg

def robot_vel_msg_callback(msg):
    global velMsg
    velMsg = msg

def robot_odo_msg_callback(msg):
    global odoMsg
    odoMsg = msg

# return value are in 5cm
def transformToRobotCoordinate(lidarAngular, lidarRanges, resol):

    global lidar_offsetx, lidar_offsety

    off_x = np.cos(lidarAngular) * lidarRanges + lidar_offsety
    off_y = np.sin(lidarAngular) * lidarRanges - lidar_offsetx
    obstaclesToRobot = (np.vstack((off_x, off_y)) / resol * 100)

    return obstaclesToRobot

def transformToWorldCoordinate(posEst, obstaclesToRobot):

    theta = posEst[-1]
    Ro = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    obstacles = np.dot(Ro, obstaclesToRobot) + np.tile(posEst[:2].reshape(2, 1), (1, obstaclesToRobot.shape[1]))

    return obstacles.T

def createParticlesGaussian(posEst, pfCnt, stateNoiseStd):

    particles = np.zeros((pfCnt, len(posEst)))
    particles[:, 0] = posEst[0] + (stateNoiseStd[0] * np.random.randn(pfCnt, 1)).flatten()
    particles[:, 1] = posEst[1] + (stateNoiseStd[1] * np.random.randn(pfCnt, 1)).flatten()
    particles[:, 2] = posEst[2] + (stateNoiseStd[2] * np.random.randn(pfCnt, 1)).flatten()
    return particles

# all the input should be convert to map's coordinate and offsetted to the center of the map
def updateMapMatrix(mapMatrix5, mapMatrixIndex5, obstacles, posEst, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied):

    end_pos = np.ones((obstacles.shape[0], 2), dtype = np.int32)
    obstaclesTmp = (obstacles + 0.5).astype(np.int32)
    start_pos = posEst[:2].astype(np.int32)
    end_pos[:, 0] = obstaclesTmp[:, 0]
    end_pos[:, 1] = obstaclesTmp[:, 1]
    flag = bresenhamLine(mapMatrix5, mapMatrixIndex5, start_pos, end_pos, updateSetFree, updateSetOccupied, currMarkFreeIndex, currMarkOccIndex)
    return mapMatrix5, mapMatrixIndex5

def listener():
    global initUWB, uwbMsg, updateUWBLocalization, lidar_offsetx, lidar_offsety, updateLidar, lidarData, resolution, velMsg, odoMsg

    rospack = rospkg.RosPack()
    config_file = rospack.get_path('autoExploration') + '/config/system.cfg'
    with io.open(config_file, 'r', encoding='utf-8') as f:
        config = libconf.load(f)

    # load topic information
    uwbTopic = config.default.uwb_localization.aMessageTopic
    lidarTopic = config.default.rplidar.aMessageTopic
    mappingTopic = config.default.lidarMapping.aMessageTopic
    velTopic = config.default.robot_vel_ctrl.aMessageTopic
    odoTopic = config.default.robot_encoder.aMessageTopic

    recordIdx = config.default.dataRecord.recordIdx
    env = config.default.exploration.env
    pfParam = config.default.pfParam
    mapSize = config.default.exploration.mapSize
    mOffset = config.default.exploration.mapOffset
    resolution = config.default.pfParam.resol
    nodeCnt = config.default.exploration.nodeCnt
    envName = config.default.dataRecord.envName

    # initialize ros
    rospy.init_node('uwbLidarMapping', anonymous=False)
    rospy.Subscriber(uwbTopic, uwbLocalization, robot_uwb_obs_msg_callback)
    rospy.Subscriber('/scan', LaserScan, robot_lidar_msg_callback)
    rospy.Subscriber(velTopic, velocityCtrl, robot_vel_msg_callback )
    rospy.Subscriber(odoTopic, odometryDualObs, robot_odo_msg_callback )
    mappingPub = rospy.Publisher(mappingTopic, mapInfor, queue_size = 20)
    mappinginforMsg = mapInfor()

    # #init the map
    # maxHeight = int(mapSize * 100 / resolution)
    # maxWidth = int(mapSize * 100 / resolution)
    # mapMatrix5 = np.zeros((maxWidth, maxHeight))
    # mapMatrixIndex5 = np.zeros((maxWidth, maxHeight), dtype = np.int32)
    resol = 0.05
    maxHeight = 80
    maxWidth = 80
    mapMatrix5 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)))
    mapMatrixIndex5 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)), dtype = np.int32)

    mapMatrixSave = np.zeros((int(maxWidth / resol), int(maxHeight / resol)))
    mapMatrixIndexSave = np.zeros((int(maxWidth / resol), int(maxHeight / resol)), dtype = np.int32)

    # resol = 0.1
    # mapMatrix10 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)))
    # mapMatrixIndex10 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)), dtype = np.int32)
    # resol = 0.2
    # mapMatrix20 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)))
    # mapMatrixIndex20 = np.zeros((int(maxWidth / resol), int(maxHeight / resol)), dtype = np.int32)
    #init parameters for updating map
    lidar_offsetx = config.default.sRobotPara.lidar_offsetx
    lidar_offsety = config.default.sRobotPara.lidar_offsety
    occupiedThreshold = 0.7
    freeThreshold = 0.3
    updateSetFree = probToLog(0.46)
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
    lidarNoiseSave = 0.0

    homePath = os.getenv("HOME") + '/Slam_test/share_asus/auto_explore/' + envName + "/"
    if not os.path.isdir(homePath):
        os.makedirs(homePath)

    while initUWB == False and not rospy.is_shutdown():
        time.sleep(0.1)
    prePos = np.array([uwbMsg.robotPosx, uwbMsg.robotPosy])
    rate = rospy.Rate(10)  # 10hz
    # homePath = os.getenv("HOME") + '/Dropbox/share_asus/Slam_test/'
    lidarCnt = 0
    updateCnt = 0

    # init the particles of the Robots
    pfParam.mapOffset = np.array([int(mOffset * 100 / resolution), int(mOffset * 100 / resolution)])
    pfParam.mapSize = mapSize # in meter
    pf = ParticleFilter(pfParam)
    # mappinginforMsg.state =: 1, map initialization is finished but did not find the optimal scan matching, 2, indicate this is the first frame of map initialization
    # 3, find the optimal point, 4, feedback to uwb localization, 5, end of visualization

    validFrames = 1
    matchedFrames = 1
    initFinish = False
    initTimes = 1
    mappinginforMsg.state = 1
    while not rospy.is_shutdown():

        
        mappinginforMsg.timestamp = rospy.Time.now()
        start_time1 = time.time()

        # if len(lidarRanges) == 0: # check if the lidar data is valid
        #     rate.sleep()
        #     continue
        # end of lidar data preparing

        if uwbMsg.state == 3:
            print 'INFO: some needed nodes is missing, waiting to reconect them!'
            rate.sleep()
        # step one init the heading of the robot to calibrate the robot and UWB beacons into one cooridnation 
        if initTheta == False:
            start_time = time.time()
            Theta = uwbMsg.theta
            if Theta < 0:
                Theta = Theta + 2 * np.pi
            elif Theta > 2*np.pi:
                Theta = Theta - 2*np.pi
            print 'INFO: heading ', np.abs(Theta * 180.0 / np.pi)
            ## for workshop initialization np.abs(Theta * 180.0 / np.pi) > 1 or np.abs(Theta * 180.0 / np.pi) > 1:#
            # if np.sqrt( (prePos[0]-uwbMsg.robotPosx)**2 + (uwbMsg.robotPosy - prePos[1])**2 ) * resolution < 50 \
            if  np.abs(Theta * 180.0 / np.pi) > 1:#np.sqrt( (prePos[0]-uwbMsg.robotPosx)**2 + (uwbMsg.robotPosy - prePos[1])**2 ) * resolution < 30:
            # if np.sqrt( (prePos[0]-uwbMsg.robotPosx)**2 + (uwbMsg.robotPosy - prePos[1])**2 ) * resolution < 50:
                start_time = time.time()
                rate.sleep()
                continue
            else:
                updateUWBLocalization = False
                theta = Theta
                # prepare the lidar range datas
                [rangeOk, updateLidar, lidarRanges, lidarAngular] = \
                    prepareLidarRange(updateLidar, lidarData, lidarIndex, angleIter)
                if rangeOk == False:
                    rate.sleep()
                    continue
            offsetx = pfParam.mapOffset[0]
            offsety = pfParam.mapOffset[1]

            u_posx = np.array(uwbMsg.posx).flatten() + offsetx
            u_posy = np.array(uwbMsg.posy).flatten() + offsety

            anchorCnt = len(uwbMsg.posx) + 2
            anchorLen = (anchorCnt - 1) * 2
            posEst = np.zeros((anchorLen + 1, 1)).flatten()
            posEst[0] = offsetx + uwbMsg.robotPosx
            posEst[1] = offsety + uwbMsg.robotPosy
            posEst[-1] = Theta #np.pi
            posEst[np.arange(2, anchorLen, 2)] = u_posx
            posEst[np.arange(3, anchorLen, 2)] = u_posy
            axisId = copy.copy(uwbMsg.axisID)
            mobileId = copy.copy(uwbMsg.mobileID)
            R = np.array(copy.copy(uwbMsg.R)).reshape(anchorCnt, anchorCnt) * 100.
            updateUWBLocalization = True
            axisRemoveId = np.array([(axisId[0]-1)*2, (axisId[0]-1)*2+1, (axisId[1]-1)*2+1])#similar with the minus 2 at the previous processing, the minus 1 here is because the position of mobile nodes from ekf_uwb_localization is consisted of two positions, however, when fusing with Lidar, only the center of these two nodes will be estimated. So the posEst will only contain the center position of two mobile nodes, so we need to minus 1 here.
            initTheta = True
            # init the particles' state

            robotStateInit = np.zeros((3, 1)).flatten()
            robotStateInit[:2] = np.array([uwbMsg.robotPosx, uwbMsg.robotPosy]) * resolution / 100.0
            robotStateInit[2] = posEst[-1]
            pf.setParticleStates(robotStateInit) #in meter
            time.sleep(1)

        mappinginforMsg.pos = posEst.copy()
        mappinginforMsg.posEstMean = posEst[:2].copy()
        mappinginforMsg.rectifypos = []
        if initMap == False:
            initMap = True
            tileLen = 10
            obstaclesToRobot = convertLidarRangeToCadiaCoor(lidarRanges, lidarNoise, lidarAngular, tileLen)
            obstacles = transformToWorldCoordinate(posEst, obstaclesToRobot)

            currMarkOccIndex = currUpdateIndex + 2
            currMarkFreeIndex = currUpdateIndex + 1
            # [mapMatrix5, mapMatrixIndex5] = updateMapMatrix(mapMatrix5, mapMatrixIndex5, obstacles,\
            #      posEst, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)
            # currUpdateIndex += 3

            [mapMatrix5, mapMatrixIndex5] = \
            updateMultipleMap(currUpdateIndex, posEst, mapMatrix5, mapMatrixIndex5, updateSetFree, updateSetOccupied, obstacles)

            tileLen = 10
            obstaclesToRobot = convertLidarRangeToCadiaCoor(lidarRanges, lidarNoiseSave, lidarAngular, tileLen)
            obstacles = transformToWorldCoordinate(posEst, obstaclesToRobot)

            [mapMatrixSave, mapMatrixIndexSave] = \
            updateMultipleMap(currUpdateIndex, posEst, mapMatrixSave, mapMatrixIndexSave, updateSetFree, updateSetOccupied, obstacles)
            currUpdateIndex += 3

            mappinginforMsg.occupiedPosx = obstacles[:, 0]
            mappinginforMsg.occupiedPosy = obstacles[:, 1]
            mappinginforMsg.state = 2
            mappingPub.publish(mappinginforMsg)

            print('INFO: map updated!')
            # mapMatrix = logToProb(mapMatrixSave)
            # sio.savemat(homePath + str(recordIdx) + '_drawing', {'data':mapMatrix, 'obstacles': obstacles, \
            #     'lidarRanges': lidarRanges, 'lidarAngular': lidarAngular, 'posEst':posEst})

            lastPosEst = posEst
            rate.sleep()
            continue
        else:
            # get the estimation from the ekf and set this state as the initial point and find the optimal state of the robot according to the particle filter
            updateUWBLocalization = False
            anchorCnt = len(uwbMsg.posx) + 2
            anchorLen = (anchorCnt - 1) * 2
            posEst = np.zeros((anchorLen + 1, 1)).flatten()
            u_posx = np.array(uwbMsg.posx).flatten() # the x coordinate of UWB beacons, in 5cm
            u_posy = np.array(uwbMsg.posy).flatten() # the y coordinate of UWB beacons, in 5cm
            posEst[np.arange(2, anchorLen, 2)] = u_posx + offsetx
            posEst[np.arange(3, anchorLen, 2)] = u_posy + offsety
            posEst[0] = uwbMsg.robotPosx + offsetx # the x coordinate of robot, in 5cm
            posEst[1] = uwbMsg.robotPosy + offsety # the y coordinate of UWB beacons, in 5cm
            posEst[-1] = uwbMsg.theta # heading of the robot, in rad
            R = np.array(copy.copy(uwbMsg.R)).reshape(anchorCnt, anchorCnt) * 100. # UWB range table of all UWB nodes, indluding the two nodes on the robot, in cm
            node_id = copy.copy(np.array(uwbMsg.node_id)) # the id of all the UWB beacons and nodes
            axisRemoveId = copy.copy(uwbMsg.axisID) # the index of the nodes that needed to be removed, since they are used to determine the global coordinate of the map
            mobileId = copy.copy(uwbMsg.mobileID) # the index of the moving nodes, a.k.a the index of two nodes on the robot
            updateUWBLocalization = True
            axisRemoveId = np.array([(axisRemoveId[0]-1)*2, (axisRemoveId[0]-1)*2+1, (axisRemoveId[1]-1)*2 + 1]) # the index of the state that needed 
            # to be removed, since they are used to determine the global coordinate of the map
            # posEst: include all the states of UWB beacons and robot and its heading
        
        if initFinish == False:
            if initTimes < 5:
                initTimes += 1
            else:
                initFinish = True
            mappingPub.publish(mappinginforMsg)
            rate.sleep()
            continue
        
        mappinginforMsg.occupiedPosx = []
        mappinginforMsg.occupiedPosy = []
        # both coarse optimation and refine optimation is executed in this function, the obstacles have been transferred to robot's coordinate
        robotState = np.zeros((3, 1)).flatten()
        robotState[0] = (posEst[0] - offsetx) * resolution / 100.0  # in meter
        robotState[1] = (posEst[1] - offsety) * resolution / 100.0  # in meter
        robotState[2] = posEst[-1]

        # optPos in meter
        [rangeOk, updateLidar, lidarRanges, lidarAngular] = \
            prepareLidarRange(updateLidar, lidarData, lidarIndex, angleIter)
        if rangeOk == False:
            rate.sleep()
            continue
        updateFlag = 0
        obsToRobot = convertLidarRangeToCadiaCoor(lidarRanges, 0, lidarAngular, 1)
        [updateFlag, optPos] = pf.doParticleFiltering(mapMatrix5, mapMatrixSave, robotState, obsToRobot, \
             odoMsg.right_velocity, odoMsg.left_velocity, mappinginforMsg, mappingPub)

        validFrames += 1
        # print("INFO: update status, ", updateFlag, "optimized position, ", optPos[:2], "Heading:", optPos[2]*180/np.pi)
        # mappinginforMsg.state = 3

        
        if updateFlag == 2:# update the map
            
            matchedFrames += 1
            # make a copy from the found optimal point
            updatePoint = optPos.copy() # note the resolution is meter, need to change to 5cm and move the center of the map
            updatePoint[0] = updatePoint[0] * 20.0 + offsetx
            updatePoint[1] = updatePoint[1] * 20.0 + offsety

            updateCnt += 1

            isUpdate = poseDifferenceLargerThan(posEst, lastPosEst)
            isUpdate = True
            if isUpdate == True:   
                lastPosEst = posEst         
                tileLen = 5
                obstaclesToRobot = convertLidarRangeToCadiaCoor(lidarRanges, lidarNoise, lidarAngular, tileLen)
                obstacles = transformToWorldCoordinate(updatePoint, obstaclesToRobot)

                currMarkOccIndex = currUpdateIndex + 2
                currMarkFreeIndex = currUpdateIndex + 1
                # [mapMatrix5, mapMatrixIndex5] = updateMapMatrix(mapMatrix5, mapMatrixIndex5, obstacles, updatePoint, currMarkOccIndex, \
                #     currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)
                # currUpdateIndex += 3

                mappinginforMsg.occupiedPosx1 = obstacles[:, 0].copy()
                mappinginforMsg.occupiedPosy1 = obstacles[:, 1].copy()

                [mapMatrix5, mapMatrixIndex5] = \
                updateMultipleMap(currUpdateIndex, posEst, mapMatrix5, mapMatrixIndex5, updateSetFree, updateSetOccupied, obstacles)

                # for map showing
                tileLen = 1
                obstaclesToRobot = convertLidarRangeToCadiaCoor(lidarRanges, lidarNoiseSave, lidarAngular, tileLen)
                obstacles = transformToWorldCoordinate(updatePoint, obstaclesToRobot)

                [mapMatrixSave, mapMatrixIndexSave] = \
                updateMultipleMap(currUpdateIndex, posEst, mapMatrixSave, mapMatrixIndexSave, updateSetFree, updateSetOccupied, obstacles)
                currUpdateIndex += 3

                mappinginforMsg.occupiedPosx = obstacles[:, 0]
                mappinginforMsg.occupiedPosy = obstacles[:, 1]

            mappinginforMsg.state = 3
            mappinginforMsg.pos = posEst.copy() # 1:end are the state of the UWB beacons
            mappinginforMsg.posEstMean = updatePoint[:2].copy() # the state of the robot, which is the start point to update the occupied map

        elif updateFlag == 1: # update the UWB
            mappinginforMsg.state = 4

        rectPos = np.zeros((6, 1)).flatten()
        vel = (odoMsg.right_velocity + odoMsg.left_velocity) / 100.0 / 2.0 # in m/s
        omega = (odoMsg.right_velocity - odoMsg.left_velocity) / 100.0 / pfParam.fWheelDistance # in rad/s

        vel_r = pfParam.distUWBOnRobot * omega / 2.0 + vel
        vel_l = vel - pfParam.distUWBOnRobot * omega / 2.0
        rectPos[2] = vel_r # the linear speed of right node, in m/s
        rectPos[3] = vel_l # the linear speed of left node, in m/s
        rectPos[4] = optPos[2] # heading
        rectPos[5] = nodeCnt
        if pfParam.visualizeParticle == False:
            mappinginforMsg.particles = pf.particles.copy().flatten()
        if updateFlag == 2: #will update the position
            rectPos[:2] = optPos[:2] # x, y in meter

        mappinginforMsg.rectifypos = rectPos.flatten()
        mappinginforMsg.id = node_id

        mappingPub.publish(mappinginforMsg)

        print("INFO: elapsed time,", time.time() - start_time1)
        rate.sleep()

    mapMatrixSave[mapMatrixSave > 500] = 500 # to ensure that the map value is not too large that the exp opertation will result in inf, thus logToProb return a nan.
    mapMatrix = logToProb(mapMatrixSave)

    scipy.misc.imsave(homePath + str(recordIdx) + "_final_map.jpg", mapMatrixSave)
    sio.savemat(homePath + str(recordIdx) + '_map_record', {'lidarCnt':lidarCnt, \
    'updateCnt': updateCnt, 'data':mapMatrix, 'time':time.time() - start_time, 'lidarNoise': lidarNoise, \
    'matchedFrames': matchedFrames, 'validFrames':validFrames})

    cnt = 0
    mappinginforMsg.state = 5 # the mapping process end
    while cnt < 10: # keep publishing the message in 2 seconds to ensure that other node can receive it
        mappingPub.publish(mappinginforMsg)
        time.sleep(0.1)
        cnt += 1

    print('INFO: end mapping')
    print('matchedFrames:', matchedFrames, 'validFrames:', validFrames)

def poseDifferenceLargerThan(posEst, lastPosEst):
    global resolution
    if (np.sqrt( (posEst[0]-lastPosEst[0])**2 + (posEst[1]-lastPosEst[1])**2 ) / 100.0 * resolution) > 0.2:
        return True
    angleDiff = (posEst[2]-lastPosEst[2])
    if angleDiff > np.pi:
        angleDiff -= np.pi * 2.0
    elif angleDiff < -np.pi:
        angleDiff += np.pi * 2.0
    if np.abs(angleDiff) > 0.1:
        return True
    return False

def convertLidarRangeToCadiaCoor(lidarRanges, lidarNoise, lidarAngular, tileLen):
    global resolution
    ranges = np.tile(lidarRanges, (1, tileLen)).flatten() + (lidarNoise * np.random.randn(len(lidarRanges)*tileLen, 1)).flatten()
    angular = np.tile(lidarAngular, (1, tileLen)).flatten()

    obstaclesToRobot = transformToRobotCoordinate(angular, ranges, resolution)
    return obstaclesToRobot

def updateMultipleMap(currUpdateIndex, posEst, mapMatrix5, mapMatrixIndex5, updateSetFree, updateSetOccupied, obstacles):
    currMarkOccIndex = currUpdateIndex + 2
    currMarkFreeIndex = currUpdateIndex + 1
    [mapMatrix5, mapMatrixIndex5] = updateMapMatrix(mapMatrix5, mapMatrixIndex5, obstacles, posEst, \
        currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)

    # obstaclesTmp = obstacles/2
    # posEst1 = posEst / 2
    # [mapMatrix10, mapMatrixIndex10] = updateMapMatrix(mapMatrix10, mapMatrixIndex10, obstaclesTmp, \
    #     posEst1, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)

    # obstaclesTmp = obstacles/4
    # posEst1 = posEst / 4
    # [mapMatrix20, mapMatrixIndex20] = updateMapMatrix(mapMatrix20, mapMatrixIndex20, obstaclesTmp, \
    #     posEst1, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied)

    return mapMatrix5, mapMatrixIndex5

def prepareLidarRange(updateLidar, lidarData, lidarIndex, angleIter):
    # prepare the lidar range datas
    updateLidar = False
    ranges = np.array(copy.copy(lidarData.ranges))
    updateLidar = True

    rangeOk = True
    if len(ranges) == 0: # check if there has received the lidar data
        rangeOk = False

    try:
        lidarRanges = copy.copy(ranges[lidarIndex])
        filt = np.logical_and(lidarRanges > 0.05, lidarRanges < 30)
        lidarRanges = lidarRanges[filt]
        lidarAngular = angleIter[filt]
        # print("INFO:lidarRanges, ", lidarRanges)
    except:
        rangeOk = False
    return rangeOk, updateLidar, lidarRanges, lidarAngular

if __name__ == '__main__':
    listener()
