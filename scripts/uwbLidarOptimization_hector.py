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

class UwbLidarOptimization(object):

    def __init__(self, config, uwbMessage, lidarMessage, odometryMessage):

        self.pfParam = config.default.pfParam
        self.exploParam = config.default.exploration
        self.robotParam = config.default.sRobotPara
        self.mapParam = config.default.occMap
        self.tfFrame = config.default.tf
        self.nodeCnt = config.default.exploration.nodeCnt

        self.pfParam.resol = self.mapParam.resol # in m
        self.updateSetFree = self.probToLog(self.mapParam.free)
        self.updateSetOccupied = self.probToLog(self.mapParam.occupied)

        # for updating occ map
        self.currMarkFreeIndex = 0
        self.currMarkOccIndex = 0
        self.currUpdateIndex = 0

        ## object of sensor messages
        self.uwbMessage = uwbMessage
        self.lidarMessage = lidarMessage
        self.odometryMessage = odometryMessage
        
        # find a proper heading to initialize mapping and RASM
        self.readyForMapping = False

        # init RASM
        resol = self.mapParam.resol #in meter
        mOffset = self.mapParam.mapOffset / resol # in 5cm
        mapSize = self.mapParam.mapSize
        self.pfParam.mapOffset = np.array([mOffset, mOffset])  # in 5cm
        self.pfParam.mapSize = mapSize # in meter
        self.pf = ParticleFilter(self.pfParam)
        self.posEst = None

        # init the map
        resol = 0.05
        # the map using for selecting the explore destination
        self.mapMatrixSave = np.zeros((int(mapSize / resol), int(mapSize / resol)))
        self.mapMatrixIndexSave = np.zeros((int(mapSize / resol), int(mapSize / resol)), dtype = np.int32)
        # map for scan matching
        self.mapMatrix5 = np.zeros((int(mapSize / resol), int(mapSize / resol)))
        self.mapMatrixIndex5 = np.zeros((int(mapSize / resol), int(mapSize / resol)), dtype = np.int32)
        # the map published to rviz
        self.mapMatrix = np.zeros((int(mapSize / resol), int(mapSize / resol)), dtype = np.byte)
        resol = 0.1
        self.mapMatrix10 = np.zeros((int(mapSize / resol), int(mapSize / resol)))
        self.mapMatrixIndex10 = np.zeros((int(mapSize / resol), int(mapSize / resol)), dtype = np.int32)
        resol = 0.2
        self.mapMatrix20 = np.zeros((int(mapSize / resol), int(mapSize / resol)))
        self.mapMatrixIndex20 = np.zeros((int(mapSize / resol), int(mapSize / resol)), dtype = np.int32)
        resol = 0.5
        self.mapMatrix30 = np.zeros((int(mapSize / resol), int(mapSize / resol)))
        self.mapMatrixIndex30 = np.zeros((int(mapSize / resol), int(mapSize / resol)), dtype = np.int32)

        # statesAndRectify.state
        # 1, map initialization is finished but did not find the optimal scan matching, 
        # 2, indicate this is the first frame of map initialization
        # 3, find the optimal point, 
        # 4, feedback to uwb localization, 
        # 5, end of visualization
        stateTopic = config.default.pfParam.aMessageTopic
        self.statePub = rospy.Publisher(stateTopic, statesAndRectify, queue_size = 1)
        self.stateMsg = statesAndRectify()

        #publish the map for showing
        mapTopic = self.mapParam.aMessageTopic
        self.mapPub = rospy.Publisher(mapTopic, OccupancyGrid, queue_size = 1) 
        self.mapMsg = OccupancyGrid()

        self.mapMsg.info.resolution = self.mapParam.resol
        self.mapMsg.info.width = self.mapMatrixIndex5.shape[0]
        self.mapMsg.info.height = self.mapMatrixIndex5.shape[1]
        self.mapMsg.info.origin.position.x = -self.mapParam.mapOffset
        self.mapMsg.info.origin.position.y = -self.mapParam.mapOffset
        self.mapMsg.info.origin.position.z = 0.0 
        self.mapMsg.info.origin.orientation.x = 0.0 
        self.mapMsg.info.origin.orientation.y = 0.0 
        self.mapMsg.info.origin.orientation.z = 0.0 
        self.mapMsg.info.origin.orientation.w = 1.0 

        # broadcast the tf tree
        self.br = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tfBuffer)
        self.oMap_odom = geometry_msgs.msg.TransformStamped()
        self.oMap_odom.header.frame_id = self.tfFrame.map
        self.oMap_odom.child_frame_id = self.tfFrame.odom

        # data for visualization
        visualTopic = config.default.visualize.robot_state
        self.marker_pub = rospy.Publisher(visualTopic, visualization_msgs.msg.Marker, queue_size=1)

    def probToLog(self, prob):
        return np.log(prob / (1.0 - prob))

    def logToProb(self, prob):
        exp = np.exp(prob)
        return exp / (1 + exp)

    def interpMapValueWithDerivaties(self, mapMatrix, obstacles, scale):

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
        mapValue = self.logToProb(np.array([mapMatrix[lbPoints[:, 1], lbPoints[:, 0]], mapMatrix[rpPoints[:, 1], lbPoints[:, 0]] \
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

    def Jacob(self, posEst, R):

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
            if (ia == self.mobileId[0] or ia == self.mobileId[1]) or (ib == self.mobileId[0] or ib == self.mobileId[1]):
                if (ia == self.mobileId[0] and ib == self.mobileId[1]) or (ia == self.mobileId[1] and ib == self.mobileId[0]):
                    continue
                iaH = (ia - 1) * 2
                ibH = (ib - 1) * 2
                xi = posEst[iaH]
                yi = posEst[iaH+1]
                xj = posEst[ibH]
                yj = posEst[ibH+1]
                if ia == self.mobileId[0]: # ia left node
                    iaH = 0
                    xi = posEst[iaH] - self.fixConstrain * ssin
                    yi = posEst[iaH+1] + fixConstrain * ccos
                    dx = ((-ccos) * (xi - xj) - ssin * (yi - yj)) * self.fixConstrain
                elif ia == self.mobileId[1]: # ia right node
                    iaH = 0
                    xi = posEst[iaH] + self.fixConstrain * ssin
                    yi = posEst[iaH+1] - self.fixConstrain * ccos
                    dx = (ccos * (xi - xj) + ssin * (yi - yj)) * self.fixConstrain
                elif ib == self.mobileId[0]: # ib left node
                    ibH = 0
                    xj = posEst[ibH] - self.fixConstrain * ssin
                    yj = posEst[ibH+1] + self.fixConstrain * ccos
                    dx = ((-ccos) * (xi - xj) - ssin * (yi - yj)) * self.fixConstrain
                elif ib == self.mobileId[1]: # ib right node
                    ibH = 0
                    xj = posEst[ibH] + self.fixConstrain * ssin
                    yj = posEst[ibH+1] - self.fixConstrain * ccos
                    dx = (ccos * (xi - xj) + ssin * (yi - yj)) * self.fixConstrain
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

        H = np.delete(H, np.s_[self.axisRemoveId], axis=0)
        H_full = np.dot(H, H.T)
        G = np.delete(G, np.s_[self.axisRemoveId], axis=0)
        G_full = np.sum(G, axis = 1).reshape((nodes-1)*2-2, 1)
        return H_full, G_full

    def estmateTransformation(self, mapMatrix, obstaclesToRobot, posEst, R, scale):

        theta = posEst[-1]# - np.pi/2
        Ro = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        obstacles = np.dot(Ro, obstaclesToRobot) + np.tile(np.array([[posEst[0]], [posEst[1]]]), (1, obstaclesToRobot.shape[1]))
        obstaclesToRobot = obstaclesToRobot.T
        obstacles = obstacles.T

        dim = len(posEst) - 3
        [mP, deltaMx, deltaMy, filt] = self.interpMapValueWithDerivaties(mapMatrix, obstacles, scale)
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

        [J_full, G_full] = self.Jacob(posEst, R)
        H = H - self.beta*J_full

        if matrix_rank(H) < dim:
            return np.zeros((dim+3, 1)).flatten(), obstacles

        HIverse = np.linalg.inv(H)
        deltaEpsilon = np.dot(HIverse, S_full + self.beta*G_full).flatten()
        epsilon = insertDataToArray(dim, deltaEpsilon, axisRemoveId)
        return epsilon, obstacles

    def insertDataToArray(self, dim, deltaEpsilon, axisRemoveId):
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

    def estimateTransformationMultiple(self, mapMatrix, lidarRanges, lidarAngular, times, posEst, mobileId, R, resol, thresold, scale, beta, axisRemoveId, fixConstrain):

        global lidar_offsetx, lidar_offsety

        off_x = np.cos(lidarAngular) * lidarRanges + self.lidar_offsety
        off_y = np.sin(lidarAngular) * lidarRanges - self.lidar_offsetx
        obstaclesToRobot = np.vstack((off_x, off_y)) / resol * 100

        updateFlag = False
        posReturn = posEst.copy()
        minScore = 100.

        for i in np.arange(times):

            [est, obstacles] = self.estmateTransformation(mapMatrix, obstaclesToRobot, posEst, R, scale)
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
                Prob = self.logToProb(mapMatrix[obstaclesTmp[:, 1], obstaclesTmp[:, 0]])
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

    def mapRepMultiMap(self, mapMatrix5, mapMatrix10, mapMatrix20, mapMatrix30, lidarRanges, lidarAngular, posEst, mobileId, R, threshold, beta, axisRemoveId, fixConstrain):

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

    def transformToRobotCoordinate(self, lidarAngular, lidarRanges, resol):

        global lidar_offsetx, lidar_offsety

        off_x = np.cos(lidarAngular) * lidarRanges + lidar_offsety
        off_y = np.sin(lidarAngular) * lidarRanges - lidar_offsetx
        obstaclesToRobot = (np.vstack((off_x, off_y)) / resol * 100)

        return obstaclesToRobot

    def transformToWorldCoordinate(self, posEst, obstaclesToRobot):

        theta = posEst[-1]
        Ro = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        obstacles = np.dot(Ro, obstaclesToRobot) + np.tile(np.array([[posEst[0]], [posEst[1]]]), (1, obstaclesToRobot.shape[1]))

        return obstacles.T

    def updateMapMatrix(self, mapMatrix5, mapMatrixIndex5, obstacles, posEst, currMarkOccIndex, currMarkFreeIndex, currUpdateIndex, updateSetFree, updateSetOccupied):

        end_pos = np.ones((obstacles.shape[0], 2), dtype = np.int32)
        obstaclesTmp = (obstacles + 0.5).astype(np.int32)
        start_pos = posEst[:2].astype(np.int32)
        end_pos[:, 0] = obstaclesTmp[:, 0]
        end_pos[:, 1] = obstaclesTmp[:, 1]
        flag = bresenhamLine(mapMatrix5, mapMatrixIndex5, start_pos, end_pos, updateSetFree, updateSetOccupied, currMarkFreeIndex, currMarkOccIndex)
        return mapMatrix5, mapMatrixIndex5

    def run(self):

        offsetx = self.pfParam.mapOffset[0]
        offsety = self.pfParam.mapOffset[1]
        # both coarse optimation and refine optimation is executed in this function, the obstacles have been transferred to robot's coordinate
        # robotState = self.posEst # in meter

        obsToRobot = self.convertLidarRangeToCadiaCoor(self.pfParam.lidarNoiseSave, 1)
        [updateFlag, optPos] = self.pf.doParticleFiltering(self.mapMatrix5, self.mapMatrixSave, self.posEst, obsToRobot, \
            self.vel, self.omega)
        
        # publish the transformation
        self.publishTransform(optPos)

        # optPos is the coordinate of the robot, therefore, we need to compute its coodinate to the center of the two UWB nodes
        # publish estiamted pos for visualization
        trans = self.computeTransformation(self.tfFrame.map, self.tfFrame.robot)
        self.publishVisulMakers(trans)
        
        trans = self.computeTransformation(self.tfFrame.map, self.tfFrame.uwb)
        # prepare data for rectifying the UWB nodes
        rectPos = np.zeros((6, 1)).flatten()
        rectPos[2] = self.right_vel# the linear speed of right node, in m/s
        rectPos[3] = self.left_vel # the linear speed of left node, in m/s
        rectPos[5] = self.nodeCnt
        self.stateMsg.state = 4
        if updateFlag == 2: ## update the state of robot, means that the optimal states are found, need to update the map
            # prepare data for correcting the estimates of UWB nodes
            if trans is not None:
                self.stateMsg.state = 3
                rectPos[0] = trans.transform.translation.x # x, y in meter
                rectPos[1] = trans.transform.translation.y

                rot = trans.transform.rotation
                e3 = tf.transformations.euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
                rectPos[4] = e3[2]

            # isUpdate = self.poseDifferenceLargerThan(optPos)
            isUpdate = True
            if isUpdate == True:
                self.lastPosEst = optPos.copy() 
                updatePoint = optPos.flatten() # note the resolution is meter, need to change to 5cm and move the center of the map
                updatePoint[0] = updatePoint[0] / self.mapParam.resol + offsetx
                updatePoint[1] = updatePoint[1] / self.mapParam.resol + offsety
                self.constructMap(updatePoint, 3, 1)

        self.stateMsg.rectifypos = rectPos.flatten()
        self.stateMsg.robotPos = optPos.flatten()
        self.statePub.publish(self.stateMsg)

        return updateFlag, optPos

    def publishVisulMakers(self, trans):

        if trans is None:
            return
        vMarker = util.creatMarker(self.tfFrame.map, [1, 0, 0], 0, [0.8, 0.1, 0.1], 3)
        tran = trans.transform.translation
        rot = trans.transform.rotation
        vMarker.pose.position.x = tran.x
        vMarker.pose.position.y = tran.y
        vMarker.pose.orientation.x = rot.x
        vMarker.pose.orientation.y = rot.y
        vMarker.pose.orientation.z = rot.z
        vMarker.pose.orientation.w = rot.w
        self.marker_pub.publish(vMarker)

    def publishTransform(self, optPos):

        # # publish the fixed frame
        # self.br.sendTransform(self.map_oMap)

        Q = tf.transformations.quaternion_from_euler(0, 0, optPos[2])
        T1 = np.dot(tf.transformations.translation_matrix((optPos[0], \
            optPos[1], 0.0)), tf.transformations.quaternion_matrix(Q)) # oMap - robot 
        # T1 = np.dot(tf.transformations.translation_matrix((optPos[0] + self.mapParam.mapOffset, \
        #     optPos[1] + self.mapParam.mapOffset, 0.0)), tf.transformations.quaternion_matrix(Q)) # oMap - robot 

        trans = self.computeTransformation(self.tfFrame.robot, self.tfFrame.odom)
        if trans is None:
            return
        tran = trans.transform.translation
        rot = trans.transform.rotation
        T2 = np.dot(tf.transformations.translation_matrix((tran.x, tran.y, tran.z)),
                    tf.transformations.quaternion_matrix(np.array([rot.x, rot.y, rot.z, rot.w]))) # robot - odom
        T3 = np.dot(T1, T2) # oMap - odom

        self.lidarMessage.mutex.acquire()
        self.oMap_odom.header.stamp = self.lidarMessage.stamp + rospy.Duration(1)
        self.lidarMessage.mutex.release()
        
        tr3 = tf.transformations.translation_from_matrix(T3)
        self.oMap_odom.transform.translation.x = tr3[0] 
        self.oMap_odom.transform.translation.y = tr3[1] 
        self.oMap_odom.transform.translation.z = tr3[2]

        q3 = tf.transformations.quaternion_from_matrix(T3)
        self.oMap_odom.transform.rotation.x = q3[0]
        self.oMap_odom.transform.rotation.y = q3[1]
        self.oMap_odom.transform.rotation.z = q3[2]
        self.oMap_odom.transform.rotation.w = q3[3]

        self.br.sendTransform(self.oMap_odom)

    def computeTransformation(self, father, child):
        # for correction
        try:
            trans2 = self.tfBuffer.lookup_transform(father, child, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("waiting until the required frame is connected!")
            return None

        return trans2

def listener():

    rospack = rospkg.RosPack()
    config_file = rospack.get_path('my_simulations') + '/config/system.cfg'
    with io.open(config_file, 'r', encoding='utf-8') as f:
        config = libconf.load(f)

    recordIdx = config.default.dataRecord.recordIdx
    envName = config.default.dataRecord.envName
    homePath = os.getenv("HOME") + '/Slam_test/share_asus/auto_explore/' + envName + "/"
    if not os.path.isdir(homePath):
        os.makedirs(homePath)
    
    rospy.init_node('uwbLidarMapping', anonymous=False)

    # init the object for the sensors
    uwbMessage = util.UWBObservation(config)
    lidarMessage = util.LaserObservation(config)
    odometryMessage = util.OdometryObservation(config)

    # declare the object of core optimization
    coreAlgo = UwbLidarOptimization(config, uwbMessage, lidarMessage, odometryMessage)

    rate = rospy.Rate(10)  # 10hz

    # For evaluating the performance of RASM
    validFrames = 1
    matchedFrames = 1

    while not rospy.is_shutdown():
        print("-------------New step--------------")
        coreAlgo.stateMsg.state = 1

        if uwbMessage.state != 3:
            print('INFO: UWB nodes are not ready!')
            rate.sleep()
            continue
        # step 1: init the heading of the robot
        if uwbMessage.headingReady == False:
            rate.sleep()
            continue
        # step 2: waiting for a good laser range
        if lidarMessage.msgUpdate == False:
            rate.sleep()
            continue
        # step 3 prepare uwb loclaization and laser ranges
        if coreAlgo.prepareDataForRASM() == False:
            rate.sleep()
            continue
        #step 4: init the map and RASM
        if coreAlgo.readyForMapping == False:
            rate.sleep()
            continue
        #step 5: execute RASM
        [updateFlag, optPos] = coreAlgo.run()

        # evaluate the mathching rate of RASM
        validFrames += 1
        if updateFlag == 2:
            matchedFrames += 1

        rate.sleep()

    # end mapping
    coreAlgo.endMapping()

    coreAlgo.mapMatrixSave[coreAlgo.mapMatrixSave > 500] = 500 # to ensure that the map value is not too large that the exp opertation will result in inf, thus logToProb return a nan.
    mapMatrix = coreAlgo.logToProb(coreAlgo.mapMatrixSave)

    scipy.misc.imsave(homePath + str(recordIdx) + "_final_map.jpg", coreAlgo.mapMatrixSave)
    sio.savemat(homePath + str(recordIdx) + '_map_record', {'data':mapMatrix, 'matchedFrames': matchedFrames, 'validFrames':validFrames})
    print('matchedFrames:', matchedFrames, 'validFrames:', validFrames)

if __name__ == '__main__':
    listener()
