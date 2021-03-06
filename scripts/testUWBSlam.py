#!/usr/bin/env python3
from my_simulations.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
import visualization_msgs.msg

import matplotlib.pyplot as plt
import rospy
import numpy as np
from bresenhamLine import bresenhamLine
import scipy.misc
import scipy.io as sio
import itertools
from numpy.linalg import matrix_rank
from ParticleFilterClass import ParticleFilter
from KFEstimation import *

import libconf
import rospkg
import time
import io, os
# import pickle
import threading
import tf
import tf2_ros
from apscheduler.schedulers.background import BackgroundScheduler
import util_tools as util

class UwbLidarOptimization(object):

    def __init__(self, config, uwbMessage, lidarMessage, odometryMessage):
        self.pfParam = config.default.pfParam
        self.exploParam = config.default.exploration
        self.robotParam = config.default.sRobotPara
        self.mapParam = config.default.occMap
        self.tfFrame = config.default.tf
        self.nodeCnt = config.default.exploration.nodeCnt

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
        self.mapParam.resol = 0.8 # in meter
        self.pfParam.resol = self.mapParam.resol # in m
        resol = self.mapParam.resol #in meter
        mOffset = self.mapParam.mapOffset / resol # in 5cm
        mapSize = self.mapParam.mapSize
        self.pfParam.mapOffset = np.array([mOffset, mOffset])  # in 5cm
        self.pfParam.mapSize = mapSize # in meter
        self.pf = ParticleFilter(self.pfParam)
        self.posEst = None
        self.prePoseEst = None
        self.curAngular = 0

        # init the map
        self.mapMatrix5 = np.zeros((int(mapSize / resol), int(mapSize / resol)))
        self.mapMatrixIndex5 = np.zeros((int(mapSize / resol), int(mapSize / resol)), dtype = np.int32)
        self.mapMatrixSave = np.zeros((int(mapSize / resol), int(mapSize / resol)))
        self.mapMatrixIndexSave = np.zeros((int(mapSize / resol), int(mapSize / resol)), dtype = np.int32)

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

        self.mutex = threading.Lock()

        self.mapMatrix = np.zeros((int(mapSize / resol), int(mapSize / resol)), dtype = np.byte)

        # data for visualization
        visualTopic = config.default.visualize.robot_state
        self.marker_pub = rospy.Publisher(visualTopic, visualization_msgs.msg.Marker, queue_size=1)
        visualTopic = config.default.visualize.obstacles
        self.marker_obs_pub = rospy.Publisher(visualTopic, visualization_msgs.msg.Marker, queue_size=1)

        # plt.ion()
        # fig = plt.figure()
        # self.ax = fig.add_subplot(111)

    def probToLog(self, prob):
        return np.log(prob / (1.0 - prob))

    def logToProb(self, prob):
        exp = np.exp(prob)
        return exp / (1 + exp)
    
    def poseDifferenceLargerThan(self, posEst):

        # print("INFO: posEst, ", posEst, ", lastPosEst, ", self.lastPosEst)
        if (np.sqrt( (posEst[0]-self.lastPosEst[0])**2 + (posEst[1]-self.lastPosEst[1])**2 )) > 0.2:
            return True
        angleDiff = (posEst[2]-self.lastPosEst[2])
        if angleDiff > np.pi:
            angleDiff -= np.pi * 2.0
        elif angleDiff < -np.pi:
            angleDiff += np.pi * 2.0
        if np.abs(angleDiff) > np.pi / 150:
            return True
        return False

    def transformToWorldCoordinate(self, robotState, obstaclesToRobot):
        theta = robotState[2]
        Ro = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        obstacles = np.dot(Ro, obstaclesToRobot) + np.tile(robotState[:2].reshape(2, 1), (1, obstaclesToRobot.shape[1]))
        return obstacles.T

    # return value are in 5cm
    def convertLidarRangeToCadiaCoor(self, lidarNoise): 

        ranges = self.laserRanges.flatten() + lidarNoise * np.random.randn(len(self.laserRanges), 1).flatten()
        angular = self.laserAngular.flatten()

        off_x = np.cos(angular) * ranges + self.robotParam.lidar_offsetx
        off_y = np.sin(angular) * ranges + self.robotParam.lidar_offsety
        obstaclesToRobot = (np.vstack((off_x, off_y)) / self.mapParam.resol)
        return obstaclesToRobot

    # all the input should be convert to map's coordinate and offsetted to the center of the map
    def updateMapMatrix(self, obstacles, robotState, indx):

        end_pos = np.ones((obstacles.shape[0], 2), dtype = np.int32)
        obstaclesTmp = (obstacles + 0.5).astype(np.int32)
        start_pos = robotState.astype(np.int32)
        end_pos[:, 0] = obstaclesTmp[:, 0]
        end_pos[:, 1] = obstaclesTmp[:, 1]
        self.mutex.acquire()
        if indx == 1: #update map for RASM
            flag = bresenhamLine(self.mapMatrix5, self.mapMatrixIndex5, start_pos, end_pos, self.updateSetFree,\
                 self.updateSetOccupied, self.currMarkFreeIndex, self.currMarkOccIndex)
        elif indx == 2: #update map for showing
            flag = bresenhamLine(self.mapMatrixSave, self.mapMatrixIndexSave, start_pos, end_pos, self.updateSetFree, \
                 self.updateSetOccupied, self.currMarkFreeIndex, self.currMarkOccIndex)
        self.mutex.release()

    # build the map
    def constructMap(self, robotState):

        self.currMarkOccIndex = self.currUpdateIndex + 2
        self.currMarkFreeIndex = self.currUpdateIndex + 1

        obstaclesToRobot = self.convertLidarRangeToCadiaCoor(self.pfParam.lidarNoise)
        obstacles = self.transformToWorldCoordinate(robotState, obstaclesToRobot)
        self.updateMapMatrix(obstacles, robotState, 1)

        obstaclesToRobot = self.convertLidarRangeToCadiaCoor(self.pfParam.lidarNoiseSave)
        obstacles = self.transformToWorldCoordinate(robotState, obstaclesToRobot)
        self.updateMapMatrix(obstacles, robotState, 2)

        self.currUpdateIndex = self.currUpdateIndex + 3

    def publishMap(self):

        try:
            self.mutex.acquire()            
            self.mapMatrixSave[self.mapMatrixSave > 500] = 500
            tMapMatrix = self.logToProb(self.mapMatrixSave)
            self.mutex.release()

            free = tMapMatrix < self.mapParam.free
            occ = tMapMatrix > self.mapParam.occupied
            unexplore = np.logical_and( tMapMatrix >= self.mapParam.free, tMapMatrix <= self.mapParam.occupied) 

            self.mapMatrix[free] = 1
            self.mapMatrix[occ] = 100
            self.mapMatrix[unexplore] = -1
            self.mapMsg.data = self.mapMatrix.flatten()

            self.mapMsg.header.stamp = rospy.Time.now()
            self.mapMsg.header.frame_id = 'map'
            self.mapPub.publish(self.mapMsg)
        except Exception as e:
            print(e)

    def prepareDataForRASM(self):
        offsetx = self.pfParam.mapOffset[0] # in 5cm
        offsety = self.pfParam.mapOffset[1] # in 5cm

        ## prepare UWB data
        self.uwbMessage.mutex.acquire()
        if self.uwbMessage.msgUpdate == False:
            self.uwbMessage.mutex.release()
            return False
        posx = np.array(self.uwbMessage.posx).flatten() # in m
        posy = np.array(self.uwbMessage.posy).flatten()
        robotPosx = self.uwbMessage.robotPosx
        robotPosy = self.uwbMessage.robotPosy
        Theta = self.uwbMessage.Theta
        self.uwbMessage.reverseUpdate()
        self.uwbMessage.mutex.release()

        ## prepare Odometry
        self.odometryMessage.mutex.acquire()
        if self.odometryMessage.msgUpdate == False:
            self.odometryMessage.mutex.release()
            return False
        self.left_vel = self.odometryMessage.l_vel
        self.right_vel = self.odometryMessage.r_vel
        self.vel = self.odometryMessage.vel
        self.omega = self.odometryMessage.omega
        self.odometryMessage.reverseUpdate()
        self.odometryMessage.mutex.release()

        ## prepare lidar data
        self.lidarMessage.mutex.acquire()
        if self.lidarMessage.msgUpdate == False:
            self.lidarMessage.mutex.release()
            return False
        self.laserRanges = np.array(self.lidarMessage.lidarRanges)
        self.laserAngular = np.array(self.lidarMessage.lidarAngular)
        self.lidarMessage.reverseUpdate()
        self.lidarMessage.mutex.release()


        ## states of UWB beacons and the two nodes mounted on the robot
        # The position of robot passed to RASM is in 5cm
        self.posEst = np.zeros(3)
        self.posEst[0] = robotPosx 
        self.posEst[1] = robotPosy
        self.posEst[2] = Theta #np.pi

        # init the map and the initial position for RASM
        if self.readyForMapping == False:
            self.readyForMapping = True
            # set the initial state for RASM
            robotStateInit = np.zeros((3, 1)).flatten()
            robotStateInit[:2] = np.array([robotPosx, robotPosy])

            # set initial pose for KF
            print("robotStateInit: ", robotStateInit)
            self.prePoseEst = robotStateInit[:2] / self.mapParam.resol

            # init the map in 5cm
            robotStateInit[0] = robotStateInit[0] / self.mapParam.resol + offsetx
            robotStateInit[1] = robotStateInit[1] / self.mapParam.resol + offsety
            self.constructMap(robotStateInit)
            self.constructMap(robotStateInit)
            self.constructMap(robotStateInit)
            self.constructMap(robotStateInit)

            self.lastPosEst = self.posEst.copy()
            self.stateMsg.state = 2

            print("INFO: map initialized and will be published frequently!")
            scheduler = BackgroundScheduler()
            scheduler.add_job(self.publishMap, 'interval', seconds=1)
            scheduler.start()

            # homePath = os.getenv("HOME") + '/PycharmProjects/pythonProject/'
            # if not os.path.isdir(homePath):
            #     os.makedirs(homePath)
            # mapMatrix = self.logToProb(self.mapMatrixSave)

            # sio.savemat(homePath  + 'map_record', {'data':mapMatrix, 'range': self.laserRanges, 'angular': self.laserAngular, 'robot':self.posEst})
        return True

    def endMapping(self):
        cnt = 0
        self.stateMsg.state = 5 # the mapping process end
        while cnt < 10: # keep publishing the message in 2 seconds to ensure that other node can receive it
            self.statePub.publish(self.stateMsg)
            time.sleep(0.1)
            cnt += 1
        print('INFO: end mapping')

    def run(self):

        offsetx = self.pfParam.mapOffset[0]
        offsety = self.pfParam.mapOffset[1]

        # # both coarse optimation and refine optimation is executed in this function, the obstacles have been transferred to robot's coordinate
        obsToRobot = self.convertLidarRangeToCadiaCoor(self.pfParam.lidarNoiseSave)
        robotStateInit = self.posEst.copy()


        yy = np.round(robotStateInit[1] / self.mapParam.resol) - self.prePoseEst[1] 
        xx = np.round(robotStateInit[0] / self.mapParam.resol) - self.prePoseEst[0]
        robotStateInit[2] = self.curAngular
        if yy > 0 or xx > 0:
            self.curAngular = np.arctan2( yy, xx )
            robotStateInit[2] = self.curAngular
            print("INFO, prePoseEst: ", self.prePoseEst, " robotStateInit: ", np.round(robotStateInit[:2] / self.mapParam.resol) )
            self.prePoseEst = np.round(robotStateInit[:2] / self.mapParam.resol)
            print("INFO, angular: ", robotStateInit[2] * 180 / np.pi, yy, xx)
            

        
        
        obstacles = self.transformToWorldCoordinate(robotStateInit, obsToRobot)
        
        # [nupdateFlag, updateFlag, optPos, score] = self.pf.doParticleFiltering(self.mapMatrix5, self.mapMatrixSave, self.posEst, obsToRobot.T, \
        #     self.vel, self.omega)

        # publish the transformation
        self.publishTransform(robotStateInit)
        

        # optPos is the coordinate of the robot, therefore, we need to compute its coodinate to the center of the two UWB nodes
        # publish estiamted pos for visualization
        trans_robot = self.computeTransformation(self.tfFrame.map, self.tfFrame.robot)
        self.publishVisulMakers(trans_robot, obstacles.T)
        trans_uwb = self.computeTransformation(self.tfFrame.map, self.tfFrame.uwb)

        # prepare data for rectifying the UWB nodes
        rectPos = np.zeros((6, 1)).flatten()
        rectPos[2] = self.right_vel # the linear speed of right node, in m/s
        rectPos[3] = self.left_vel # the linear speed of left node, in m/s
        rectPos[5] = self.nodeCnt
        self.stateMsg.state = 1
        position = robotStateInit.copy()

        updateFlag = 1
        if updateFlag == 1: ## update the state of robot, means that the optimal states are found, need to update the map
            # prepare data for correcting the estimates of UWB nodes
            if trans_uwb is not None:
                self.stateMsg.state = 3
                rectPos[0] = trans_uwb.transform.translation.x # x, y in meter
                rectPos[1] = trans_uwb.transform.translation.y
                rot = trans_uwb.transform.rotation
                e3 = tf.transformations.euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
                rectPos[4] = e3[2]

            # if trans_robot is not None:
            #     position[0] = trans_robot.transform.translation.x
            #     position[1] = trans_robot.transform.translation.y
            #     rot = trans_robot.transform.rotation
            #     e3 = tf.transformations.euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            #     position[2] = e3[2]

            isUpdate = self.poseDifferenceLargerThan(robotStateInit)
            # isUpdate = True
            if isUpdate == True:
                
                self.lastPosEst = robotStateInit.copy()
                updatePoint = robotStateInit.copy() # note the resolution is meter, need to change to 5cm and move the center of the map
                updatePoint[0] = updatePoint[0] / self.mapParam.resol + offsetx
                updatePoint[1] = updatePoint[1] / self.mapParam.resol + offsety
                self.constructMap(updatePoint)
            # if nupdateFlag == 0:
            #     sio.savemat('map_record', {'data':self.mapMatrixSave, 'range': self.laserRanges, 'angular':self.laserAngular, 'robot':self.posEst})

        self.stateMsg.rectifypos = rectPos.flatten()
        self.stateMsg.robotPos = position.flatten()
        self.statePub.publish(self.stateMsg)

        return updateFlag, position

    def publishVisulMakers(self, trans, obsToRobot):

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

        if obsToRobot is None:
            return
        vMarker = util.creatMarker(self.tfFrame.map, [0, 1, 0], 1, [0.1, 0.1, 0.1], 8)
        for i in np.arange(obsToRobot.shape[1]):
            point = geometry_msgs.msg.Point()
            point.x = obsToRobot[0, i]
            point.y = obsToRobot[1, i]
            vMarker.points.append(point) 
        self.marker_obs_pub.publish(vMarker)

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

    # ## wating for the ready of UWB sensors
    # while uwbMessage.initUWB == False and not rospy.is_shutdown():
    #     time.sleep(0.1)

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

    # coreAlgo.mapMatrixSave[coreAlgo.mapMatrixSave > 500] = 500 # to ensure that the map value is not too large that the exp opertation will result in inf, thus logToProb return a nan.
    # mapMatrix = coreAlgo.logToProb(coreAlgo.mapMatrixSave)

    # scipy.misc.imsave(homePath + str(recordIdx) + "_final_map.jpg", coreAlgo.mapMatrixSave)
    # sio.savemat('map_record', {'data':coreAlgo.mapMatrixSave, 'matchedFrames': matchedFrames, 'validFrames':validFrames})
    # print('matchedFrames:', matchedFrames, 'validFrames:', validFrames)

if __name__ == '__main__':
    listener()
