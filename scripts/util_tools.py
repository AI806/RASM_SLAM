#!/usr/bin/env python
import visualization_msgs.msg
import numpy as np
import rospy
from my_simulations.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
import threading

# uwb range measures
class RangeObservation(object):

    def __init__(self, config):
        rangeTopic = config.default.uwb_range_measure.realMessageTopic
        rospy.Subscriber(rangeTopic, uwbRangeMsg, self.robot_range_msg_callback)
        self.mutex = threading.Lock()
        self.ids = []
        self.ranges = []
        self.r_cov = []
        self.count = 0
        self.state = -1
        self.ap = []
        self.cn = []
        self.msgUpdate = False

    def robot_range_msg_callback(self, msg):

        self.mutex.acquire()
        self.ids = np.array(msg.ids)
        self.ranges = np.array(msg.ranges)
        self.r_cov = np.array(msg.r_cov)
        self.count = msg.count
        self.state = msg.state
        self.ap = msg.ap
        self.cn = msg.cn
        self.msgUpdate = True
        self.mutex.release()
        
    def reverseUpdate(self):
        self.msgUpdate = False
# planned path
class GlobalTrajectory(object):

    def __init__(self, config):
        posRectifyTopic = config.default.exploration.trajMessage
        rospy.Subscriber(posRectifyTopic, planedPath, self.path_msg_callback)
        self.mutex = threading.Lock()
        self.path_posx = []
        self.path_posy = []
        self.pathFlag = -1
        self.msgUpdate = False

    def path_msg_callback(self, msg):
        self.mutex.acquire()
        self.path_posx = np.array(msg.path_posx)
        self.path_posy = np.array(msg.path_posy)
        self.pathFlag = msg.pathFlag
        self.msgUpdate = True
        self.mutex.release()

    def reverseUpdate(self):
        self.msgUpdate = False
# refined uwb states and robot's state 
class UwbPosRectify(object):

    def __init__(self, config):
        posRectifyTopic = config.default.pfParam.aMessageTopic
        rospy.Subscriber(posRectifyTopic, statesAndRectify, self.pos_rectify_msg_callback)
        self.mutex = threading.Lock()
        self.rectifypos = []
        self.state = -1
        self.robotPos = []
        self.msgUpdate = False

    def pos_rectify_msg_callback(self, msg):

        self.mutex.acquire()
        self.rectifypos = np.array(msg.rectifypos)
        self.robotPos = np.array(msg.robotPos)
        self.state = msg.state
        self.msgUpdate = True
        self.mutex.release()

    def reverseUpdate(self):
        self.msgUpdate = False

# information of the built map
class MapObsObj(object):

    def __init__(self, config):
        mapTopic = config.default.tf.map
        rospy.Subscriber(mapTopic, OccupancyGrid, self.robot_map_obs_msg_callback)
        self.mutex = threading.Lock()
        self.activeMap = None
        self.aRegion = None
        self.setRegion = False
        self.msgUpdate = False
    
    def setActiveReqion(self, region):
        self.aRegion = region.copy()
        self.setRegion = True

    def robot_map_obs_msg_callback(self, msg):
        if self.setRegion == False:
            return
        self.mutex.acquire()
        map_w = np.array(msg.data).reshape(msg.info.width, msg.info.height).astype(np.int8)
        self.activeMap = map_w[self.aRegion[2]:self.aRegion[3], self.aRegion[0]:self.aRegion[1]]
        self.msgUpdate = True
        self.mutex.release()

    def reverseUpdate(self):
        self.msgUpdate = False
        
# raw estimated state of UWB nodes using filter
class UWBObservation(object):

    def __init__(self, config):
        uwbTopic = config.default.uwb_localization.aMessageTopic
        self.tfFrame = config.default.tf
        self.minMoveDist = config.default.exploration.minMoveDist
        self.angularOffset = config.default.exploration.angularOffset * np.pi / 180.0
        self.uwbSub = rospy.Subscriber(uwbTopic, uwbLocalization, self.robot_uwb_obs_msg_callback)
        self.mutex = threading.Lock()

        # parameter for UWB localization
        self.uwbInit = False
        self.headingReady = False
        self.frame = 0
        self.state = -1
        self.posx = []
        self.posy = []
        self.Theta = 0
        self.preRobotPos = None
        self.robotPosx = None
        self.robotPosy = None
        self.nodeID = None

        # parameter for active region
        self.mapParam = config.default.occMap
        self.mapSize = self.mapParam.mapSize
        self.activeRegion = None # minX - maxX, minY- maxY
        self.biasRegion = [6, 2, 6, 4] # in meter

        self.resol = int(1 / self.mapParam.resol)
        self.offset = int(self.mapParam.mapOffset)

        self.msgUpdate = False

    def robot_uwb_obs_msg_callback(self, msg):

        self.mutex.acquire()
        self.robotPosx = msg.robotPosx
        self.robotPosy = msg.robotPosy
        self.state = msg.state
        self.posx = np.array(msg.posx)
        self.posy = np.array(msg.posy)
        self.nodeID = np.array(msg.node_id)

        self.Theta = msg.theta - self.angularOffset
        if self.frame == 0:
            self.frame = 1
            self.preRobotPos = np.array([self.robotPosx, self.robotPosy])
        else:
            if self.headingReady == False and np.sqrt( (self.preRobotPos[0] - self.robotPosx)**2 + \
                    (self.robotPosy - self.preRobotPos[1])**2 ) >= self.minMoveDist:
                self.headingReady = True
        
        if self.state == 3:
            self.uwbInit = True

        if self.activeRegion is not None:
            self.robotPosx_goal = ((self.robotPosx + self.offset) * self.resol) - self.activeRegion[0]
            self.robotPosy_goal = ((self.robotPosy + self.offset) * self.resol) - self.activeRegion[2]
            
        self.msgUpdate = True
        self.mutex.release()
        
    def getActiveRegion(self):

        self.mutex.acquire()
        posEstx = np.array(self.posx).astype(int)
        posEsty = np.array(self.posy).astype(int)
        
        minX = int(np.min(posEstx) - self.biasRegion[0] + self.offset)
        minY = int(np.min(posEsty) - self.biasRegion[1] + self.offset)
        maxX = int(np.max(posEstx) + self.biasRegion[2] + self.offset)
        maxY = int(np.max(posEsty) + self.biasRegion[3] + self.offset)

        if minX < 0:
            minX = 0
        if minY < 0:
            minY = 0
        if maxX >= self.mapSize:
            maxX = self.mapSize
        if maxY >= self.mapSize:
            maxY = self.mapSize

        self.activeRegion = (np.array([minX, maxX, minY, maxY]) * self.resol)  #in the map's resolution

        self.posx_goal = ((posEstx.copy() + self.offset) * self.resol) - self.activeRegion[0]
        self.posy_goal = ((posEsty.copy() + self.offset) * self.resol) - self.activeRegion[2]
        self.robotPosx_goal = ((self.robotPosx + self.offset) * self.resol) - self.activeRegion[0]
        self.robotPosy_goal = ((self.robotPosy + self.offset) * self.resol) - self.activeRegion[2]

        self.mutex.release()

    def closeObj(self):
        self.uwbSub.unregister()

    def reverseUpdate(self):
        self.msgUpdate = False
# laser range measures   
class LaserObservation(object):

    def __init__(self, config):
        lidarTopic = config.default.hokuyolidar.aMessageTopic
        maxAngular = config.default.hokuyolidar.maxAngular
        minAngular = config.default.hokuyolidar.minAngular 
        angulResol = config.default.hokuyolidar.angulResol
        validAngular = np.array(config.default.hokuyolidar.validAngular)
        
        maxCnt = (maxAngular - minAngular) / angulResol
        lidarIndex = np.arange(0, maxCnt, 1).astype(int)
        angleIter = np.arange(minAngular, maxAngular, angulResol)

        validAnguLeft = int( (validAngular[0] - minAngular) / angulResol)
        validAnguRight = int( (validAngular[1] - minAngular) / angulResol)
        validIdx = np.arange(validAnguLeft, validAnguRight, 1).astype(int)

        self.angleIter = angleIter[validIdx] * np.pi / 180.0
        self.lidarIndex = lidarIndex[validIdx]
        
        self.minRange = config.default.hokuyolidar.minRange
        self.maxRange = config.default.hokuyolidar.maxRange
        rospy.Subscriber(lidarTopic, LaserScan, self.robot_lidar_msg_callback)
        self.mutex = threading.Lock()
        self.lidarRanges = None 
        self.lidarAngular = None
        self.msgUpdate = False
        self.stamp = None

    def robot_lidar_msg_callback(self, msg):

        self.mutex.acquire()
        self.msgUpdate = True
        if len(msg.ranges) < 500: # check if there has received the lidar data
            self.msgUpdate = False
            self.mutex.release()
            return

        try:
            lidarRanges = np.array(msg.ranges)
            lidarRanges = lidarRanges[self.lidarIndex]
            filt = np.logical_and(lidarRanges > self.minRange, lidarRanges < self.maxRange)
            self.lidarRanges = lidarRanges[filt]
            self.lidarAngular = self.angleIter[filt]
            self.stamp = msg.header.stamp
        except Exception as e:
            print(e)
            self.msgUpdate = False
            self.lidarRanges = None
            self.lidarAngular = None
        self.mutex.release()

    def reverseUpdate(self):
        self.msgUpdate = False
# odometry measures
class OdometryObservation(object):

    def __init__(self, config):

        self.isSimulation = config.default.gazebo.simulation
        self.fWheelDistance = config.default.sRobotPara.fWheelDistance
        if self.isSimulation == True:
            odoTopic = config.default.robot_encoder.simulateMessageTopic
            rospy.Subscriber(odoTopic, Odometry, self.robot_odo_msg_callback )
        else:
            odoTopic = config.default.robot_encoder.realMessageTopic
            rospy.Subscriber(odoTopic, odometryDualObs, self.robot_odo_msg_callback )
        self.mutex = threading.Lock()
        self.l_vel = 0
        self.r_vel = 0
        self.vel = 0
        self.omega = 0
        self.msgUpdate = False

    def robot_odo_msg_callback(self, msg):

        self.mutex.acquire()
        self.msgUpdate = True
        if self.isSimulation:
            self.vel = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
            self.omega = msg.twist.twist.angular.z
            self.l_vel = (self.vel - 0.5 * self.omega * self.fWheelDistance) 
            self.r_vel = (self.vel + 0.5 * self.omega * self.fWheelDistance)
        else:
            self.r_vel = msg.right_velocity / 100.0
            self.l_vel = msg.left_velocity / 100.0
            self.vel = (self.r_vel + self.l_vel) / 2.0
            self.omega = (self.r_vel - self.l_vel) / self.fWheelDistance
        self.mutex.release()
    
    def reverseUpdate(self):
        self.msgUpdate = False

class RecordingData(object):

    def __init__(self):

        self.robotState = None
        self.refPos = None
        self.obstacles = None
        self.u = None
        self.traj = None
        

    def setValue(self, u, traj, robotState, refPos, obstacles):

        if traj is not None:
            if self.traj is None:
                self.u = u
                self.traj = traj
            else:
                self.u = np.vstack((self.u, u))
                self.traj = np.dstack((self.traj, traj))
   
        if robotState is not None:
            if self.robotState is None:
                self.robotState = robotState
                self.refPos = refPos
                self.obstacles = obstacles
            else:
                self.robotState = np.vstack((self.robotState, robotState))
                self.refPos = np.vstack((self.refPos, refPos))
                self.obstacles = np.dstack((self.obstacles, obstacles))

# function for creating the visual marker for rviz
def creatMarker(frame_id, color, type, scale, id):
    vMarker = visualization_msgs.msg.Marker()
    vMarker.header.frame_id = frame_id
    vMarker.id = id
    if type == 0: # arrow
        vMarker.type = vMarker.ARROW
        vMarker.action = vMarker.ADD
    elif type == 1: # point
        vMarker.type = vMarker.POINTS
        vMarker.action = vMarker.ADD
    elif type == 2: # point
        vMarker.type = vMarker.LINE_STRIP
        vMarker.action = vMarker.ADD
    
    vMarker.pose.position.z = 0
    vMarker.scale.x = scale[0]
    vMarker.scale.y = scale[1]
    vMarker.scale.z = scale[2]
    vMarker.lifetime = rospy.Duration.from_sec(1)
    vMarker.color.a = 1
    vMarker.color.r = color[0]
    vMarker.color.g = color[1]
    vMarker.color.b = color[2]
    vMarker.header.stamp = rospy.Time.now()
    return vMarker