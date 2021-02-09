#!/usr/bin/env python
from my_simulations.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from move_base_msgs.msg import *
from geometry_msgs.msg import PoseStamped

from lidar_simulate import *

import rospy
import numpy as np
import libconf
import rospkg
import time
import io, os
import copy
import scipy.io as sio
import util_tools
import actionlib
import threading

class MapObsObj(object):

    def __init__(self, config):
        mapTopic = config.default.tf.map
        rospy.Subscriber(mapTopic, OccupancyGrid, self.robot_map_obs_msg_callback)
        self.mutex = threading.Lock()
        self.activeMap = None
        self.mapUpdate = False
        self.aRegion = None
        self.setRegion = False
    
    def setActiveReqion(self, region):
        self.aRegion = region.copy()
        self.setRegion = True

    def robot_map_obs_msg_callback(self, msg):
        if self.setRegion == False:
            return
        
        self.mutex.acquire()
        self.setRegion = False
        self.activeMap = np.array(msg.data).reshape(msg.info.width, msg.info.height).astype(np.int8)
        self.activeMap = self.activeMap[self.aRegion[2]:self.aRegion[3], self.aRegion[0]:self.aRegion[1]].copy()
        self.mapUpdate = True
        self.mutex.release()


class UWBRegionObj(object):

    def __init__(self, config):
        uwbTopic = config.default.uwb_localization.aMessageTopic
        rospy.Subscriber(uwbTopic, uwbLocalization, self.robot_uwb_obs_msg_callback)
        self.mapParam = config.default.occMap
        self.mapSize = self.mapParam.mapSize
        self.activeRegion = np.array([0, 0, 0, 0]) # minX - maxX, minY- maxY
        self.biasRegion = [6, 2, 6, 4] # in meter

        self.mutex = threading.Lock()
        self.uwbInit = False
        self.state = 0
        self.posx = None
        self.posy = None
        self.nodeID = None
        self.robotPosx = 0
        self.robotPosy = 0
        self.resol = int(1 / self.mapParam.resol)
        self.offset = int(self.mapParam.mapOffset)

    def robot_uwb_obs_msg_callback(self, msg):

        if msg.state == 3:
            self.uwbInit = True
        else:
            return

        posEstx = np.array(msg.posx).astype(int)
        posEsty = np.array(msg.posy).astype(int)

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

        self.mutex.acquire()
        self.activeRegion = (np.array([minX, maxX, minY, maxY]) * self.resol)  #in the map's resolution
        self.state = msg.state
        self.posx = ((posEstx.copy() + self.offset) * self.resol) - self.activeRegion[0]
        self.posy = ((posEsty.copy() + self.offset) * self.resol) - self.activeRegion[2]
        self.nodeID = copy.copy(msg.node_id)
        self.robotPosx = ((msg.robotPosx + self.offset) * self.resol) - self.activeRegion[0]
        self.robotPosy = ((msg.robotPosy + self.offset) * self.resol) - self.activeRegion[2]
        self.mutex.release()

class GoalAssign(object):

    def __init__(self, config, mapObj, uwbObj):

        #parameter for occ map
        self.mapParam = config.default.occMap
        self.occupied = self.mapParam.occupied
        self.free = self.mapParam.free

        # map infor
        self.map = None
        self.mapUpdate = False
        self.mapObj = mapObj
        self.mapOffset = None
        
        # adaptively find the terminate condition
        self.adapThres = 70

        # publish exploration state
        self.exploreParam = config.default.exploration
        self.exploreStatePub = rospy.Publisher(self.exploreParam.stateMessage, exploreState, queue_size = 10)
        self.exploreStateMsg = exploreState()

        # uwb nodes' states
        self.uwbObj = uwbObj
        self.uwbPosx = None
        self.uwbPosy = None
        self.nodeID = None

        # parameter for path planning
        self.scale = 2.0
        self.extendL = 10 * self.scale
        self.minTorDist = 8 * self.scale
        self.sampleSpeed = 0.5 * 100
        self.deltaT = 0.1
        self.pathPlanTimeout = 20000.0
        self.startPos = None

        self.debug = False

        # parameter for the exploration state
        self.exploredRegion = 0
        self.currentExploreUwbNode = -1

        # results
        self.bestGoal = None
        self.nextRegion = -1
        self.finalPath = None
        self.optimalScore = 100

    def prepareData(self):

        # prepare UWB data
        self.uwbObj.mutex.acquire()
        self.uwbPosx = self.uwbObj.posx.copy()
        self.uwbPosy = self.uwbObj.posy.copy()
        self.nodeID = copy.copy(self.uwbObj.nodeID)
        self.startPos = np.array([self.uwbObj.robotPosx, self.uwbObj.robotPosy]).reshape(1, 2)
        self.mapObj.setActiveReqion(self.uwbObj.activeRegion.copy())
        self.uwbObj.mutex.release()

        # prepare map data
        self.mapObj.mutex.acquire()
        self.mapUpdate = self.mapObj.mapUpdate

        print "self.mapUpdate: ", self.mapUpdate
        if self.mapUpdate == False:
            self.mapObj.mutex.release()
            return False
        else:
            self.mapObj.mapUpdate = False
            self.map = self.mapObj.activeMap.astype(np.float32)
            self.map[self.map == 1] = 0
            self.map[self.map == -1] = 0.5
            self.map[self.map == 100] = 1

            self.pathPlanMap = self.mapObj.activeMap.copy()
            self.pathPlanMap[self.pathPlanMap <= 1] = 0
            self.pathPlanMap[self.pathPlanMap == 100] = 1
        self.mapObj.mutex.release()
        return True

    def run(self): # return value: 1: finish the exploration process, 2: continue exploring current UWB node, 3:set new goal to explore

        # prepare the required sensor data
        if self.prepareData() == False:
            return 2

        print "INFO: Begin to plan the path!"

        self.exploreStateMsg.state = 1
        self.chooseNextBestGoal()
        print "INFO: max entropy is ", self.optimalScore
        if (self.optimalScore < 0.12):## the environment is well explored
            self.exploreStateMsg.state = 3 # finish the exploration
            self.exploreStatePub.publish(self.exploreStateMsg)
            print 'INFO: finish the mapping process, good job!'
            return 1
        else:
            if self.currentExploreUwbNode == self.nextRegion:
                return 2
            else:
                print 'INFO: go to region defined by node ', self.nextRegion, ' max entorpy:', self.optimalScore
                return 3

    def probToLog(self, prob):
        return np.log(prob / (1.0 - prob))

    def logToProb(self, prob):
        exp = np.exp(prob)
        return exp / (1 + exp)

    def doPathPlan(self, goal_pos, path_plan):
        
        path_plan.setGoal(goal_pos)
        optPath = path_plan.RRT_PathPlan(self.pathPlanTimeout)
        if optPath is None:
            return None, None
        optPath = path_plan.PathSampling(optPath, 0, 2, 2*self.scale, 0)
        optPath = path_plan.PathOptimal(optPath, self.extendL, self.minTorDist)
        followPath = path_plan.PathSampling(optPath, self.sampleSpeed, 1, 1, self.deltaT)
        newGoal = np.zeros(2)
        newGoal[0] = followPath[-2, 0]
        newGoal[1] = followPath[-2, 1]
        return followPath[:-1, :2], newGoal

    def chooseNextBestGoal(self):

        maxScore = -0.001
        goal_pos = np.zeros((1, 2))
        passedNode = -1
        regionPiexl = 5
        
        path_plan = RRT_PathPlanInit(self.pathPlanMap, self.startPos, self.extendL, self.minTorDist, self.scale, self.debug)
        for i in np.arange(len(self.uwbPosx)):
            idx = self.nodeID[i]

            goal_pos[0, 0] = self.uwbPosx[i]
            goal_pos[0, 1] = self.uwbPosy[i]
            [plannedPath, goalPos] = self.doPathPlan(goal_pos, path_plan)

            entropySum = 0
            entryCnt = 0
            for j in np.arange(plannedPath.shape[0]):
                point = plannedPath[j, :]
                subsubMap = self.map[int(point[1]-regionPiexl):int(point[1]+regionPiexl), int(point[0]-regionPiexl):int(point[0]+regionPiexl)]
                entropy = subsubMap * np.log(subsubMap + 1e-10) + (1 - subsubMap) * np.log((1 - subsubMap) + 1e-10)

                entropySum += - np.sum(entropy.flatten())
                entryCnt += subsubMap.shape[0] * subsubMap.shape[1]
            entropySum = entropySum / (entryCnt)
            
            if entropySum > maxScore:
                self.optimalScore = entropySum
                self.finalPath = plannedPath.copy()
                self.nextRegion = idx
                self.bestGoal = goalPos
                self.bestGoal = self.bestGoal * self.mapParam.resol

class Robot(object):
    goal = MoveBaseGoal()

    def __init__(self, config):

        self.global_frame = config.default.tf.map

        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

        Robot.goal.target_pose.header.frame_id = self.global_frame
        

    def sendGoal(self, point):
        Robot.goal.target_pose.pose.position.x = point[0]
        Robot.goal.target_pose.pose.position.y = point[1]
        Robot.goal.target_pose.pose.orientation.w = 1.0
        Robot.goal.target_pose.header.stamp = rospy.Time.now()
        self.client.send_goal(Robot.goal)

    def cancelGoal(self):
        self.client.cancel_goal()

    def getState(self):
        return self.client.get_state()

if __name__ == '__main__':
    
    rospack = rospkg.RosPack()
    config_file = rospack.get_path('my_simulations') + '/config/system.cfg'
    with io.open(config_file, 'r', encoding='utf-8') as f:
        config = libconf.load(f)

    # initialize ros
    rospy.init_node('goalAssignNode', anonymous=False)

    regionObj = UWBRegionObj(config)
    mapObj = MapObsObj(config)
    # mapStateObj = MapStateObj(config)
    goalAssign = GoalAssign(config, mapObj, regionObj)
    robot = Robot(config)

    print("INFO: wait for uwb initalization")
    while not regionObj.uwbInit and not rospy.is_shutdown():
        time.sleep(0.1)
    print 'INFO: UWB initialized!'
    rate = rospy.Rate(0.5)  # 1hz

    # pathPlannedPath = os.getenv("HOME") + '/Slam_test/share_asus/auto_explore/' + envName + "/"
    # if not os.path.isdir(pathPlannedPath):
    #     os.makedirs(pathPlannedPath)

    print "Init state: ", robot.getState()

    exploreState = False
    while not rospy.is_shutdown(): # and mapStateObj.mapState != 5:

        # mapStateObj.state =: 1, map initialization is finished but did not find the optimal scan matching, 2, indicate the first frame of map initialization
        # 4, find the optimal point
        print "New step -----"
        exploreState = goalAssign.run()
        if goalAssign.startPos is None:
            rate.sleep()
            continue
        print "Start point: ", goalAssign.startPos, " goal: ", goalAssign.bestGoal
        if exploreState == 1: # finish the exploretion
            break
        else:
            if exploreState == 3:
                robot.sendGoal(goalAssign.bestGoal)
            else:
                print "explored ", robot.getState(), " percent"
        rate.sleep()


