#!/usr/bin/env python
from my_simulations.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from move_base_msgs.msg import *
from geometry_msgs.msg import *

import rospy
import numpy as np
import libconf
import rospkg
import time
import io, os
import scipy.io as sio
import util_tools
import actionlib
import threading
import util_tools as util
from RRT_PathPlan import *
from apscheduler.schedulers.background import BackgroundScheduler

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
        self.noedeInit = False

        # parameter for path planning
        self.scale = 2.0
        self.extendL = 10 * self.scale
        self.minTorDist = 8 * self.scale
        self.extendLCheck = 5 * self.scale
        self.sampleSpeed = 0.5 * 100
        self.deltaT = 0.1
        self.pathPlanTimeout = 2000.0
        self.startPos = None
        self.activeRegion = None
        self.offset = int(self.mapParam.mapOffset)

        self.debug = False

        # parameter for the exploration state
        self.exploredRegion = 0
        self.currentExploreUwbNode = -1
        self.curPlannedPath = None
        self.minEntropy = self.exploreParam.minEntropy

        # results
        self.nextRegion = -2
        self.finalPath = None
        self.optimalScore = 100
        self.plannedPath = None

        # parameter for publishing trjectory
        self.trajPub = rospy.Publisher(self.exploreParam.trajMessage, planedPath, queue_size = 2)
        self.trajMsg = planedPath()

        # parameter for tf tree
        self.tfFrame = config.default.tf

        # rviz visualization
        visualTopic = config.default.visualize.global_traj
        self.marker_pub = rospy.Publisher(visualTopic, visualization_msgs.msg.Marker, queue_size=1)
        self.pathShow = None
        # initial trajectory publisher
        print("INFO: map initialized and will be published frequently!")
        scheduler = BackgroundScheduler()
        scheduler.add_job(self.publishTraj, 'interval', seconds=0.1)
        scheduler.start()

        # for debugging
        self.debug_pathcheck = True
        if self.debug_pathcheck:
            plt.cla()
            plt.ion()
            plt.figure(1)

    def prepareData(self):

        # prepare UWB data
        self.uwbObj.mutex.acquire()
        if self.noedeInit == False:
            self.uwbPosx = self.uwbObj.posx_goal.copy()
            self.uwbPosy = self.uwbObj.posy_goal.copy()
            self.nodeID = self.uwbObj.nodeID.copy()
            self.activeRegion = self.uwbObj.activeRegion.copy()
            self.mapObj.setActiveReqion(self.activeRegion)
            self.noedeInit = True
            print("INFO: uwb states are initialized!")
        self.startPos = np.array([self.uwbObj.robotPosx_goal, self.uwbObj.robotPosy_goal]).reshape(1, 2)
        self.uwbObj.mutex.release()

        # prepare map data
        self.mapObj.mutex.acquire()
        self.msgUpdate = self.mapObj.msgUpdate

        if self.msgUpdate == False:
            self.mapObj.mutex.release()
            return False
        else:
            self.mapObj.msgUpdate = False
            self.map = self.mapObj.activeMap.astype(np.float32)
            self.map[self.map == 1] = 0
            self.map[self.map == -1] = 0.5
            self.map[self.map == 100] = 1

            self.pathPlanMap = self.mapObj.activeMap.copy()
            self.pathPlanMap[self.pathPlanMap <= 1] = 0
            self.pathPlanMap[self.pathPlanMap == 100] = 1
        self.mapObj.reverseUpdate()
        self.mapObj.mutex.release()
        return True

    def collisionDetection(self, p1, p2):
        mapSz = self.pathPlanMap.shape

        start = np.floor(p2)
        goal = np.floor(p1)

        theta = np.arctan2( goal[1]-start[1], goal[0]-start[0] )
        offX = self.extendLCheck * np.sin(theta)
        offY = self.extendLCheck * np.cos(theta)
        polyx = np.array([start[0]-offX, goal[0]-offX, goal[0]+offX, start[0]+offX, start[0]-offX]).astype(int)
        polyy = np.array([start[1]+offY, goal[1]+offY, goal[1]-offY, start[1]-offY, start[1]+offY]).astype(int)

        lu = np.array([np.min(polyx), np.min(polyy)])
        rb = np.array([np.max(polyx), np.max(polyy)])
        lu[lu<0] = 0
        if rb[0] > mapSz[1]:
            rb[0] = mapSz[1]
        if rb[1] > mapSz[0]:
            rb[1] = mapSz[0]
        outerMap = self.pathPlanMap[lu[1]:rb[1], lu[0]:rb[0]]
        [pointY, pointX] = np.where(outerMap == 1)

        pointX = pointX + lu[0]
        pointY = pointY + lu[1]
        iscollision = False

        v_b = goal[:2] - start[:2]
        n_v_b = (v_b[0] ** 2 + v_b[1] ** 2)
        if len(pointX) >= 1:
            v_a_x = pointX - start[0]
            v_a_y = pointY - start[1]
            a_dot_b = ( v_a_x * v_b[0] + v_a_y * v_b[1] ) / n_v_b
            v_e_x = v_a_x - a_dot_b * v_b[0]
            v_e_y = v_a_y - a_dot_b * v_b[1]
            distMat = np.sqrt(v_e_x ** 2 + v_e_y ** 2)
            filter = distMat < self.extendLCheck
            if np.sum(filter) > 0:
                iscollision = True
        return iscollision

    def checkPathReplan(self):

        idx = 0
        n = self.curPlannedPath.shape[0]

        distMat = np.sqrt( (self.curPlannedPath[idx+1:, 0] - self.curPlannedPath[idx:-1, 0]) ** 2 + \
            (self.curPlannedPath[idx+1:, 1] - self.curPlannedPath[idx:-1, 1]) ** 2 )
        interDist = 0
        i = idx
        p1 = self.curPlannedPath[0, :]
        goalPos = self.curPlannedPath[-1, :]

        if self.debug_pathcheck == True:
            plt.cla()
            plt.imshow(self.pathPlanMap)
            plt.plot(self.curPlannedPath[:, 0], self.curPlannedPath[:, 1], '-r')

        while i < n-1 and i >= idx:
            interDist += distMat[i-idx]
            if interDist > 20: #100cm
                interDist = 0
                p2 = self.curPlannedPath[i, :2].flatten()
                distToGoal = np.sqrt( (goalPos[0] - p2[0]) ** 2 + (goalPos[1] - p2[1]) ** 2)
                if distToGoal > 30: #150cm
                    path_replan = self.collisionDetection(p1, p2)
                    p1 = p2
                    if path_replan == True:
                        return True
                else:
                    break
            i += 1
        return False

    def run(self): # return value: 1: finish the exploration process, 2: continue exploring current UWB node, 3:set new goal to explore

        self.exploreStateMsg.state = 1
        self.currentExploreUwbNode = self.nextRegion
        self.chooseNextBestGoal()
        
        print("INFO: max entropy is ", self.optimalScore)
        if (self.optimalScore < self.minEntropy):## the environment is well explored
            self.exploreStateMsg.state = 3 # finish the exploration
            self.exploreStatePub.publish(self.exploreStateMsg)
            print('INFO: finish the mapping process, good job!')
            return 2

        status = 0
        if self.currentExploreUwbNode != self.nextRegion: ## need to publish the new path
            self.curPlannedPath = self.plannedPath.copy()
            self.sendTraj(1) # new path
            self.pathShow = self.finalPath.copy()
            status = 1
        else: # need to check if current path is occulused
            if self.checkPathReplan() == True:
                self.curPlannedPath = self.plannedPath.copy()
                self.sendTraj(1)
                self.pathShow = self.finalPath.copy()
                status = 1
                print("INFO: Replan the path!")
        print('INFO: go to region defined by node ', self.nextRegion, ' max entorpy:', self.optimalScore)
        return status

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
        return followPath[:-1, :2]

    def pathEntropy(self, path):
        if path is None:
            return -1

        entropyMean = 0
        entryCnt = 0
        regionPiexl = 5
        for j in np.arange(path.shape[0]):
            point = path[j, :]
            subsubMap = self.map[int(point[1]-regionPiexl):int(point[1]+regionPiexl), int(point[0]-regionPiexl):int(point[0]+regionPiexl)]
            entropy = subsubMap * np.log(subsubMap + 1e-10) + (1 - subsubMap) * np.log((1 - subsubMap) + 1e-10)

            entropyMean += - np.sum(entropy.flatten())
            entryCnt += subsubMap.shape[0] * subsubMap.shape[1]

        entropyMean = entropyMean / float(entryCnt)
        return entropyMean

    def chooseNextBestGoal(self):

        goal_pos = np.zeros((1, 2))  
        self.optimalScore = 100
        maxScore = -1
        lastBestScore = -1
        if self.curPlannedPath is not None:
            lastBestScore = self.pathEntropy(self.curPlannedPath)
        
        print("INFO: lastBestScore, ", lastBestScore)
        path_plan = RRT_PathPlanInit(self.pathPlanMap, self.startPos, self.extendL, self.minTorDist, self.scale, self.debug)
        for i in np.arange(len(self.uwbPosx)):
            idx = self.nodeID[i]

            goal_pos[0, 0] = self.uwbPosx[i]
            goal_pos[0, 1] = self.uwbPosy[i]
            plannedPath = self.doPathPlan(goal_pos, path_plan)
            entropyMean = self.pathEntropy(plannedPath)
            if entropyMean > maxScore:
                maxScore = entropyMean
                self.optimalScore = entropyMean

                # record the best path in current frame, for path replan 
                self.finalPath = np.zeros((plannedPath.shape[0], 2))
                self.finalPath[:, 0] = (plannedPath[:, 0] + self.activeRegion[0]) * self.mapParam.resol - self.offset
                self.finalPath[:, 1] = (plannedPath[:, 1] + self.activeRegion[2]) * self.mapParam.resol - self.offset

                # record the planned path within the coordinate of interested region
                self.plannedPath = plannedPath.copy()

                # check if need to change the explore target, so change the explore ID
                if maxScore > 1.5 * lastBestScore and lastBestScore < self.minEntropy * 2:
                    self.nextRegion = idx

    def sendTraj(self, state):

        # publish the path to motion planning    
        self.trajMsg.path_posx = self.finalPath[:, 0].flatten()
        self.trajMsg.path_posy = self.finalPath[:, 1].flatten()
        self.trajMsg.pathFlag = state
        self.trajPub.publish(self.trajMsg)

    def publishTraj(self):
        # publish the path to rviz
        if self.pathShow is None:
            return

        vMarker = util.creatMarker(self.tfFrame.map, [1, 0, 0], 2, [0.05, 1, 1], 5)
        for i in np.arange(self.pathShow.shape[0]):
            point = geometry_msgs.msg.Point()
            point.x = self.pathShow[i, 0]
            point.y = self.pathShow[i, 1]
            vMarker.points.append(point)
        self.marker_pub.publish(vMarker)

if __name__ == '__main__':
    
    rospack = rospkg.RosPack()
    config_file = rospack.get_path('my_simulations') + '/config/system.cfg'
    with io.open(config_file, 'r', encoding='utf-8') as f:
        config = libconf.load(f)

    # initialize ros
    rospy.init_node('goalAssignNode', anonymous=False)

    regionObj = util.UWBObservation(config)
    mapObj = util.MapObsObj(config)
    goalAssign = GoalAssign(config, mapObj, regionObj)

    print("INFO: wait for uwb initalization")
    while not regionObj.uwbInit and not rospy.is_shutdown():
        time.sleep(0.1)
    regionObj.getActiveRegion()
    print('INFO: UWB region is initialized!')
    rate = rospy.Rate(0.5)  # 1hz

    # pathPlannedPath = os.getenv("HOME") + '/Slam_test/share_asus/auto_explore/' + envName + "/"
    # if not os.path.isdir(pathPlannedPath):
    #     os.makedirs(pathPlannedPath)

    exploreState = False
    while not rospy.is_shutdown(): # and mapStateObj.mapState != 5:

        # mapStateObj.state =: 1, map initialization is finished but did not find the optimal scan matching, 2, indicate the first frame of map initialization
        # 4, find the optimal point
        print("New step -----")
        # prepare the required sensor data
        if goalAssign.prepareData() == False:
            rate.sleep()
            continue

        # print("INFO: pathPlanMap: ", goalAssign.pathPlanMap.shape)

        if goalAssign.startPos is None:
            rate.sleep()
            continue
            
        start_time = time.time()
        exploreState = goalAssign.run()
        print("INFO: time elapsed: ", time.time() - start_time)
        
        if exploreState == 2: # finish the exploretion
            break
        rate.sleep()


