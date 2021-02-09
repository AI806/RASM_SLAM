#!/usr/bin/env python
from my_simulations.msg import *
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from visualization_msgs.msg import *

import numpy as np
import time
import rospy
import libconf
import io
import rospkg
from dwa_occ_avoid import DynamicWindowApproach
import util_tools as util
import scipy.io as sio

class MotionPlanning(object):

    def __init__(self, config, lidarMsg, odometryMsg, trajMsg, posRectify, mapMsg, activeRegion):

        self.isSimulation = config.default.gazebo.simulation

        robotParam = config.default.sRobotPara
        dwaParam = config.default.dwa_motion
        self.mapParam = config.default.occMap
        self.tfFrame = config.default.tf
        self.oneMeterPixels = robotParam.oneMeterPixels
        self.lidar_offsetx = robotParam.lidar_offsetx
        self.lidar_offsety = robotParam.lidar_offsety
        self.fWheelDistance = robotParam.fWheelDistance

        # parameter for subscribed msg
        self.lidarMsg = lidarMsg
        self.odometryMsg = odometryMsg
        self.trajMsg = trajMsg
        self.posRectify = posRectify
        self.mapMsg = mapMsg
        self.activeRegion = activeRegion

        # parameter for publishing speed
        if self.isSimulation:
            self.velMsg = Twist()
            velTopic = config.default.robot_vel_ctrl.aMessageTopic_simu
            self.velPub = rospy.Publisher(velTopic, Twist, queue_size = 20)
        else:
            self.velMsg = velocityCtrl()
            velTopic = config.default.robot_vel_ctrl.aMessageTopic
            self.velPub = rospy.Publisher(velTopic, velocityCtrl, queue_size = 20)

        # parameter for map
        self.resol = int(1 / self.mapParam.resol)
        self.offset = int(self.mapParam.mapOffset)

        # parameter for robot control
        velCtrParam = config.default.robot_vel_ctrl
        self.stopSpeed = velCtrParam.stopSpeed
        self.turnSpeed = 2 * velCtrParam.turnSpeed / self.fWheelDistance
        self.mintTurnAng = velCtrParam.mintTurnAng * np.pi / 180.0
        self.stopDist = velCtrParam.stopDist # in m
        self.obsCtAngular = np.zeros((velCtrParam.stopFrame, 1))

        # parameter for dwa
        self.deltaT = dwaParam.deltaT
        self.reachDist = dwaParam.reachDist # in meter, 1.5 for indoor, 2.5 for outdoor
        
        # max v, max omega, acc v, acc omega, resoultion v, resol omega
        self.Kinematic = np.array([0.6, self.toRadian(60.0), 0.8, self.toRadian(400.0), 0.02, self.toRadian(1)])
        # heading, dist, vel
        virtualAppTime = 3
        # self.evalParamNomal = np.array([0.05, 0.2, 0.3, virtualAppTime])
        self.evalParamNomal = np.array([0.08, 0.2, 0.1, virtualAppTime])
        self.minDWAControlDist = dwaParam.minDWAControlDist

        # parameter to define emergent stop region
        self.c = robotParam.stop_front
        self.a = - self.c / (robotParam.stop_side ** 2)
        
        obstacleR = 0.5
        self.dwa = DynamicWindowApproach(self.evalParamNomal, self.Kinematic, obstacleR, self.deltaT)

        # parameters for odometry
        self.vel = 0
        self.oemga = 0
        
        # parameter for laser range
        self.lidarRange = None
        self.lidarAngular = None

        # parameter for trajectory
        self.path_posx = None
        self.path_posy = None
        self.newPath = -1
        self.pathIdx = 0
        self.pathInit = False
        self.trajVisual = None

        # parameter for estimated state of robot
        self.robotPos = None

        # parameter for map
        self.activeMap = None

        # visualization
        visualTopic = config.default.visualize.dwa_traj
        self.traj_pub = rospy.Publisher(visualTopic, visualization_msgs.msg.Marker, queue_size=1)
        visualTopic = config.default.visualize.ref_goal
        self.ref_goal_pub = rospy.Publisher(visualTopic, visualization_msgs.msg.Marker, queue_size=1)

        # for debuging
        self.debug = False
        if self.debug == True:
            self.dataRecord = util.RecordingData()

    def toRadian(self, theta):
        return theta * np.pi / 180.0

    def prepareData(self):

        # robot state
        self.posRectify.mutex.acquire()
        if self.posRectify.msgUpdate == False:
            self.posRectify.mutex.release()
            return False
        # print("UWB pos ok!")
        self.robotPos = self.posRectify.robotPos.copy()
        self.posRectify.mutex.release()

        # lidar
        self.lidarMsg.mutex.acquire()
        if self.lidarMsg.msgUpdate == False:
            self.lidarMsg.mutex.release()
            return False
        # print("LiDAR range ok!")
        self.lidarRange = self.lidarMsg.lidarRanges.copy()
        self.lidarAngular = self.lidarMsg.lidarAngular.copy()
        self.obstacles = self.getLidarInfo()
        self.lidarMsg.mutex.release()
        
        # odom
        self.odometryMsg.mutex.acquire()
        if self.odometryMsg.msgUpdate == False:
            self.odometryMsg.mutex.release()
            return False
        # print("odometry ok!")
        self.vel = self.odometryMsg.vel
        self.omega = self.odometryMsg.omega
        self.odometryMsg.mutex.release()

        # trajectory
        self.trajMsg.mutex.acquire()
        if self.trajMsg.msgUpdate == True:
        #     self.newPath = False
        #     self.trajMsg.mutex.release()
        #     return False
        # else:
            self.pathInit = True
            self.newPath = True
            self.path_posx = self.trajMsg.path_posx
            self.path_posy = self.trajMsg.path_posy
        # print("Traj ok!")
        self.trajMsg.mutex.release()

        # map
        self.mapMsg.mutex.acquire()
        if self.mapMsg.msgUpdate == True:
            self.activeMap = self.mapMsg.activeMap.copy()
        self.mapMsg.mutex.release()

        # reset the update flag to receive new data
        self.odometryMsg.reverseUpdate()
        self.trajMsg.reverseUpdate()
        self.posRectify.reverseUpdate()
        self.mapMsg.reverseUpdate()
        self.lidarMsg.reverseUpdate()
        return True

    def getLidarInfo(self):

        theta = self.robotPos[2]
        # filter the invalid range
        filt = np.logical_or(np.isinf(self.lidarRange), self.lidarRange<0.1)
        ranges = self.lidarRange[np.logical_not(filt)]
        angle_array = self.lidarAngular[np.logical_not(filt)]

        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        off_x = np.cos(angle_array) * ranges + self.lidar_offsetx # in m
        off_y = np.sin(angle_array) * ranges + self.lidar_offsety

        obstaclesToRobot = np.vstack((off_x, off_y))
        obstacles = np.dot(R, obstaclesToRobot) + np.tile(np.array([[self.robotPos[0]], [self.robotPos[1]]]), \
            (1, obstaclesToRobot.shape[1])) # in global coordinate, 1cm

        return obstacles.T # return N x 2

    def stopRobot(self):

        stoppedFlag = False
        u = np.zeros(2)
        if np.abs(self.vel) < 0.1 and np.abs(self.omega) < (5 * np.pi / 180.):
            stoppedFlag = True
        else:
            u[0] = self.vel * self.stopSpeed
            u[1] = self.omega * self.stopSpeed
        return stoppedFlag, u

    def publishVel(self, u, refGoal):

        if self.isSimulation: # gazebo
            self.velMsg.linear.x = u[0]
            self.velMsg.angular.z = u[1]
        else: # robot
            self.velMsg.wheel_vel_left = u[0] - (self.fWheelDistance/2.0)*u[1]
            self.velMsg.wheel_vel_right = u[0] + (self.fWheelDistance/2.0)*u[1]
        self.velPub.publish(self.velMsg)

        if self.trajVisual is not None:
            vMarker = util.creatMarker(self.tfFrame.map, [0, 1, 0], 2, [0.08, 1, 1], 6)
            # print("self.trajVisual: ", self.trajVisual)
            for i in np.arange(self.trajVisual.shape[0]):
                point = geometry_msgs.msg.Point()
                point.x = self.trajVisual[i, 0]
                point.y = self.trajVisual[i, 1]
                vMarker.points.append(point)
            self.traj_pub.publish(vMarker)

        if refGoal is not None:
            vMarker = util.creatMarker(self.tfFrame.map, [0, 0, 1], 1, [0.5, 0.5, 0.5], 7)
            point = geometry_msgs.msg.Point()
            point.x = refGoal[0]
            point.y = refGoal[1]
            vMarker.points.append(point)
            self.ref_goal_pub.publish(vMarker)
            
    def updateThePath(self):
        
        pathLen = len(self.path_posx)
        followPath = np.zeros((pathLen, 2))
        followPath[:, 0] = np.array(self.path_posx) / self.mapParam.resol # in meter
        followPath[:, 1] = np.array(self.path_posy) / self.mapParam.resol
        self.newPath = False
        self.pathIdx = 0

    def turnControlling(self, target_theta):

        # get the heading of the robot, check if need to turn the robot
        curAngle = self.robotPos[2]
        if curAngle > np.pi:
            curAngle = curAngle - 2*np.pi
        if curAngle < -np.pi:
            curAngle = curAngle + 2*np.pi

        if target_theta > np.pi:
            target_theta = target_theta - 2*np.pi
        if target_theta < -np.pi:
            target_theta = target_theta + 2*np.pi

        # find the different between robot's current heading and the target heading
        diff_theta = curAngle - target_theta
        if diff_theta > np.pi:
            diff_theta = - diff_theta + 2*np.pi
        if diff_theta < -np.pi:
            diff_theta = diff_theta + 2*np.pi

        # print("INFO: curAngle, ", curAngle, " target_theta, ", target_theta, " diff, ", diff_theta)
        # if not reach the required minimal angular difference
        u = np.zeros(2)
        reachFlag = False
        if np.abs(diff_theta) > self.mintTurnAng:
            u[1] = np.sign(diff_theta) * self.turnSpeed
        else:
            reachFlag = True
        
        self.trajVisual = None
        return reachFlag, u

    def checkEmergencyStop(self):
        stopFlag = False
        if len(self.lidarRange) > 0:
            filt = np.logical_and( self.lidarRange > 0.05, self.lidarRange < 2 ) 
            ranges = self.lidarRange[filt]
            ang = self.lidarAngular[filt]
            obsx = ranges * np.cos(ang) + self.lidar_offsetx
            obsy = ranges * np.sin(ang) + self.lidar_offsety
            met = self.a * (obsx ** 2) + self.c
            r = obsx < met
            if np.sum(r.flatten()) > 0:
                stopFlag = True
                self.trajVisual = None
        return stopFlag

    def pickRefPos(self):
        # no valid path
        pathLen = len(self.path_posx)
        if pathLen <= self.pathIdx:
            return None, False
        
        # reach the last point in the path
        if pathLen == self.pathIdx + 1:
            return np.array([self.path_posx[-1], self.path_posy[-1]]), True

        cumDist = np.sqrt( (self.robotPos[0]-self.path_posx[self.pathIdx])**2 + (self.robotPos[1]-self.path_posy[self.pathIdx])**2 )
        aDist = np.sqrt( (self.path_posx[self.pathIdx:-1] - self.path_posx[self.pathIdx+1:]) ** 2 + \
                (self.path_posy[self.pathIdx:-1] - self.path_posy[self.pathIdx+1:]) ** 2 ) # lent(aDist) >= 1

        idx = 0
        while cumDist < self.minDWAControlDist and self.pathIdx < pathLen-1:
            cumDist += aDist[idx]
            idx += 1
            self.pathIdx += 1

        refPos = np.zeros(2)
        refPos[0] = self.path_posx[self.pathIdx]
        refPos[1] = self.path_posy[self.pathIdx]

        return refPos, True

    def dwaControl(self):

        u = np.zeros(2)
        [refPos, state] = self.pickRefPos()
        if state == False:
            return u, None

        robotState = np.zeros((5,1)).flatten()
        robotState[:3] = self.robotPos[:3]
        robotState[3] = self.vel
        robotState[4] = self.omega

        if self.debug == True:
            self.dataRecord.setValue(None, None, robotState, refPos, self.obstacles)

        [u, traj] = self.dwa.dwaMethod(robotState, refPos, self.obstacles)
        self.trajVisual = traj

        if self.debug == True:
            self.dataRecord.setValue(u, traj, None, None, None)
        
        if traj is None:
            return u, None
        else:
            return u, refPos
    
    def saveData(self, savePath):
        sio.savemat(savePath, {'u':self.dataRecord.u, 'traj': self.dataRecord.traj, \
            'robotState':self.dataRecord.robotState, 'refPos':self.dataRecord.refPos, \
            'obstacles':self.dataRecord.obstacles})

def listener():

    rospy.init_node('core_algorithm_model', anonymous=False)
    fps = 10.0
    rate = rospy.Rate(fps)  # 10hz

    rospack = rospkg.RosPack()
    config_file = rospack.get_path('my_simulations') + '/config/system.cfg'
    with io.open(config_file, 'r', encoding='utf-8') as f:
        config = libconf.load(f)

    lidarMsg = util.LaserObservation(config)
    odometryMsg = util.OdometryObservation(config)
    trajMsg = util.GlobalTrajectory(config)
    posRectify = util.UwbPosRectify(config)
    mapMsg = util.MapObsObj(config)
    uwbMsg = util.UWBObservation(config)

    print("Waiting for the initialization of explored region!")
    while not uwbMsg.uwbInit and not rospy.is_shutdown():
        time.sleep(0.1)
    uwbMsg.getActiveRegion()
    activeRegion = uwbMsg.activeRegion.copy()
    mapMsg.setActiveReqion(activeRegion)

    uwbMsg.closeObj()
    print("Getting the region for exploration")

    # initial the object of motion planning
    motionPlan = MotionPlanning(config, lidarMsg, odometryMsg, trajMsg, posRectify, mapMsg, activeRegion)

    robotStateIdx = 1

    if motionPlan.debug == True:
        savePath = "/home/guanmy/gazebo_ws/dataRcord/" + "debuging"
    while not rospy.is_shutdown():
        # velMsg.state = 1 ## 1 for normal control, 2 for stop, 3 for reach destination, 4 for emgency stop, 5 for turning
        # prepare the sensor datas
        print("---------New step-----------")
        refGoal = None
        u = np.zeros(2)
        dataReady = motionPlan.prepareData()

        if motionPlan.pathInit == False:
            motionPlan.publishVel(u, refGoal)
            print("INFO: Waiting for the global path!")
            rate.sleep()
            continue
        
        if dataReady == False:
            time.sleep(0.05)
            continue
        
        # new path coming?
        if motionPlan.newPath == True:
            motionPlan.updateThePath()
            target_theta = np.arctan2( motionPlan.path_posy[1] - motionPlan.path_posy[0], \
                motionPlan.path_posx[1] - motionPlan.path_posx[0] )
            robotStateIdx = 4
            print('Info: new path comes, running -> stopping') 

        if robotStateIdx == 1: # running
            [u, refGoal] = motionPlan.dwaControl()
            if refGoal is None:
                print("raise the exception!")
                break
            # u[1] = -u[1]

        elif robotStateIdx == 2: # turning
            [reachFlag, u] = motionPlan.turnControlling(target_theta)
            if reachFlag == True:
                robotStateIdx = 3
                print('Info: end the turning process, turning -> stopping')
                continue

        elif robotStateIdx == 3: # stopping
            [reachFlag, u] = motionPlan.stopRobot()
            # u[1] = -u[1]
            if reachFlag == True:
                robotStateIdx = 1
                print('Info: robot stopped, stopping -> running')

        elif robotStateIdx == 4: # stopping
            [reachFlag, u] = motionPlan.stopRobot()
            # u[1] = -u[1]
            if reachFlag == True:
                robotStateIdx = 2
                print('Info: robot stopped, stopping -> turning')

        print("INFO: control command ", u, " running state: ", robotStateIdx)
        motionPlan.publishVel(u, refGoal)
        rate.sleep()

    if motionPlan.debug == True:
        # save the data for debuging
        print("Saving the data!")
        motionPlan.saveData(savePath)            

if __name__ == '__main__':
    listener()
