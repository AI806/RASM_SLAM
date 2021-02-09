#!/usr/bin/env python3
import struct
import time
import itertools
import numpy as np
import copy
import numpy.matlib
from my_simulations.msg import uwbRangeMsg
from my_simulations.msg import uwb_sensor
import rospy
import rospkg
import io, os
import threading
import libconf

class rangePublisher(object):

    def __init__(self, config):

        self.fixConstraint = config.default.pfParam.distUWBOnRobot
        self.nodeCnt = config.default.exploration.nodeCnt
        pubTopic = config.default.uwb_range_measure.realMessageTopic
        subTopic = config.default.uwb_range_measure.simulateMessageTopic

        self.axisNodes = np.array(config.default.uwb_range_measure.axisNodes)

        self.r_tab = np.zeros((self.nodeCnt, self.nodeCnt))
        self.r_cov = np.zeros((self.nodeCnt, self.nodeCnt))
        self.r_los = np.zeros((self.nodeCnt, self.nodeCnt))
        self.nodeIds = np.zeros((1, self.nodeCnt))
        self.var_zero_r = 1e30
        self.mea_var = 0.01
        self.msgCnt = 0
        self.mutex = threading.Lock()
        self.rangeMsg = uwbRangeMsg()
        rospy.Subscriber(subTopic, uwb_sensor, self.robot_range_msg_callback)
        self.rangePub = rospy.Publisher(pubTopic, uwbRangeMsg, queue_size = 20)
            
    def robot_range_msg_callback(self, msg):
        
        rangeCnt = len(msg.rangeIndx)
        self.mutex.acquire()
        self.msgCnt += 1
        self.nodeIds = np.array(msg.ids)
        self.rangeMsg.count = len(self.nodeIds)
        ## convert the range measures to range table
        for i in np.arange(rangeCnt):
            m = int(msg.rangeIndx[i] / msg.codec)
            n = int(msg.rangeIndx[i] % msg.codec)
            self.r_tab[m, n] = self.r_tab[n, m] = msg.ranges[i]
            if msg.los[i] == False: 
                self.r_cov[m, n] = self.r_cov[n, m] = self.var_zero_r
                self.r_los[m, n] = self.r_los[n, m] = False
            else:
                self.r_cov[m, n] = self.r_cov[n, m] = self.mea_var
                self.r_los[m, n] = self.r_los[n, m] = True

        axis_index_1 = msg.ids.index(self.axisNodes[0])
        if axis_index_1 != 1:
            self.changeOrder(1, axis_index_1)
        
        axis_index_2 = list(self.nodeIds).index(self.axisNodes[1])
        if axis_index_2 != 2:
            self.changeOrder(2, axis_index_2)

        nodeCnt = len(msg.ids)
        robot_node_1 = list(self.nodeIds).index(msg.maxUwbNo)
        if robot_node_1 != 0:
            self.changeOrder(0, robot_node_1)
            self.nodeIds[0] = nodeCnt-2
            self.rangeMsg.ap = nodeCnt-2
        robot_node_2 = list(self.nodeIds).index(msg.maxUwbNo + 1)
        if robot_node_2 != 3:
            self.changeOrder(3, robot_node_2)
            self.nodeIds[3] = nodeCnt-1
            self.rangeMsg.cn = nodeCnt-1

        # print(self.nodeIds)
        self.r_cov[0, 3] = self.r_cov[3, 0] = 0.00001
        self.r_tab[0, 3] = self.r_tab[3, 0] = self.fixConstraint
        self.r_los[0, 3] = self.r_los[3, 0] = False
        self.mutex.release()
    
    def changeOrder(self, idx1, idx2):
        self.r_tab[ [idx1, idx2], :] = self.r_tab[ [idx2, idx1], :]
        self.r_tab[ :, [idx1, idx2]] = self.r_tab[ :, [idx2, idx1],]
        self.r_cov[ [idx1, idx2], :] = self.r_cov[ [idx2, idx1], :]
        self.r_cov[ :, [idx1, idx2]] = self.r_cov[ :, [idx2, idx1],]

        self.nodeIds[[idx1, idx2]] = self.nodeIds[[idx2, idx1]]

    def run(self):

        self.rangeMsg.timestamp = rospy.Time.now()

        self.mutex.acquire()
        if self.msgCnt > 0:
            self.msgCnt = 0

            # print("-----------------")
            # print(self.nodeIds)
            # print(self.r_tab)
            # print(self.r_los)
            self.rangeMsg.state = 2 # has valida data
            self.rangeMsg.ranges = list(self.r_tab.flatten())
            self.rangeMsg.r_cov = list(self.r_cov.flatten())
            self.rangeMsg.ids = list(self.nodeIds.flatten())
        else:
            self.rangeMsg.state = 1 #NO UWB data
        self.mutex.release()
        self.rangePub.publish(self.rangeMsg)
        

if __name__ == '__main__':

    rospack = rospkg.RosPack()
    config_file = rospack.get_path('my_simulations') + '/config/system.cfg'
    with io.open(config_file, 'r', encoding='utf-8') as f:
        config = libconf.load(f)

    rospy.init_node('uwbRangeNode', anonymous=False)
    rPub = rangePublisher(config)

    fps = 10
    rate = rospy.Rate(fps)  # 10hz
    
    while not rospy.is_shutdown():
        rPub.run()
        rate.sleep()
