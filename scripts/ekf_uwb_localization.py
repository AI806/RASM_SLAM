#!/usr/bin/env python3
from my_simulations.msg import *
import geometry_msgs.msg
import visualization_msgs.msg
import numpy as np
import time
import rospy
from slam_ekf_v2 import *
import serial
import serial.tools.list_ports as prtlst
import libconf
import io
import rospkg
import scipy.io as sio
import io, os
# import pickle
import threading
import tf2_ros
import tf
import util_tools as util

class UwbLocalization(object):

    def __init__(self, config, posRectMsg, rangeMsg):
        uwbTopic = config.default.uwb_localization.aMessageTopic
        self.uwbPub = rospy.Publisher(uwbTopic, uwbLocalization, queue_size = 20)
        self.uwbMsg = uwbLocalization()

        # main parameter structures
        self.pfParam = config.default.pfParam
        self.exploreParam = config.default.exploration
        self.robotParam = config.default.sRobotPara
        self.tfFrame = config.default.tf
        self.mapParam = config.default.occMap

        self.halfFixConstrain = self.pfParam.distUWBOnRobot / 2.0
        self.angularOffset = self.exploreParam.angularOffset * np.pi / 180
        self.fixNodes = np.array(config.default.uwb_range_measure.fixNodes)
        self.axisIdx = np.array(config.default.uwb_range_measure.axisNodes)

        # for mutex variabes among threads  
        self.mutex = threading.Lock()

        # messages subscribed from external modules
        self.posRectMsg = posRectMsg
        self.rangeMsg = rangeMsg

        # flag to indicate the initial state of ekf
        self.ekfInitial = False
        self.isAxisInitialized = False

        self.cntTime = 0

        # init the public vairables
        self.uwbIds = []
        self.uwbCount = 0
        self.uwbRanges = []
        self.uwbRcov = []
        self.uwbState = -1
        self.uwbAp = []
        self.uwbCn = []
        self.rectifyPos = []
        self.rectifyState = -1
        self.ac_out = None

        #initializing the tf
        # self.br = tf2_ros.TransformBroadcaster()

        self.tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tfBuffer)

        # self.uMap_uwb_joint = geometry_msgs.msg.TransformStamped()
        # self.uMap_uwb_joint.header.frame_id = self.tfFrame.map
        # self.uMap_uwb_joint.child_frame_id = self.tfFrame.odom

        # self.centered_uwb_pose = geometry_msgs.msg.PoseStamped()
        # self.centered_uwb_pose.header.frame_id = self.tfFrame.uwb
        # self.centered_uwb_pose.pose.orientation.x = 0
        # self.centered_uwb_pose.pose.orientation.y = 0
        # self.centered_uwb_pose.pose.orientation.z = 0
        # self.centered_uwb_pose.pose.orientation.w = 0

        # self.centered_uwb_pose.pose.position.x = self.robotParam.uwb_offsetx
        # self.centered_uwb_pose.pose.position.y = self.robotParam.uwb_offsetz
        # self.centered_uwb_pose.pose.position.z = 0

        # data for visualization
        visualTopic = config.default.visualize.uwb_points
        self.marker_pub = rospy.Publisher(visualTopic, visualization_msgs.msg.MarkerArray, queue_size=1)

    def changeIndexSorting(self, n, idx1, idx2):
        id_idx = np.arange(n)
        id_idx = np.delete(id_idx, [idx1, idx2], 0)
        new_id_idx = np.zeros((n, 1)).flatten()
        new_id_idx[:2] = [idx1, idx2]
        new_id_idx[2:] = id_idx
        return new_id_idx.astype(int)

    def reverseIDinRangeTable(self, r_tab, src, dst): # dst < src
        if src == dst:
            return r_tab
        r = r_tab.shape[0]
        new_tab = r_tab.copy()
        # change row
        new_tab[dst, :] = r_tab[src, :]
        tmp_tab = np.delete(r_tab, np.s_[src], axis=0)
        idx = 0
        for i in np.arange(r):
            if i == dst:
                continue
            new_tab[i, :] = tmp_tab[idx, :]
            idx += 1

        # change col
        r_tab = new_tab.copy()
        new_tab[:, dst] = r_tab[:, src]
        tmp_tab = np.delete(r_tab, np.s_[src], axis=1)
        idx = 0
        for i in np.arange(r):
            if i == dst:
                continue
            new_tab[:, i] = tmp_tab[:, idx]
            idx += 1
        return new_tab

    def publishPoses(self, idx1, idx2, p2d, v2d):

        n = len(self.ac_out.id_record)
        self.uwbMsg.posx = self.ac_out.m_k[0:n] 
        self.uwbMsg.posy = self.ac_out.m_k[n:2*n] 
        leftUwbPos = np.array([self.uwbMsg.posx[idx1], self.uwbMsg.posy[idx1]])
        rightUwbPos = np.array([self.uwbMsg.posx[idx2], self.uwbMsg.posy[idx2]])
        middelPos = ((leftUwbPos + rightUwbPos) / 2.)
        vecLR = leftUwbPos - rightUwbPos

        # the reason of minuing pi/2 is that the real heading of the robot is vertical with the line connection of node 1 and node 2
        Theta = np.arctan2(vecLR[1, 0], vecLR[0, 0]) - np.pi / 2.0

        self.uwbMsg.theta = Theta # - angularOffset
        self.uwbMsg.posx = np.delete(self.uwbMsg.posx, [idx1, idx2], 0)
        self.uwbMsg.posy = np.delete(self.uwbMsg.posy, [idx1, idx2], 0)

        new_id_idx = self.changeIndexSorting(n, idx1, idx2)
        tmp_record = np.array(self.ac_out.id_record).astype(int)
        idRecord = tmp_record[new_id_idx]
        self.uwbMsg.node_id = self.ac_out.id_record

        [idx1, idx2, missingNodes] = self.ac_out.find_axisIdx(list(idRecord), self.axisIdx)
        if missingNodes == False:
            p_k = np.diag(self.ac_out.p_k_r[:, 0])
            self.uwbMsg.axisID = np.array([idx1, idx2])
            self.uwbMsg.pK = p_k[:2, :2].reshape((1,4)).flatten()
        else:
            self.uwbMsg.state = 4
            print('INFO: some critical nodes missing!')

        # publish the tf tree from uMap to odom
        [tr, qua] = self.publishTransformation(middelPos, Theta)

        # prepare data for visualization in rviz
        # show uwb states
        # print len(self.uwbMsg.posx), len(idRecord), idRecord
        vMarkerArrray = visualization_msgs.msg.MarkerArray()
        vMarker = util.creatMarker(self.tfFrame.map, [0, 1, 0], 1, [0.3, 0.3, 0.3], 1)
        
        for i in np.arange(len(idRecord)):
            point = geometry_msgs.msg.Point()
            point.x = p2d[0, i]
            point.y = p2d[1, i]
            point.z = 0.1
            vMarker.points.append(point) 
        vMarkerArrray.markers.append(vMarker)

        # show robot state
        vMarker = util.creatMarker(self.tfFrame.map, [0, 1, 0], 0, [0.8, 0.1, 0.1], 2)
        vMarker.pose.position.x = tr[0]
        vMarker.pose.position.y = tr[1]
        vMarker.pose.orientation.x = qua[0]
        vMarker.pose.orientation.y = qua[1]
        vMarker.pose.orientation.z = qua[2]
        vMarker.pose.orientation.w = qua[3]
        vMarkerArrray.markers.append(vMarker)
        self.marker_pub.publish(vMarkerArrray)

    def publishTransformation(self, middelPos, Theta):
        
        Q = tf.transformations.quaternion_from_euler(0, 0, Theta)
        T1 = np.dot(tf.transformations.translation_matrix((middelPos[0], middelPos[1], 0.0)),
            tf.transformations.quaternion_matrix(Q)) # uMap - uwb_joint

        try:
            trans = self.tfBuffer.lookup_transform(self.tfFrame.uwb, self.tfFrame.robot, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("waiting until the required frame is connected!")
            return None, None
        tran = trans.transform.translation
        rot = trans.transform.rotation
        T2 = np.dot(tf.transformations.translation_matrix((tran.x, tran.y, tran.z)),
                    tf.transformations.quaternion_matrix(np.array([rot.x, rot.y, rot.z, rot.w]))) # uwb_joint - robot
        
        T3 = np.dot(T1, T2) # uMap - robot
        tr3 = tf.transformations.translation_from_matrix(T3)
        q3 = tf.transformations.quaternion_from_matrix(T3)

        # self.uMap_uwb_joint.header.stamp = rospy.Time.now() + rospy.Duration(1)
        # self.uMap_uwb_joint.transform.translation.x = tr3[0]
        # self.uMap_uwb_joint.transform.translation.y = tr3[1]
        # self.uMap_uwb_joint.transform.translation.z = tr3[2]
        # self.uMap_uwb_joint.transform.rotation.x = q3[0]
        # self.uMap_uwb_joint.transform.rotation.y = q3[1]
        # self.uMap_uwb_joint.transform.rotation.z = q3[2]
        # self.uMap_uwb_joint.transform.rotation.w = q3[3]
        # self.br.sendTransform(self.uMap_uwb_joint)

        self.uwbMsg.robotPosx = tr3[0]
        self.uwbMsg.robotPosy = tr3[1]
        eular = tf.transformations.euler_from_quaternion(q3)
        self.uwbMsg.theta = eular[2]

        return tr3, q3

    def prepareSensorData(self):

        # prepare range measurements
        self.rangeMsg.mutex.acquire()
        if self.rangeMsg.msgUpdate == False:
            self.rangeMsg.mutex.release()
            return 0
        self.uwbIds = list(self.rangeMsg.ids)
        self.uwbCount = self.rangeMsg.count
        self.uwbRanges = np.array(self.rangeMsg.ranges).reshape(self.uwbCount, self.uwbCount)
        self.uwbRcov = np.array(self.rangeMsg.r_cov).reshape(self.uwbCount, self.uwbCount)
        self.uwbState = self.rangeMsg.state
        self.uwbAp = self.rangeMsg.ap
        self.uwbCn = self.rangeMsg.cn
        self.rangeMsg.reverseUpdate()
        self.rangeMsg.mutex.release()

        # prepare rectify infors
        self.posRectMsg.mutex.acquire()
        if self.posRectMsg.msgUpdate == False:
            self.rectifyState = -1
        else:
            self.rectifyPos = np.array(self.posRectMsg.rectifypos)
            self.rectifyState = self.posRectMsg.state
            self.posRectMsg.reverseUpdate()
        self.posRectMsg.mutex.release()

        if self.ekfInitial == False:
            self.ekfInitial = True
            self.ac_out = slam_ekf([self.uwbAp], [self.uwbCn], self.axisIdx, self.fixNodes, self.uwbIds[1:])
            self.ac_out.initial(self.uwbRanges, self.uwbRcov)

        return 1

    # 1 for initial stage, 
    # 2 for intialized, 
    # 3 for normal state
    # 4 for some nodes is not connected
    def run(self):
        self.uwbMsg.state = 1

        ## prepare the data for EKF
        self.prepareSensorData()

        visualization = False
        # if self.isAxisInitialized == False:
        #     visualization = True

        # run ekf core algorithm
        [p2d, v2d, missingNodes] = self.ac_out.run(self.uwbRanges, self.uwbRcov, visualization)

        # check if the initilization finishs
        self.record_cnt = len(self.ac_out.id_record)
        if self.isAxisInitialized == False and self.record_cnt == len(self.uwbIds) and \
            np.abs(np.mean(self.ac_out.err_std_mtx[:self.record_cnt, :self.record_cnt].flatten()) - 0.5) < 1e-5:
            self.isAxisInitialized = True
            self.uwbMsg.state = 3

        # prepare to publish the estimated poses
        if missingNodes == False:
            [idx1, idx2, missingNodes] = self.ac_out.find_axisIdx(self.ac_out.id_record, self.fixNodes)
            if missingNodes == False:
                self.publishPoses(idx1, idx2, p2d, v2d)
                if self.isAxisInitialized == True:
                    self.uwbMsg.state = 3
                else:
                    self.uwbMsg.state = 2
            else:
                self.uwbMsg.state = 4
        else:
            self.uwbMsg.state = 4

        # # check if need to rectify the UWB estimations
        if self.rectifyState == 3 and len(self.rectifyPos) > 0:
            self.rectifyState = 0
            # rectifypos, 
            # 0: x in meter, 
            # 1: y in meter,
            # 2: vr in m/s, 
            # 3: vl in m/s, 
            # 4: heading in rad, 
            # 5: node count
            self.ac_out.setFeedback(self.rectifyPos, self.halfFixConstrain)
            self.ac_out.run(self.uwbRanges, self.uwbRcov, False)
            print('INFO: the index of rectified frames', self.cntTime)
            self.cntTime += 1

        self.uwbMsg.timestamp = rospy.Time.now()
        self.uwbPub.publish(self.uwbMsg)

def listener():

    rospack = rospkg.RosPack()
    config_file = rospack.get_path('my_simulations') + '/config/system.cfg'
    with io.open(config_file, 'r', encoding='utf-8') as f:
        config = libconf.load(f)

    recordIdx = config.default.dataRecord.recordIdx
    isRecord = config.default.dataRecord.isRecord

    envName = config.default.dataRecord.envName
    homePath = os.getenv("HOME") + '/Slam_test/share_asus/auto_explore/' + envName + "/"
    if not os.path.isdir(homePath):
        os.makedirs(homePath)

    rospy.init_node('ekf_uwb_localization', anonymous=False)
    
    fps = 10.0
    rate = rospy.Rate(fps)  # 10hz

    # for data recording
    curDir = None
    storedPos = None
    waiteCnt = 1

    # initializing the EKF
    rangeObs = util.RangeObservation(config)
    posRectify = util.UwbPosRectify(config)
    uwbEst = UwbLocalization(config, posRectify, rangeObs)

    print('INFO: waiting for the initializtion of UWB ranging!')
    while not rospy.is_shutdown() and rangeObs.state != 2:
        time.sleep(0.1)
    print('INFO: UWB ranging thread is initialized!')

    while not rospy.is_shutdown():

        start_time1 = time.time()
        if waiteCnt < 000:
            waiteCnt += 1
            rate.sleep()
            continue
    
        # core algorithm
        uwbEst.run()

        # wait for the ready of UWB estimation 
        if uwbEst.isAxisInitialized == False:
            continue

        # if uwbEst.isAxisInitialized == True:
        #     storedPos = np.zeros((uwbEst.record_cnt, 2))
        #     storedPos[:, 0] = np.array(uwbEst.ac_out.m_k[0:uwbEst.record_cnt]).flatten()
        #     storedPos[:, 1] = np.array(uwbEst.ac_out.m_k[uwbEst.record_cnt:2*uwbEst.record_cnt]).flatten()
        # if storedPos is not None:
        #     tmpPos = np.zeros((uwbEst.record_cnt, 2))
        #     tmpPos[:, 0] = np.array(uwbEst.ac_out.m_k[0:uwbEst.record_cnt]).flatten()
        #     tmpPos[:, 1] = np.array(uwbEst.ac_out.m_k[uwbEst.record_cnt:2*uwbEst.record_cnt]).flatten()
        #     id_record = uwbEst.ac_out.id_record
        #     storedPos = np.vstack((storedPos, tmpPos))

        rate.sleep()
        
    # if isRecord is True: # save the position of beacons into a mat file
    #     path = homePath + str(recordIdx) + '_uwb_pos_record'
    #     sio.savemat(path, {'pos': storedPos, 'nodeID': id_record})

if __name__ == '__main__':
    listener()
