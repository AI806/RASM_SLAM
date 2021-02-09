#!/usr/bin/env python
from my_simulations.msg import *
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

posRectify = statesAndRectify()
rangeMsg = uwbRangeMsg()
updateRange = True
updateOdo = True
updateUWB = True

def robot_range_msg_callback(msg):
    global rangeMsg
    if updateRange == True:
        rangeMsg = msg

def robot_mapping_msg_callback(msg):
    global posRectify
    if updateUWB:
        posRectify = msg

def getPozyxSerialPort():
    pts = prtlst.comports()
    cnt = 1
    for pt in pts:
        cnt += 1
        if 'STM32 Virtual' in pt.description or 'J-Link' in pt.description:
            return pt.device
    return None

def reverseIDinRangeTable(r_tab, src, dst): # dst < src
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

def changeIndexSorting(n, idx1, idx2):
    id_idx = np.arange(n)
    id_idx = np.delete(id_idx, [idx1, idx2], 0)
    new_id_idx = np.zeros((n, 1)).flatten()
    new_id_idx[:2] = [idx1, idx2]
    new_id_idx[2:] = id_idx
    return new_id_idx.astype(int)

def listener():

    global posRectify, odoMsg, updateRange, rangeMsg
    # general parameters
    oneMeterPixels = 100
    fps = 10
    two_pi = np.pi*2
    deltaT = 1. / fps

    rospack = rospkg.RosPack()
    config_file = rospack.get_path('my_simulations') + '/config/system.cfg'
    with io.open(config_file, 'r', encoding='utf-8') as f:
        config = libconf.load(f)

    # topic infors
    uwbTopic = config.default.uwb_localization.aMessageTopic
    posRectifyTopic = config.default.pfParam.aMessageTopic
    rangeTopic = config.default.uwb_range_measure.realMessageTopic
    
    fixConstrain = config.default.pfParam.distUWBOnRobot
    recordIdx = config.default.dataRecord.recordIdx
    isRecord = config.default.dataRecord.isRecord
    nodeCntRecord = config.default.exploration.nodeCnt

    halfFixConstrain = fixConstrain / 2.0
    angularOffset = config.default.exploration.angularOffset * np.pi / 180
    envName = config.default.dataRecord.envName

    uwb_offsetx = config.default.sRobotPara.lidar_offsetx
    uwb_offsety = config.default.sRobotPara.lidar_offsety
    uwb_offset = np.array([uwb_offsetx, uwb_offsety]).reshape(2, 1)

    rospy.init_node('ekf_uwb_localization', anonymous=False)
    rospy.Subscriber(posRectifyTopic, statesAndRectify, robot_mapping_msg_callback)
    rospy.Subscriber(rangeTopic, uwbRangeMsg, robot_range_msg_callback)
    uwbPub = rospy.Publisher(uwbTopic, uwbLocalization, queue_size = 20)
    uwbMsg = uwbLocalization()

    # store UWB beacons' positions
    # homePath = os.getenv("HOME") + '/Slam_test/UWB_Lidar_Slam/semi_auto_explore/' + envName + "/"
    homePath = os.getenv("HOME") + '/Slam_test/share_asus/auto_explore/' + envName + "/"
    if not os.path.isdir(homePath):
        os.makedirs(homePath)

    print('INFO: waiting for the initializtion of UWB ranging!')
    while not rospy.is_shutdown() and rangeMsg.state != 2:
        time.sleep(0.1)
    print('INFO: UWB ranging thread is initialized!')
    updateRange = False
    new_ids = list(copy.copy(rangeMsg.ids))
    ranges = np.array(rangeMsg.ranges).reshape(rangeMsg.count, rangeMsg.count)
    r_cov = np.array(rangeMsg.r_cov).reshape(rangeMsg.count, rangeMsg.count)
    updateRange = True
    # find ID of node chosen as master head
    fixNodes = np.array(config.default.uwb_range_measure.fixNodes)
    axisIdx = np.array(config.default.uwb_range_measure.axisNodes)

    ac_out = slam_ekf([rangeMsg.ap], [rangeMsg.cn], axisIdx, fixNodes, new_ids[1:])
    ac_out.initial(ranges, r_cov)

    rate = rospy.Rate(fps)  # 10hz
    cntTime = 0
    isAxisInitialized = False
    estimationCnt = 0
    uwbState = 1 # 1 for initial stage, 2 for intialized, 3 for some nodes is not connected
    curDir = None
    storedPos = None
    waiteCnt = 1
    middelPos = None
    while not rospy.is_shutdown():
        start_time1 = time.time()
        if waiteCnt < 000:
            waiteCnt += 1
            rate.sleep()
            continue

        updateUWB = False
        # or rectifiedUwbLocalization.state == 4) and \
        if (posRectify.state == 3) and len(posRectify.rectifypos) > 0:
            posRectify.state = 0

            # rectifiedUwbLocalization.rectifypos, 0: x in meter, 1: y in meter, 2: vr in m/s, 3: vl in m/s, 4: heading in rad, 5: node count
            stateFeedback = np.array(posRectify.rectifypos)
            ac_out.setFeedback(stateFeedback, halfFixConstrain)
            ac_out.run(ranges, r_cov, False)
            print('INFO: the index of rectified frames', cntTime)
            cntTime += 1
        updateUWB = True

        updateRange = False
        new_ids = list(copy.copy(rangeMsg.ids))
        ranges = np.array(rangeMsg.ranges).reshape(rangeMsg.count, rangeMsg.count)
        r_cov = np.array(rangeMsg.r_cov).reshape(rangeMsg.count, rangeMsg.count)
        updateRange = True

        if isAxisInitialized == False:
            [p2d, v2d, missingNodes] = ac_out.run(ranges, r_cov, True)
            record_cnt = len(ac_out.id_record)
            
            if record_cnt == len(new_ids) and np.abs(np.mean(ac_out.err_std_mtx[:record_cnt, :record_cnt].flatten()) - 0.5) < 1e-5:
                isAxisInitialized = True
                print('INFO: axis is in the right geometry state!')
                uwbState = 2
                storedPos = np.zeros((record_cnt, 2))
                storedPos[:, 0] = np.array(ac_out.m_k[0:record_cnt]).flatten()
                storedPos[:, 1] = np.array(ac_out.m_k[record_cnt:2*record_cnt]).flatten()
        else:
            [p2d, v2d, missingNodes] = ac_out.run(ranges, r_cov, False)
            if storedPos is not None:
                tmpPos = np.zeros((record_cnt, 2))
                tmpPos[:, 0] = np.array(ac_out.m_k[0:record_cnt]).flatten()
                tmpPos[:, 1] = np.array(ac_out.m_k[record_cnt:2*record_cnt]).flatten()
                id_record = copy.copy(ac_out.id_record)
                storedPos = np.vstack((storedPos, tmpPos))

        uwbMsg.state = uwbState
        uwbMsg.timestamp = rospy.Time.now()
        if missingNodes == False and isAxisInitialized == True:
            [idx1, idx2, missingNodes] = ac_out.find_axisIdx(ac_out.id_record, fixNodes)
            if missingNodes == False:
                n = len(ac_out.id_record)

                # fixNodes = [1, 2], 1: left node, 2 right node
                uwbMsg.posx = ac_out.m_k[0:n] 
                uwbMsg.posy = ac_out.m_k[n:2*n] 
                leftUwbPos = np.array([uwbMsg.posx[idx1], uwbMsg.posy[idx1]])
                rightUwbPos = np.array([uwbMsg.posx[idx2], uwbMsg.posy[idx2]])
                middelPos = ((leftUwbPos + rightUwbPos) / 2.)
                vecLR = leftUwbPos - rightUwbPos

                # the reason of minuing pi/2 is that the real heading of the robot is vertical with the line connection of node 1 and node 2
                Theta = np.arctan2(vecLR[1], vecLR[0]) - np.pi / 2.0 
                if Theta < 0:
                    Theta = Theta + 2 * np.pi
                elif Theta > 2*np.pi:
                    Theta = Theta - 2*np.pi

                uwbMsg.theta = Theta # - angularOffset

                # print 'cur_theta:', curDir
                uwbMsg.vx = (v2d[0, idx1] + v2d[0, idx2]) / 2.0
                uwbMsg.vy = (v2d[1, idx1] + v2d[1, idx2]) / 2.0

                uwbMsg.posx = np.delete(uwbMsg.posx, [idx1, idx2], 0)
                uwbMsg.posy = np.delete(uwbMsg.posy, [idx1, idx2], 0)

                new_id_idx = changeIndexSorting(n, idx1, idx2)
                tmp_record = np.array(ac_out.id_record).astype(int)
                idRecord = tmp_record[new_id_idx]

                r_tab = ac_out.r_tab_r
                r_tab = r_tab[0:n, 0:n]
                r_tab = reverseIDinRangeTable(r_tab, idx1, 0)
                r_tab = reverseIDinRangeTable(r_tab, idx2, 1)
                uwbMsg.R = r_tab.flatten()
                uwbMsg.mobileID = np.array([0, 1])
                uwbMsg.node_id = idRecord

                [idx1, idx2, missingNodes] = ac_out.find_axisIdx(list(idRecord), axisIdx)
                if missingNodes == False:
                    p_k = np.diag(ac_out.p_k_r[:, 0])
                    uwbMsg.axisID = np.array([idx1, idx2])
                    uwbMsg.pK = p_k[:2, :2].reshape((1,4)).flatten()
                else:
                    uwbMsg.state = 3
                    print('INFO: some critical nodes missing!')
            else:
                uwbMsg.state = 3
                print('INFO: some critical nodes missing!')
        else:
            [idx1, idx2, missingNodes] = ac_out.find_axisIdx(ac_out.id_record, fixNodes)
            if missingNodes == False:
                n = len(ac_out.id_record)
                uwbMsg.posx = ac_out.m_k[0:n]
                uwbMsg.posy = ac_out.m_k[n:2*n]
                leftUwbPos = np.array([uwbMsg.posx[idx1], uwbMsg.posy[idx1]])
                rightUwbPos = np.array([uwbMsg.posx[idx2], uwbMsg.posy[idx2]])
                middelPos = ((leftUwbPos + rightUwbPos) / 2.)
                vecLR = leftUwbPos - rightUwbPos

                # the reason of minuing pi/2 is that the real heading of the robot is vertical with the line connection of node 1 and node 2
                Theta = np.arctan2(vecLR[1], vecLR[0]) - np.pi / 2.0 
                if Theta < 0:
                    Theta = Theta + 2 * np.pi
                elif Theta > 2*np.pi:
                    Theta = Theta - 2*np.pi

                uwbMsg.theta = Theta# - angularOffset

                uwbMsg.vx = (v2d[0, idx1] + v2d[0, idx2]) / 2.0
                uwbMsg.vy = (v2d[1, idx1] + v2d[1, idx2]) / 2.0

                uwbMsg.posx = np.delete(uwbMsg.posx, [idx1, idx2], 0)
                uwbMsg.posy = np.delete(uwbMsg.posy, [idx1, idx2], 0)

                new_id_idx = changeIndexSorting(n, idx1, idx2)
                tmp_record = np.array(ac_out.id_record).astype(int)
                idRecord = tmp_record[new_id_idx]

                r_tab = ac_out.r_tab_r
                r_tab = r_tab[0:n, 0:n]
                r_tab = reverseIDinRangeTable(r_tab, idx1, 0)
                r_tab = reverseIDinRangeTable(r_tab, idx2, 1)
                uwbMsg.R = r_tab.flatten()
                uwbMsg.mobileID = np.array([0, 1])
                uwbMsg.node_id = idRecord

        if curDir is not None:
            print("INFO: Heading, ", curDir * 180 / np.pi)
        if middelPos is not None:
            theta = uwbMsg.theta[0]
            R = np.array( [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]] )
            uwb_offset_t = np.dot(R, uwb_offset)
            uwbMsg.robotPosx = middelPos[0] + uwb_offset_t[0, 0]
            uwbMsg.robotPosy = middelPos[1] + uwb_offset_t[1, 0]
            middelPos = None
        uwbPub.publish(uwbMsg)
        rate.sleep()

    if isRecord is True: # save the position of beacons into a mat file
        
        path = homePath + str(recordIdx) + '_uwb_pos_record'
        sio.savemat(path, {'pos': storedPos, 'nodeID': id_record})

if __name__ == '__main__':
    listener()
