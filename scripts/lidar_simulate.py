import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import rospy
import time

class Tree_Node(object):
    def __init__(self, pre_pos, dist, ind, indPrev):
        self.indPre = indPrev

def LidarSimulate_pf(particles, maps, curDirect, pointCnt, posOffset):
    baseTheta = np.pi / pointCnt
    # if curDirect > np.pi / 2 and curDirect < np.pi * 1.5:
    #     theta_l = np.arange(2.5 * np.pi - curDirect, 2 * np.pi, baseTheta)
    #     theta_r = np.arange(0, 1.5 * np.pi - curDirect, baseTheta)
    #     theta_ref = np.append(theta_l, theta_r)
    # else:
    #     theta_ref = np.arange(np.pi / 2 - curDirect, 1.5 * np.pi - curDirect, baseTheta)

    if curDirect >= 0 and curDirect < np.pi/2:
        theta_r = np.arange(1.5 * np.pi + curDirect, 2 * np.pi, baseTheta)
        theta_l = np.arange(0, curDirect + np.pi/2, baseTheta)
        theta_ref = np.append(theta_r, theta_l)
    elif curDirect >= np.pi*1.5 and curDirect < np.pi*2:
        theta_r = np.arange(-np.pi/2 + curDirect, 2 * np.pi, baseTheta)
        theta_l = np.arange(0, curDirect - np.pi*1.5 - baseTheta, baseTheta)
        theta_ref = np.append(theta_r, theta_l)
    else:
        theta_ref = np.arange(curDirect - np.pi/2, np.pi/2+curDirect, baseTheta)


    pf_cnt = particles.shape[0]
    [obsy, obsx] = np.where(maps == 1)
    obsy = obsy + posOffset[1] - 1
    obsx = obsx + posOffset[0] - 1

    obsMt = np.vstack((obsx, obsy)).T
    obs_cnt = len(obsx)

    angle_v = np.zeros((pf_cnt, obs_cnt))
    obsx = np.tile(obsx, (pf_cnt, 1))
    obsy = np.tile(obsy, (pf_cnt, 1))
    p_x = np.tile(particles[:, 0], (obs_cnt, 1)).T
    p_y = np.tile(particles[:, 1], (obs_cnt, 1)).T
    diff_x = obsx - p_x
    diff_y = obsy - p_y
    filter_v = diff_x == 0
    filter_x = diff_x > 0
    filter_y = diff_y > 0
    angle_v[np.logical_and(filter_v, filter_y)] = np.pi/2
    angle_v[np.logical_and(filter_v, np.logical_not(filter_y))] = 0
    n_filter_v = np.logical_not(filter_v)
    angle_v[n_filter_v] = np.arctan(diff_y[n_filter_v] / diff_x[n_filter_v])
    sec_quard = np.logical_not(filter_x)
    fou_quard = np.logical_and(np.logical_not(filter_y), filter_x)
    angle_v[sec_quard] = angle_v[sec_quard] + np.pi
    angle_v[fou_quard] = angle_v[fou_quard] + np.pi*2

    norm_loc = np.sum(particles ** 2, axis=1)
    norm_loc = np.tile(norm_loc, (obs_cnt, 1))
    norm_obs = np.sum(obsMt ** 2, axis=1)
    norm_obs = np.tile(norm_obs, (pf_cnt, 1))
    vDist = np.sqrt(norm_loc.T + norm_obs - 2 * np.dot(particles, obsMt.T))

    lidarData = np.zeros((pf_cnt, pointCnt))
    thetaRange = np.tile(theta_ref, (obs_cnt, 1))
    baseTheta = baseTheta / 2
    idx = 0
    for dist, angle in zip(vDist, angle_v):
        lidar_angle = np.tile(angle, (pointCnt, 1)).T
        lidar_dist = np.tile(dist, (pointCnt, 1)).T
        theta_filter = np.abs(lidar_angle - thetaRange) >= baseTheta
        lidar_dist[theta_filter] = np.inf
        lidarData[idx, :] = np.min(lidar_dist, axis=0)
        idx += 1

    return lidarData, theta_ref

def FollowController(b,zeta,ReferencePos,CurrentPos,ReferenceV,ReferenceW):

    R_matrix = np.array([[np.cos(CurrentPos[2]),np.sin(CurrentPos[2]),0],\
                [-np.sin(CurrentPos[2]), np.cos(CurrentPos[2]), 0],\
                         [0,0,1]])
    if ReferencePos[2] > np.pi:
        ReferencePos[2] = ReferencePos[2] - 2 * np.pi
    elif ReferencePos[2] < -np.pi:
        ReferencePos[2] = (2 * np.pi + ReferencePos[2])

    if CurrentPos[2] > np.pi:
        CurrentPos[2] = CurrentPos[2] - 2 * np.pi
    elif CurrentPos[2] < -np.pi:
        CurrentPos[2] = (2 * np.pi + CurrentPos[2])

    T_vector = ReferencePos - CurrentPos
    if T_vector[2] > np.pi:
        T_vector[2] = T_vector[2] - 2 * np.pi
    elif T_vector[2] < -np.pi:
        T_vector[2] = 2 * np.pi + T_vector[2]
    PoseError = np.dot(R_matrix, T_vector)
    max_v = 0.5
    max_w = 0.6
    K = np.zeros((3,1))
    K[0] = 2 * zeta * np.sqrt(ReferenceW**2 + b*ReferenceV**2)
    K[2] = K[0]
    K[1] = b * np.abs(ReferenceV)
    V = np.append(ReferenceV*np.cos(PoseError[2]), ReferenceW)
    Matrix_K = np.array([[K[0],0,0],[0,K[1]*np.sign(ReferenceV),K[2]]])
    Velocity = V + np.dot(Matrix_K, PoseError)
    v_f = Velocity[0]
    w_f = Velocity[1]
    if v_f > max_v:
        v_f = max_v
    if np.abs(w_f) > max_w:
        w_f = max_w * np.sign(w_f)
    return v_f, w_f

def construtMap_test(m, n, scale, wall):
    maps = np.zeros((m, n))
    walls = np.array([[0, 270, 84, wall], [0, 200, 80, wall], \
                      [0, 140, 82, wall], [0, 80, 84, wall], [120, 60, 40, wall], \
                      [160, 100, 10, wall], [200, 100, 10, wall], [120, 140, 90, wall], \
                      [120, 220, 90, wall], [210, 220, 30, wall], [260, 220, 40, wall], \
                      [210, 140, 90, wall], [210, 60, 90, wall], [80, 0, wall, 40], \
                      [80, 60, wall, 40], [80, 120, wall, 40], [80, 180, wall, 40], \
                      [80, 240, wall, 30], [120, 60, wall, 100], [120, 180, wall, 60], \
                      [120, 260, wall, 40], [160, 60, wall, 42], [210, 0, wall, 20], \
                      [210, 40, wall, 30], [210, 90, wall, 150], [210, 260, wall, 40]])
    walls = walls * scale
    maps[0:wall * scale, :] = 1
    maps[m - wall * scale:m, :] = 1
    maps[:, 0:wall * scale] = 1
    maps[:, n - wall * scale:n] = 1

    for rect in walls:
        maps[rect[0]:rect[0] + rect[2], rect[1]:rect[1] + rect[3]] = 1
    return maps

class RRT_PathPlanInit(object):

    def __init__(self, maps, st_pos, extend, torDist, scale, debug):
        self.maps = maps
        self.st_pos = st_pos
        # self.goal = goal
        self.extend = extend
        self.torDist = torDist
        self.mapSz = maps.shape
        self.debug = debug
        self.scale = scale
        # self.finishThres = finishThress
        self.pMax = self.maps.shape
        self.pMin = np.array([3, 3])
        self.maxDist = 300 / self.scale
        self.finishThres = 100 / self.scale
        self.nearNeibor = 3

        self.points = self.st_pos
        self.indPrev = [0]
        self.nearCnt = 0

        if self.debug:
            plt.cla()
            plt.ion()
            plt.figure(3)
            plt.imshow(self.maps)
            plt.plot(self.st_pos[0, 0], self.st_pos[0, 1], 'ob')
            plt.pause(0.1)

    def setNewGoal(self, st_pos, goal):
        self.st_pos = st_pos
        self.goal = goal
    
    def setGoal(self, goal_pos):
        self.goal = goal_pos
        if self.debug:
            plt.plot(self.goal[0, 0], self.goal[0, 1], 'or')
            plt.pause(0.1)
        
    def checkAvaiablePath(self):
        dist = np.sqrt( (self.goal[0, 0] - self.points[:, 0]) ** 2 + (self.goal[0, 1] - self.points[:, 1]) ** 2)
        minDist = np.min(dist)
        # print "minDist: ", minDist, self.finishThres
        if minDist <= self.finishThres:
            idx = np.where(dist == minDist)
            print idx, idx[0]
            return False, idx
        else:
            return True, -1

    def RRT_PathPlan(self, pathPlanTimeout):
        
        pRand = np.zeros((2, 1))
        start_time = time.time()
        elapsedTime = 0.0
        tmpThres = self.finishThres
        pMaxx = self.pMax[0] - self.pMin[0]
        pMaxy = self.pMax[1] - self.pMin[1]
        iterCnt = 0

        [flag, idx] = self.checkAvaiablePath()

        while flag and not rospy.is_shutdown():
            pRand[1] = int(pMaxx * np.random.rand(1))
            pRand[0] = int(pMaxy * np.random.rand(1))
            dist = self.MatrixDist(pRand, self.points)
            distDecend = np.sort(dist, kind='mergesort')
            distIdx = np.argsort(dist, kind='mergesort')

            if self.nearCnt < self.nearNeibor:
                nearest = self.nearCnt+1
            else:
                nearest = self.nearNeibor+1
            if self.debug:
                plt.plot(pRand[0], pRand[1], '*y')
            for near in np.arange(0,nearest):
                idx = distIdx[near]
                q_new = np.floor(self.Steer(pRand, self.points[idx], distDecend[idx], self.maxDist))
                if self.CollisionDetection(q_new, self.points[idx], self.extend, self.torDist):
                    self.points = np.vstack((self.points, q_new))

                    if self.debug:
                        plt.plot(q_new[0], q_new[1], '+b')

                    self.nearCnt += 1
                    self.indPrev.append(idx)
                    if self.MatrixDist(q_new, self.goal) <= tmpThres:
                        # if self.CollisionDetection(q_new, self.goal[0,:], self.extend, self.torDist):
                        flag = False

                    if self.debug:
                        plt.plot([self.points[idx,0],q_new[0]], [self.points[idx,1],q_new[1]], 'r')
                        plt.show()
                        plt.pause(0.05)

                    break

            elapsedTime = time.time() - start_time
            # print 'Path plan INFO: elapsedTime ', elapsedTime, ' timeout ', pathPlanTimeout
            if elapsedTime > pathPlanTimeout:
                iterCnt += 1
                if iterCnt > 3:
                    break
                tmpThres = tmpThres * 1.3
                print('Path plan INFO: timeout, new threshold: ', tmpThres)
                start_time = time.time()
                # break
        if iterCnt > 3:
            return None

        path = []
        path.append(self.goal[0,:])
        if idx == -1:
            point_len = len(self.points) - 1
        else:
            point_len = idx
        path.append(self.points[point_len])
        pathIdx = self.indPrev[point_len]

        j = 0
        while True:
            path.append(self.points[pathIdx])
            pathIdx = self.indPrev[pathIdx]
            if pathIdx == 0:
                break
            j = j + 1
        path.append(self.st_pos[0,:])
        path = np.asarray(path)
        path = path[::-1, :]

        if self.debug:
            plt.plot(path[:, 0], path[:, 1], 'b')
            plt.show()
            plt.pause(5)
        return path

    def PathSmoothing(self, path, minDist, flag):
        optPath = path
        alpha = 0.5
        beta = 0.2
        torelance = 0.00001
        change = torelance
        while change >= torelance:
            change = 0
            for ip in np.arange(1, path.shape[0]-1):
                prePath = optPath[ip, :]
                newPoint = optPath[ip, :] - alpha * (optPath[ip, :] - path[ip, :])
                newPoint = newPoint - beta * (2 * newPoint - \
                                                          optPath[ip - 1, :] - optPath[ip + 1, :])

                if flag:
                    newBound = np.array([newPoint[0] - minDist, newPoint[1] - minDist, newPoint[0] + minDist, newPoint[1] + minDist]).astype(int)
                    newBound[newBound < 0] = 0
                    if newBound[3] > self.mapSz[1]:
                        newBound[3] = self.mapSz[1]
                    if newBound[2] > self.mapSz[0]:
                        newBound[2] = self.mapSz[0]
                    pointRect = self.maps[newBound[1]:newBound[3], newBound[0]:newBound[2]]

                    if np.sum(pointRect[:]) <= 1:
                        optPath[ip, :] = newPoint
                        change = change + np.linalg.norm(optPath[ip, :] - prePath)
                else:
                    optPath[ip, :] = newPoint
                    change = change + np.linalg.norm(optPath[ip, :] - prePath)
        return optPath

    def PathSampling(self, path, speed, flag, scale, interval):
        pointCnt = path.shape[0]
        newpath = []
        if flag == 2:
            for i in np.arange(1, pointCnt):
                cr = path[i, :]
                pr = path[i - 1, :]
                disT = np.sqrt((cr[0] - pr[0]) ** 2 + (cr[1] - pr[1]) ** 2)
                interCnt = np.round(disT / scale)
                if interCnt <= 1:
                    interCnt = 2
                x = np.linspace(pr[0], cr[0], interCnt)
                y = np.linspace(pr[1], cr[1], interCnt)
                intPath = np.array([x, y]).T
                if len(newpath) == 0:
                    newpath = intPath
                else:
                    newpath = np.vstack((newpath, intPath[1:intPath.shape[0],:]))
        elif flag == 1:
            minDist = 80 / self.scale
            optPath = self.PathSampling(path, 0, 2, 150 / self.scale, 0)
            pathShow = self.PathSmoothing(optPath, minDist, True)
            optPath = self.PathSampling(pathShow, 0, 2, 60 / self.scale, 0)
            pathShow = self.PathSmoothing(optPath, minDist, True)
            optPath = self.PathSampling(pathShow, 0, 2, 8 / self.scale, 0)
            pathShow = self.PathSmoothing(optPath, minDist, False)

            path_len = pathShow.shape[0]
            newpath = np.zeros((path_len, 5))

            prePoint = pathShow[:path_len - 1, :]
            curPoint = pathShow[1:path_len, :]
            pointDist = np.sqrt((curPoint[:, 0] - prePoint[:, 0]) ** 2 + (curPoint[:, 1] - prePoint[:, 1]) ** 2)
            filter = (pointDist != 0)
            pointDist = pointDist[filter]
            filter = np.append(filter, True)
            newpath[:, 0:2] = pathShow[filter, :]

            pointCnt = newpath.shape[0]
            prePoint = newpath[:pointCnt - 1, :]
            curPoint = newpath[1:pointCnt, :]
            angles = np.arccos((curPoint[:, 0] - prePoint[:, 0]) / pointDist)
            filter = curPoint[:, 1] < prePoint[:, 1]
            angles[filter] = - angles[filter]

            newpath[0:pointCnt - 1, 2] = angles
            newpath[pointCnt - 1, 2] = angles[pointCnt-2]
            newpath[:, 3] = speed

            angleChange = newpath[1:, 2] - newpath[0:pointCnt - 1, 2]
            newpath[pointCnt - 1, 4] = 0
            filter = np.abs(angleChange) > np.pi
            angleChange[filter] = angleChange[filter] - np.sign(angleChange[filter]) * np.pi * 2
            newpath[0:pointCnt - 1, 4] = angleChange / interval

            # filter = newpath[:, 4] > np.pi / 2
            # newpath[filter, 4] = np.pi /2

            filter = np.abs(angleChange) >= 20 * np.pi / 180
            bufferCnt = 10
            for id in np.arange(0, pointCnt-1):
                if filter[id]:
                    cenVel = speed * (2 * np.pi - np.abs(angleChange[id])) / (2 * np.pi)
                    changeVelL = np.linspace(speed, cenVel, bufferCnt)
                    changeVelR = np.linspace(cenVel, speed, bufferCnt)
                    if id >= bufferCnt:
                        newpath[id - bufferCnt:id, 3] = (newpath[id - bufferCnt: id, 3] + changeVelL) / 2
                    else:
                        newpath[:id, 3] = (newpath[:id, 3] + changeVelL[:id]) / 2

                    if id + bufferCnt <= len(angleChange):
                        newpath[id:id + bufferCnt, 3] = (newpath[id:id + bufferCnt, 3] + changeVelR) / 2
                    else:
                        newpath[id:, 3] = (newpath[id:, 3] + changeVelR[:newpath.shape[0] - id]) / 2
        return newpath

    def AdjustPath(self, path, Range, pointCnt):
        newpath = path
        isFinish = False
        lidarDirection = np.arange(0, 2*np.pi, 2*np.pi/pointCnt) + np.pi/2
        sinTheta = np.sin(lidarDirection)
        cosTheta = np.cos(lidarDirection)
        pathLen = path.shape[0]
        iterFlag = np.zeros((pathLen, 1))
        while not isFinish:
            prePath = newpath
            for j in np.arange(1, pathLen-1):
                if iterFlag[j]:
                    continue
                location = newpath[j, :]
                [lLidar, rLidar] = self.LidarSimulation(location, self.maps, 0, Range, pointCnt)
                lidar = np.vstack(lLidar, rLidar)
                lidar[lidar > 20] = Range * 2
                lidar[lidar == np.inf] = Range * 2

                lidar = signal.medfilt(lidar, 5)
                location = np.tile(location, (len(lidar), 1))
                newPoint = location + np.array([lidar * sinTheta, lidar * cosTheta])
                W = lidar / np.sum(lidar)
                tmp = np.dot(newPoint, W)
                newpath[j, :] = tmp

            change = np.sum(np.abs(prePath - newpath), axis=1)
            iterFlag[change < 10] = True
            if np.sum(iterFlag[:]) > pathLen * 0.6:
                isFinish = True
        return newpath

    def PathOptimal(self, path, extend, minDist):
        pointCnt = path.shape[0]
        flag = True
        while flag:
            prePath = path
            preCnt = path.shape[0]
            for interval in np.arange(1, 4):
                newpath = path[0, :]
                iter = 1 + interval
                if iter >= pointCnt:
                    newpath = np.vstack((newpath, path[1:path.shape[0]-1, :]))
                while iter < pointCnt:
                    cr = path[iter, :]
                    pr = path[iter - interval - 1, :]

                    notCollision = self.CollisionDetection(cr, pr, extend, minDist)
                    if notCollision == True:
                        preiter = iter
                        iter = iter + interval + 1
                        if iter >= pointCnt:
                            intPath = path[preiter:path.shape[0]-1, :]
                        else:
                            intPath = path[preiter, :]
                    else:
                        preiter = iter
                        iter = iter + 1
                        if iter >= pointCnt:
                            intPath = path[preiter-interval:path.shape[0]-1, :]
                        else:
                            intPath = path[preiter-interval, :]
                    newpath = np.vstack((newpath, intPath))
                newpath = np.vstack((newpath, path[path.shape[0]-1, :]))
                path = newpath
                pointCnt = path.shape[0]
            pointCnt = path.shape[0]
            if pointCnt == preCnt:
                diff = prePath - newpath
                if np.sum(diff[:]) == 0:
                    flag = False
        return newpath

    def LidarSimulate(self, location, curAngle, range, pointCnt):
        hlfPoint = pointCnt / 2
        location = location / self.scale
        p_y = location[1]
        p_x = location[0]
        mapSz = self.maps.shape
        range = range / self.scale
        sub_rect = [p_x - range, p_y - range, p_x + range, p_y + range]
        sub_rect[sub_rect < 0] = 0
        if sub_rect[2] > mapSz[0]:
            sub_rect[2] = mapSz[0]
        if sub_rect[3] > mapSz[1]:
            sub_rect[3] = mapSz[1]
        sub_patch = self.maps[sub_rect[0]:sub_rect[2]+1, sub_rect[1]:sub_rect[3]+1]
        [obsx, obsy] = np.where(sub_patch == 1)
        obsy = obsy + sub_rect(2) - 1
        obsx = obsx + sub_rect(1) - 1
        obsMt = np.array([[obsx], [obsy]])

        filter = obsx >= p_x
        left = True
        right = True
        if np.sum(filter) == 0:
            left = False
        elif np.sum(filter) == len(obsx):
            right = False
        hlfPoint = pointCnt / 2
        baseTheta = 2 * np.pi / pointCnt
        lLidar = np.zeros((hlfPoint, 1))
        rLidar = np.zeros((hlfPoint, 1))
        if left:
            lObsx = obsx[filter]
            lObsy = obsy[filter]
            lDist = self.MatrixDist(location, obsMt)
            lTheta = np.arcsin((lObsy - p_y) / lDist) + np.pi / 2
            for i in np.arange(0, hlfPoint-1):
                idx = np.abs(lTheta - baseTheta * i) < baseTheta / 2
                if np.sum(idx) >= 1:
                    lLidar[i + 1] = np.min(lDist[idx])
                else:
                    lLidar[i + 1] = np.inf
        if right:
            rObsx = obsx[not filter]
            rObsy = obsy[not filter]
            rDist = self.MatrixDist(location, obsMt)
            rTheta = np.arcsin((rObsy - p_y) / rDist) + np.pi / 2
            for i in np.arange(0, hlfPoint-1):
                idx = np.abs(rTheta - baseTheta * i) < baseTheta / 2
                if np.sum(idx) >= 1:
                    rLidar[i + 1] = np.min(rDist[idx])
                else:
                    rLidar[i + 1] = np.inf
            lidarData = np.vstak(lLidar, rLidar[::-1])
            lidarData[lidarData == 0] = np.inf
            oriPos = np.round((curAngle + 90) * np.pi / 180 / baseTheta)
            if oriPos > pointCnt:
                oriPos = np.mod(oriPos, pointCnt)
            if oriPos > hlfPoint:
                lLidar = lidarData[oriPos:-1:oriPos - hlfPoint + 2]
                rLidar = np.vstack(lidarData[oriPos+1:], lidarData[0:oriPos - hlfPoint+1])
            else:
                rLidar = lidarData[oriPos:oriPos + hlfPoint]
                lLidar = np.vstack(lidarData[oriPos-1:-1:-1], lidarData[len(lidarData)-1:-1:oriPos+hlfPoint+1])
            lLidar = lLidar * self.scale
            rLidar = rLidar * self.scale

    def LidarSimulate_pf(self, curDirect, pointCnt):
        baseTheta = np.pi / pointCnt
        if curDirect >= 0 and curDirect < np.pi/2:
            theta_r = np.arange(1.5 * np.pi + curDirect, 2 * np.pi, baseTheta)
            theta_l = np.arange(0, curDirect + np.pi/2, baseTheta)
            theta_ref = np.append(theta_r, theta_l)
        elif curDirect >= np.pi*1.5 and curDirect < np.pi*2:
            theta_r = np.arange(-np.pi/2 + curDirect, 2 * np.pi, baseTheta)
            theta_l = np.arange(0, curDirect + np.pi/2, baseTheta)
            theta_ref = np.append(theta_r, theta_l)
        else:
            theta_ref = np.arange(curDirect - np.pi/2, np.pi/2+curDirect, baseTheta)
        pf_cnt = self.particles.shape[0]

        [obsx, obsy] = np.where(self.maps == 1)
        obsMt = np.array([[obsx], [obsy]])
        obs_cnt = len(obsx)

        angle_v = np.zeros(pf_cnt, obs_cnt)
        obsx = np.tile(obsx, (pf_cnt, 1))
        obsy = np.tile(obsy, (pf_cnt, 1))
        p_x = np.tile(self.particles[:, 1], (obs_cnt, 1))
        p_y = np.tile(self.particles[:, 2], (obs_cnt, 1))
        diff_x = obsx - p_x
        diff_y = obsy - p_y
        filter_v = np.find(diff_x == 0)
        filter_y = np.find(diff_x > 0)
        filter_x = np.find(diff_y > 0)
        angle_v[filter_v and filter_y] = np.pi
        angle_v[filter_v and (not filter_y)] = 0
        left_logical = (not filter_v) and filter_x
        right_logical = (not filter_v) and (not filter_x)
        angle_v[left_logical] = np.arctan(diff_y[left_logical] / diff_x[left_logical]) + np.pi / 2
        angle_v[right_logical] = np.arctan(diff_y[right_logical] / diff_x[right_logical]) + np.pi * 1.5

        norm_loc = np.sum(self.particles ** 2, axis=1)
        norm_loc = np.tile(norm_loc, (obs_cnt, 1))
        norm_obs = np.sum(obsMt ** 2, axis=1)
        norm_obs = np.tile(norm_obs, (pf_cnt, 1))
        vDist = np.sqrt(norm_loc + norm_obs.T - 2 * np.dot(self.particles, obsMt.T))

        lidarData = np.zeros(pf_cnt, pointCnt)
        thetaRange = np.tile(theta_ref, (obs_cnt, 1))
        baseTheta = baseTheta / 2
        idx = 0
        for dist, angle in zip(vDist, angle_v):
            lidar_angle = np.tile(angle, (pointCnt, 1))
            lidar_dist = np.tile(dist, (pointCnt, 1))
            theta_filter = np.where(np.abs(lidar_angle - thetaRange) >= baseTheta)
            lidar_dist[theta_filter] = 10000
            lidarData[idx, :] = np.min(lidar_dist, axis=0)

        return lidarData

    def MatrixDist(self, pos1, pos2):
        d = np.sqrt((pos1[0]-pos2[:,0])**2+(pos1[1]-pos2[:,1])**2)
        return d

    def Steer(self, pRand, curPoints, val, maxDist):
        # d = np.sqrt((pRand[0]-curPoints[0])**2 + (pRand[1]-curPoints[1])**2)
        # print 'd:', d, 'val:', val
        if val >= maxDist:
            qnew = curPoints + ((pRand.T - curPoints)*maxDist)/val
        else:
            qnew = pRand.T
        return qnew[0,:]

    def CollisionDetection(self, p1, p2, extend, minDist):
        start = np.floor(p2)
        goal = np.floor(p1)

        theta = np.arctan2( goal[1]-start[1], goal[0]-start[0] )
        offX = extend * np.sin(theta)
        offY = extend * np.cos(theta)
        polyx = np.array([start[0]-offX, goal[0]-offX, goal[0]+offX, start[0]+offX, start[0]-offX]).astype(int)
        polyy = np.array([start[1]+offY, goal[1]+offY, goal[1]-offY, start[1]-offY, start[1]+offY]).astype(int)

        lu = np.array([np.min(polyx), np.min(polyy)])
        rb = np.array([np.max(polyx), np.max(polyy)])
        lu[lu<0] = 0
        if rb[0] > self.mapSz[1]:
            rb[0] = self.mapSz[1]
        if rb[1] > self.mapSz[0]:
            rb[1] = self.mapSz[0]
        outerMap = self.maps[lu[1]:rb[1], lu[0]:rb[0]]
        [pointY, pointX] = np.where(outerMap == 1)

        pointX = pointX + lu[0]
        pointY = pointY + lu[1]
        hasObsInter = False
        hasObsPoint = False
        v_b = goal[:2] - start[:2]
        n_v_b = (v_b[0] ** 2 + v_b[1] ** 2)
        if len(pointX) >= 1:
            v_a_x = pointX - start[0]
            v_a_y = pointY - start[1]
            a_dot_b = ( v_a_x * v_b[0] + v_a_y * v_b[1] ) / n_v_b
            v_e_x = v_a_x - a_dot_b * v_b[0]
            v_e_y = v_a_y - a_dot_b * v_b[1]
            distMat = np.sqrt(v_e_x ** 2 + v_e_y ** 2)
            filter = distMat < extend
            if np.sum(filter) > 0:
                hasObsInter = True

        # newBound = np.array([goal[0]-minDist, goal[1]-minDist, goal[0]+minDist, goal[1]+minDist]).astype(int)
        # newBound[newBound < 0] = 0
        # if newBound[3] > self.mapSz[1]:
        #     newBound[3] = self.mapSz[1]
        # if newBound[2] > self.mapSz[0]:
        #     newBound[2] = self.mapSz[0]
        # pointRect = self.maps[newBound[1]:newBound[3], newBound[0]:newBound[2]]
        # if np.sum(pointRect[:]) > 1:
        #     hasObsPoint = True
        hasObsPoint = False
        if hasObsInter or hasObsPoint:
            iscollision = False
        else:
            iscollision = True
        return iscollision

    
