import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import rospy
import time



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
        # print("minDist: ", minDist, " finishThres: ", self.finishThres)
        if minDist <= self.finishThres:
            idx = np.where(dist == minDist)
            # print("idx: ", idx, " idx[0]: ", idx[0][0])
            return False, idx[0][0]
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
            if elapsedTime > pathPlanTimeout:
                iterCnt += 1
                if iterCnt > 3:
                    break
                tmpThres = tmpThres * 1.3
                print('Path plan INFO: timeout, new threshold: ', tmpThres)
                start_time = time.time()
                
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

        hasObsPoint = False
        if hasObsInter or hasObsPoint:
            iscollision = False
        else:
            iscollision = True
        return iscollision

    
