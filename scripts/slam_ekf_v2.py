import struct
import time
import itertools
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import copy
from scipy import stats
import numpy.matlib
from scipy.linalg import block_diag

class uwbRange(object):

    def __init__(self, ser, ap, constNode, fixConstrain, ext, nodesIDs):
        self.ser = ser
        self.ap = ap
        self.cn = constNode
        self.fixConstrain = fixConstrain * 100
        self.ids = nodesIDs

        self.mea_var = 0.01
        self.nlos_thresh = 6
        self.var_zero_r = 1e30

        self.dt_p2p = 0.005
        self.dt_srch = 0.05
        self.ext = ext
        self.cmd_range_bat = [0xB5, 0x00, 0x01]
        self.cmd_getrange = [0xC7, 0x00]
        self.cmd_search = [0xB7] + [0x01] + ext
        self.cmd_range_sin = [0xB5, 0x00, 0x00] + ext
        self.cmd_getsearch = [0xB9, 0x00, 0x01] + ext
        self.cmd_remote = [0xC6, 0x00]
        self.cmd_send_range = [0xC5, 0x00]
        self.cmd_clr_rem_buff = [0xC5, 0x00, 0] + ext
        self.cmd_clr_loc_buff = [0xC7, 0x00, 0] + ext

    def setAp(self, ap, cn):
        self.ap = ap
        self.cn = cn

    def do_ranging(self, id_now):
        id_all = self.ap + id_now
        n = len(id_all)
        srn = list(itertools.combinations(list(range(n)), 2))
        self.ser.write(bytearray(self.cmd_range_bat+id_now+self.ext))
        time.sleep(len(srn)*self.dt_p2p)

    def do_search(self):
        # self.ser.write(bytearray(self.cmd_search))
        # time.sleep(self.dt_srch)
        # self.ser.write(bytearray(self.cmd_getsearch))
        # ids = np.array(struct.unpack('b'*20, self.ser.read(20)))
        # ids = list(ids[np.nonzero(ids)[0]])
        # ids = [17, 16, 24, 23, 22, 19, 20]

        # ids = [17, 16, 24, 23, 22, 19]
        return self.ids

    def test_ranging(self, id_new):

        self.do_ranging(id_new)
        time.sleep(self.dt_p2p)
        self.ser.write(bytearray(self.cmd_getrange+[1]+self.ext))
        s = self.ser.read(8)
        data_list = list(struct.unpack('bbhbbh',s))
        passTest = False
        try:
            if data_list[3] != 0 and np.abs(data_list[2] - self.fixConstrain) < 50:
                passTest = True
        except:
            return passTest
        return passTest

    def get_range_table(self, flag):
        id_new = self.do_search()
        id_all = self.ap + id_new
        n = len(id_all)
        srn = list(itertools.combinations(range(n), 2))
        num_ranges = len(srn)
        self.do_ranging(id_new)
        # time.sleep(len(srn)*self.dt_p2p)
        r_tab = np.zeros((n, n))  # ranging table
        r_cov = np.ones((n, n)) * self.var_zero_r
        self.ser.write(bytearray(self.cmd_getrange+[num_ranges]+self.ext))
        for i in range(num_ranges):
            s = self.ser.read(8)
            try:
                data_list = list(struct.unpack('bbhbbh', s))
                if data_list[3] != 0:
                    ix = id_all.index(data_list[0])
                    iy = id_all.index(data_list[1])
                    idx = srn.index((min(ix, iy), max(ix, iy)))
                    r_tab[srn[idx][0], srn[idx][1]] = data_list[2]
                    r_tab[srn[idx][1], srn[idx][0]] = r_tab[srn[idx][0], srn[idx][1]]
                    if data_list[3]-data_list[4] > self.nlos_thresh:
                        r_cov[srn[idx][0], srn[idx][1]] = self.var_nlos_r
                        r_cov[srn[idx][1], srn[idx][0]] = self.var_nlos_r
                    else:
                        r_cov[srn[idx][0], srn[idx][1]] = self.mea_var
                        r_cov[srn[idx][1], srn[idx][0]] = self.mea_var
            except:
                pass
        if flag == True:
            [idx1, idx2, missingNodes] = self.find_axisIdx(id_all, np.array([self.ap, self.cn]))
            if missingNodes == False:
                r_tab[idx1, idx2] = self.fixConstrain
                r_tab[idx2, idx1] = self.fixConstrain
                r_cov[idx1, idx2] = 1e-5
                r_cov[idx2, idx1] = 1e-5
        r_tab = r_tab/100.
        return r_tab, r_cov, id_all, n

    def find_axisIdx(self, id_record, axIdx):
        try:
            idx1 = id_record.index(axIdx[0])
            idx2 = id_record.index(axIdx[1])
        except:
            return [], [], True
        return idx1, idx2, False

class slam_ekf(object):
    """Self-calibration

    Attributes:
        ap: central node id
        ac: anchor id
        alpha: significant value used in hypothesis testing for outlier detection
        axis_limit: defines the limit of x-axis and y-axis in a plot
        cluster: e.g. 3 clusters with their cluster members: ([1,2,3],[4,5,6],[7,8,9])
        dt_p2p: waiting time for one p2p ranging
        dt_srch: waiting time for nodes search
        dt_clr: waiting time for buffer clear or remote command
        dt_mot: difference between two consecutive time steps, used for motion prediction
        mot_var: variance used in motion model
        mea_var: variance of p2p ranging
        nlos_thresh: above which the link is treated as NLOS
        outlier_detect: run outlier detection if =1;
        srch_lim: limit of how many new nodes (except the central node) to be found initially
        update_var: when new node(s) are detected, their position variance is set to update_var
        x_sign: =1 or =-1 means the 2nd node is on positive or negative x-axis;
        y_sign: =1 or =-1 means the 3nd node is on positive or negative y-axis;
    """

    def __init__(self, ap, constNode, axIdx, fixNode, nodesIDs):
        """Return a new self_calib object."""

        self.ap = ap
        self.axIdx = axIdx
        self.cn = constNode
        self.fixNode = fixNode
        self.ids = nodesIDs

        self.z = np.zeros((1, 1))
        self.nlos_thresh = 6
        self.update_var = .1
        self.alpha = 0.01

        # self.dt_mot = 0.008
        # self.mot_var = 0.008
        # self.mot_var1 = 0.1

        self.dt_mot = 0.03
        self.mot_var = 0.05
        self.mot_var1 = 0.1
        self.mea_var = 0.01
        self.var_nlos_r = 1e1
        self.var_zero_r = 1e30
        self.ref_node = [1, 2]
        self.axis_limit = [-20, 100, -30, 100]

        self.x_sign = 1
        self.y_sign = 1
        self.err_std_cur = []
        self.iter = 0
        self.i_los = []

        self.m_k = np.zeros((4, 1))
        self.p_k = np.zeros((4, 4))
        
        self.id_prev = []
        self.id_record = []
        self.new2record = []
        self.id_record_prev = []
        self.m_k_r = np.zeros((4, 20))
        self.p_k_r = np.zeros((4, 4*20))
        self.err_std_cur = []
        self.err_std_mtx = np.ones((20, 20)) * 20
        self.r_tab_r = np.zeros((20, 20))
        self.r_cov_r = np.ones((20, 20))*1e30
        self.ix = []
        self.ix_prev = []
        self.n = 0
        self.drop_ids = []
        self.srch_lim = 2
        self.search = False

        sizeOfrecord = 30
        self.sizeOfrecord = sizeOfrecord
        self.px_result = np.zeros((20, sizeOfrecord))
        self.py_result = np.zeros((20, sizeOfrecord))

        self.std_px_result = [100]*sizeOfrecord
        self.std_py_result = [100]*sizeOfrecord

        self.max_std = 0.06
        self.result = None
        self.closeFlag = False

    def setFeedback(self, stateFeedback, halfFixConstrain):

        nodeCnt = int(stateFeedback[-1])
        theta = stateFeedback[-2]
        ssin = np.sin(theta)
        ccos = np.cos(theta)
        print("INFO: stateFeedback, ", stateFeedback)

        # # print("INFO: m_k before, ", self.m_k.T)
        # # # left
        # # print("INFO: m_k before, ", self.m_k[np.array([0, 3, nodeCnt, nodeCnt+3]), 0])
        # self.m_k[nodeCnt*2+3, 0] = stateFeedback[3] * ccos # vx
        # self.m_k[nodeCnt*3+3, 0] = stateFeedback[3] * ssin # vy
        
        # # right
        # self.m_k[nodeCnt*2, 0] = stateFeedback[2] * ccos # vx
        # self.m_k[nodeCnt*3, 0] = stateFeedback[2] * ssin # vy

        # # if np.abs(stateFeedback[0]) > 0 and np.abs(stateFeedback[1]) > 0:
        # # the arangement of m_k is x0,x1,...,xn; y0,y1,...,yn; vx0,vx1,...,vxn; vy0, vy1,...,vyn

        # # left # index = 3 * 4 + 0:1
        # self.m_k[3, 0] = stateFeedback[0] - halfFixConstrain * ssin 
        # self.m_k[nodeCnt+3, 0] = stateFeedback[1] + halfFixConstrain * ccos
        # # right node
        # self.m_k[0, 0] = stateFeedback[0] + halfFixConstrain * ssin # x
        # self.m_k[nodeCnt, 0] = stateFeedback[1] - halfFixConstrain * ccos # y

        # # left
        # self.m_k[nodeCnt*2+3, 0] = stateFeedback[2] * ccos # vx
        # self.m_k[nodeCnt*3+3, 0] = stateFeedback[2] * ssin # vy
        # # right
        # self.m_k[nodeCnt*2, 0] = stateFeedback[3] * ccos # vx
        # self.m_k[nodeCnt*3, 0] = stateFeedback[3] * ssin # vy

        # left # index = 3 * 4 + 0:1
        self.m_k[0, 0] = stateFeedback[0] - halfFixConstrain * ssin 
        self.m_k[nodeCnt, 0] = stateFeedback[1] + halfFixConstrain * ccos
        # right node
        self.m_k[3, 0] = stateFeedback[0] + halfFixConstrain * ssin # x
        self.m_k[nodeCnt+3, 0] = stateFeedback[1] - halfFixConstrain * ccos # y

        # print("INFO: m_k after, ", self.m_k.T)
        # the arangement of p_k is x0,x1,...,xn; y0,y1,...,yn; vx0,vx1,...,vxn; vy0, vy1,...,vyn
        self.p_k.flags.writeable = True
        self.p_k[np.array([0, 3, nodeCnt, nodeCnt+3, 2*nodeCnt, 2*nodeCnt+3, 3*nodeCnt, 3*nodeCnt+3]), 0] = 1e-20

    def setAp(self, ap, cn):
        self.ap = ap
        self.cn = cn

    def do_search(self):
        # ids = [17, 16, 24, 23, 22, 19, 20]
        # ids = [17, 16, 24, 23, 22, 19]
        return self.ids[0:min(len(self.ids), self.srch_lim)]

    def get_range_table(self, r_tab, id_cur, r_cov):
        n = len(id_cur)
        new_r_tab = r_tab[:n, :n]
        new_r_cov = r_cov[:n, :n]
        return new_r_tab, new_r_cov

    def initial(self, r_tab, r_cov):
        id_cur = self.do_search()
        [new_r_tab, r_cov] = self.get_range_table(r_tab, self.ap + id_cur, r_cov)

        self.n = len(id_cur) + 1
        self.err_std_cur = np.ones((self.n, 1))*10
        mk = np.zeros((4*self.n, 1))
        pk = np.ones((4*self.n, 1))*1e0
        mk[self.ref_node[1]] = r_tab[self.ref_node[0], self.ref_node[1]]
        mk[self.ref_node[1]+self.n] = 0
        pk[self.ref_node[0], :] = 1e-20
        pk[self.ref_node[1], 0] = self.mea_var
        pk[self.ref_node[1], 1:self.n] = 1e-20
        self.id_record = self.ap + id_cur

        self.m_k = copy.copy(mk)
        self.p_k = copy.copy(pk)
        id_curr = self.ap + id_cur
        self.ix = []
        for i in range(len(id_curr)):
            self.ix += [self.id_record.index(id_curr[i])]
        self.r_tab_r[np.ix_(self.ix, self.ix)] = new_r_tab
        self.id_record_prev = copy.copy(self.id_record)
        self.id_prev = copy.copy(id_cur)
        self.ix_prev = copy.copy(self.ix)

    def update_ids(self):
        if self.search:
            self.id_new = self.do_search()
        else:
            self.id_new = copy.copy(self.id_prev)
        self.new2record = list(set(self.id_new)-set(self.id_record_prev))

        if len(self.new2record) > 1: # update just one node at a time
            self.new2record = [self.new2record[0]]
            self.id_new = self.id_prev + self.new2record

        if len(self.new2record) > 0:
            self.id_record = self.id_record_prev + self.new2record
        else:
            self.id_record = copy.copy(self.id_record_prev)
        self.drop_ids = []
        if len(self.id_new) < len(self.id_prev):
            self.drop_ids = list(set(self.id_prev)-set(self.id_new))
        self.n = len(self.id_new) + 1
        id_curr = self.ap + self.id_new
        self.ix = []
        for i in range(len(id_curr)):
            self.ix += [self.id_record.index(id_curr[i])]

    def update_r_table(self, r_tab, r_cov):
        self.r_tab_r[0, :] = 0
        self.r_tab_r[:, 0] = 0
        r_temp = self.r_tab_r[np.ix_(self.ix, self.ix)]
        nonzero_ix = np.where(r_tab > 0)
        r_temp[nonzero_ix] = r_tab[nonzero_ix]
        r_temp[0, :] = r_tab[0, :]
        r_temp[:, 0] = r_tab[:, 0]
        self.r_tab_r[np.ix_(self.ix, self.ix)] = r_temp

        self.r_cov_r[0, :] = 0
        self.r_cov_r[:, 0] = 0
        c_temp = self.r_cov_r[np.ix_(self.ix, self.ix)]
        c_temp[nonzero_ix] = r_cov[nonzero_ix]
        c_temp[0, :] = r_cov[0, :]
        c_temp[:, 0] = r_cov[:, 0]
        self.r_cov_r[np.ix_(self.ix, self.ix)] = c_temp

        n = len(self.id_record)
        self.err_std_cur = np.array(self.err_std_mtx[np.triu_indices(n, k=1)]).reshape(n * (n - 1) // 2, 1)[:, 0]

    def para_update_new_ids(self):
        n = len(self.id_record)
        m_k_new = self.m_k_r[:, 0:n].reshape(4*n, 1)
        p_k_new = self.p_k_r[:, 0:n].reshape(4*n, 1)
        p2D = np.zeros((2, n))
        p2D[0, :] = m_k_new[0:n].flatten()
        p2D[1, :] = m_k_new[n:2*n].flatten()
        for i in range(len(self.new2record)):
            numi = self.id_record.index(self.new2record[i])
            r_temp = self.r_tab_r[0:len(self.id_record_prev), :]
            ri = r_temp[:, numi]
            neighbs = [list(x) for x in np.where(ri > 0)]
            neighbs = neighbs[0]
            rii = ri[neighbs]
            pos_neighb = p2D[:, neighbs]
            if len(neighbs) >= 3:
                A = np.zeros((len(neighbs), 3))
                A[:, 0] = np.ones((1, len(neighbs)))
                A[:, 1] = -2*pos_neighb[0, 0:len(neighbs)]
                A[:, 2] = -2*pos_neighb[1, 0:len(neighbs)]
                b = np.array(rii[0:len(neighbs)]**2 - pos_neighb[0, 0:len(neighbs)].T**2 - pos_neighb[1, 0:len(neighbs)].T**2).reshape(len(neighbs), 1)
                xE = np.dot(inv(np.dot(A.T, A)), np.dot(A.T, b))
                m_k_new[numi] = xE[1, 0]
                m_k_new[n+numi] = xE[2, 0]
                p_k_new[numi, :] = self.update_var
                p_k_new[n+numi, :] = self.update_var
        self.m_k_r[:, 0:n] = m_k_new.reshape(4, n)
        self.p_k_r[:, 0:n] = p_k_new.reshape(4, n)
        self.m_k = self.m_k_r[:, 0:n].reshape(4*n, 1)
        self.p_k = self.p_k_r[:, 0:n].reshape(4*n, 1)

    def dynamic_F(self):
        n = len(self.id_record)
        num_unkn = 2 * n
        F = np.eye(2*num_unkn)
        F[np.ix_(range(num_unkn), range(num_unkn, 2*num_unkn))] = self.dt_mot*np.eye(num_unkn)
        return F

    def dynamic_Q(self):
        n = len(self.id_record)
        num_unkn = 2*n
        Qsub = [[self.dt_mot**2, self.dt_mot], [self.dt_mot, 1]]

        m_var1 = np.array([[self.mot_var, self.mot_var], [self.mot_var, self.mot_var]])
        m_var1 = np.kron(np.eye(2), m_var1)
        m_var2 = np.array([[self.mot_var1, self.mot_var1], [self.mot_var1, self.mot_var1]])
        m_var2 = np.kron(np.eye(2), m_var2)
        m_var_matrix = None
        for i in np.arange(n):
            tmp_id = self.id_record[i]
            if tmp_id == self.ap[0] or tmp_id == self.cn[0]:
                m_tmp = m_var2
            else:
                m_tmp = m_var1

            if m_var_matrix is None:
                m_var_matrix = m_tmp
            else:
                m_var_matrix = block_diag(m_var_matrix, m_tmp)

        Q = m_var_matrix * np.kron(np.eye(num_unkn), Qsub)
        x_idx_from = list(range(0, 2*num_unkn, 2))
        v_idx_from = list(range(1, 2*num_unkn, 2))
        idx_from = x_idx_from+v_idx_from
        x_idx_to = list(range(0, num_unkn, 1))
        v_idx_to = list(range(num_unkn, 2*num_unkn, 1))
        idx_to = x_idx_to+v_idx_to
        Q[idx_to, :] = Q[idx_from, :]     # swap row 0 with row 4...
        Q[:, idx_to] = Q[:, idx_from]     # ...and column 0 with column 4
        return Q

    def cv(self, F, Q):
        n = len(self.id_record)
        nodeShape = 2*n
        nodeShape = len(self.m_k)
        # update the state of UWBs on the robot according to the information from encoder
        xF = np.dot(F, self.m_k.reshape(nodeShape,1))
        xk_plus_1 = xF+np.dot(F,np.random.multivariate_normal([0]*nodeShape, Q, 1).T)

        # n = len(self.id_record)
        # num_unkn = 2*n
        # xF = np.dot(F, self.m_k.reshape(2*num_unkn, 1))
        # xk_plus_1 = xF+np.dot(F, np.random.multivariate_normal([0]*2*num_unkn, Q, 1).T)
        return xk_plus_1

    def jacob(self, x):
        n = len(self.id_record)
        srn_all = list(range(n))
        srn = list(itertools.combinations(srn_all, 2))
        H = np.zeros((len(srn), 2*n))
        for i in range(len(srn)):
            iax = srn[i][0]
            ibx = srn[i][1]
            iay = srn[i][0]+n
            iby = srn[i][1]+n
            d = ((x[iax]-x[ibx])**2+(x[iay]-x[iby])**2)**0.5
            H[i, iax] = (x[iax]-x[ibx])/d
            H[i, ibx] = (x[ibx]-x[iax])/d
            H[i, iay] = (x[iay]-x[iby])/d
            H[i, iby] = (x[iby]-x[iay])/d
        H_full = np.zeros((len(srn), 4*n))
        H_full[:, list(range(0, 2*n))] = H
        return H_full

    def meas(self, x):
        n = len(self.id_record)
        srn_all = list(range(n))
        srn = list(itertools.combinations(srn_all, 2))
        z = np.zeros((len(srn), 1))
        h = np.zeros((len(srn), 1))
        for i in range(len(srn)):
            ia = srn[i][0]
            ib = srn[i][1]
            z[i] = self.r_tab_r[ia, ib]
            h[i] = ((x[ia]-x[ib])**2+(x[ia+n]-x[ib+n])**2)**0.5
        return z, h

    def ekf(self):
        n = len(self.id_record)
        F_k = self.dynamic_F()
        Q_kminus1 = self.dynamic_Q()
        m_k_given_kminus1 = self.cv(F_k, Q_kminus1)
        P_k_given_kminus1 = Q_kminus1 + np.dot(np.dot(F_k, np.diag(self.p_k[:, 0])), np.transpose(F_k))
        H_hat_k = self.jacob(m_k_given_kminus1)
        z_k, h_k = self.meas(m_k_given_kminus1)
        r_cov_diag = np.diag(np.array(self.r_cov_r[np.triu_indices(n, k = 1)]).reshape(n*(n-1)//2, 1)[:, 0])
        S_k = np.dot(np.dot(H_hat_k, P_k_given_kminus1), np.transpose(H_hat_k)) + r_cov_diag
        self.outlier_detection(z_k, h_k)
        z_k = z_k[self.i_los, :]
        h_k = h_k[self.i_los, :]
        H_hat_k = H_hat_k[self.i_los, :]
        S_k = S_k[np.ix_(self.i_los, self.i_los)]
        K_k = np.dot(np.dot(P_k_given_kminus1, np.transpose(H_hat_k)), inv(S_k))
        self.m_k = m_k_given_kminus1 + np.dot(K_k, (z_k-h_k))
        self.p_k = np.diag(P_k_given_kminus1 - np.dot(np.dot(K_k, H_hat_k), P_k_given_kminus1)).reshape(4*n, 1)

    def outlier_detection(self, z_k, h_k):
        e_k = z_k - h_k
        lt = stats.norm.ppf(self.alpha / 2, loc=0, scale=1)
        rt = stats.norm.ppf(1 - self.alpha / 2, loc=0, scale=1)
        self.i_los = []
        k = 0
        for i in range(len(self.id_record)-1):
            for j in range(i+1, len(self.id_record)):
                e_k_norm = e_k[k] / self.err_std_cur[k]
                if lt <= e_k_norm[0] <= rt:
                    self.i_los += [k]
                else:
                    self.r_tab_r[i, j] = 0
                    self.r_tab_r[j, i] = 0
                    self.r_cov_r[i, j] = self.var_zero_r
                    self.r_cov_r[j, i] = self.var_zero_r
                k += 1

    def update_record(self):
        self.m_k_r[:, 0:len(self.id_record)] = self.m_k.reshape(4, len(self.id_record))
        self.p_k_r[:, 0:len(self.id_record)] = self.p_k.reshape(4, len(self.id_record))

    def get_2d_pos(self):
        p2d = self.m_k_r[0:2, :]
        v2d = self.m_k_r[2:4, :]
        return p2d, v2d

    def coord_transform(self, p2d, v2d, idx1, idx2):
        ## the coordinate transformation process coded as below can do both rotate and translate for all the nodes to the x-axis. This will result in the predefined axis nodes overlap with x-axis.
        p2 = p2d[:,idx2]
        p1 = p2d[:,idx1]
        A = np.array([ [p1[0], -p1[1]], [p1[1], p1[0]] ])
        B = np.array([ [p2[0], -p2[1]], [p2[1], p2[0]] ])
        aXis = np.array([[np.linalg.norm(p2 - p1)], [0]])
        b = np.dot(inv(A - B), -aXis)
        costh = b[0,0]
        sinth = b[1,0]
        R = np.array([[costh, -sinth],[sinth, costh]])
        T = -np.dot(A, b)

        p2d_new = np.dot(R, p2d) + np.matlib.repmat(T, 1, p2d.shape[1])
        v2d_new = np.dot(R, v2d)
        return p2d_new, v2d_new

    def find_axisIdx(self, id_record, axIdx):
        try:
            idx1 = id_record.index(axIdx[0])
            idx2 = id_record.index(axIdx[1])
        except:
            return [], [], True
        return idx1, idx2, False

    def enable_search(self):
        if max(self.std_px_result) < self.max_std and max(self.std_py_result) < self.max_std:
            self.search = True
            self.srch_lim += 1
            self.err_std_mtx[np.ix_(self.ix, self.ix)] = .5
            self.max_std += .02
        else:
            self.search = False

    def visual(self, p2d, v2d):
        plt.figure(1)
        px = p2d[0, 0]
        py = p2d[1, 0]
        preResult = np.array([px, py]).reshape(1, 2)
        if self.result is None:
            self.result = np.array([preResult]).reshape(1, 2)
        else:
            self.result = np.vstack((self.result, preResult))

        # print self.result.shape
        if self.result is not None:
            plt.plot(self.result[:, 0], self.result[:, 1], '-b')

        plt.plot(p2d[0, 1:len(self.id_record)], p2d[1, 1:len(self.id_record)], 'go', markersize=5)
        for i in range(len(self.id_record)):
            plt.text(p2d[0, i], p2d[1, i], self.id_record[i], fontsize=15)
            # v2d_norm = v2d[:,i]
            # plt.arrow(p2d[0,i], p2d[1,i], v2d_norm[0], v2d_norm[1], hold=None, fc="k", ec="k", head_width=0.05, head_length=0.1 )
        plt.axis(self.axis_limit)
        plt.xlabel('x-axis in meters')
        plt.ylabel('y-axis in meters')
        plt.pause(1e-10)
        plt.clf()

    def stat_report(self, p2d):
        px_preResult = p2d[0, 0:len(self.id_record)]
        py_preResult = p2d[1, 0:len(self.id_record)]
        self.px_result[np.ix_(list(range(len(self.id_record))), [self.iter % self.sizeOfrecord])] = px_preResult.reshape(len(self.id_record), 1)
        self.py_result[np.ix_(list(range(len(self.id_record))), [self.iter % self.sizeOfrecord])] = py_preResult.reshape(len(self.id_record), 1)

        self.std_px_result = np.zeros((len(self.id_record), 1))
        self.std_py_result = np.zeros((len(self.id_record), 1))
        for i in range(len(self.id_record)):
            px_result_curr = self.px_result[i, :]
            self.std_px_result[i] = np.std(px_result_curr)
            py_result_curr = self.py_result[i, :]
            self.std_py_result[i] = np.std(py_result_curr)

    def run(self, ranges, r_cov, visual):
        
        self.update_ids()
        [new_r_tab, new_r_cov] = self.get_range_table(ranges, self.id_record, r_cov)
        
        
        self.update_r_table(new_r_tab, new_r_cov)
        self.z = np.zeros((len(self.id_record), 1))

        if len(self.new2record) > 0:
            self.para_update_new_ids()
        self.ekf()
        self.update_record()

        p2d, v2d = self.get_2d_pos()
        [axisIdx1, axisIdx2, missingNodes] = self.find_axisIdx(self.id_record, self.axIdx)
        if missingNodes == True:
            return [], [], True
        p2d, v2d = self.coord_transform(p2d, v2d, axisIdx1, axisIdx2)

        p2d[1, 0] = abs(p2d[1, 0])
        v2d[:,0:2] = 0
        self.m_k_r = np.concatenate((p2d, v2d), axis=0)
        self.m_k = self.m_k_r[:,0:len(self.id_record)].reshape(4*len(self.id_record), 1)

        self.stat_report(p2d)
        self.enable_search()

        self.iter += 1

        if visual == True:
            self.visual(p2d, v2d)
            print('-----------')
            print(new_r_tab)
            print(self.id_record, self.ids)
            print(self.err_std_mtx[0:len(self.id_record), 0:len(self.id_record)])
            # print(max(self.std_px_result), max(self.std_py_result))
            # print(self.r_tab_r[:7, :7])
        else:
            if self.closeFlag is False:
                self.closeFlag = True
                plt.close()

        self.id_record_prev = copy.copy(self.id_record)
        self.id_prev = copy.copy(self.id_new)
        self.ix_prev = copy.copy(self.ix)

        return p2d, v2d, False
