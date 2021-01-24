import numpy as np
from functools import reduce
import math
class nlmpc(object):
    def __init__(self, xref, T):
        self.xref = xref
        self.NX = 4  # x = x, y, v, yaw
        self.NU = 2  # a = [accel, steer]
        self.T = T # horizon length

        # mpc parameters
        self.R = np.diag([0.01, 0.01])  # input cost matrix
        self.Rd = np.diag([0.01, 1.0])  # input difference cost matrix
        self.Q = np.diag([1.0, 1.0, 0.01, 0.00])  # state cost matrix
        self.wq = 1.0
        self.Qf = self.Q  # state final matrix
        # self.GOAL_DIS = 1.5  # goal distance
        # self.STOP_SPEED = 0.5 / 3.6  # stop speed
        # MAX_TIME = 500.0  # max simulation time

        # # iterative paramter
        # MAX_ITER = 3  # Max iteration
        # DU_TH = 0.1  # iteration finish param

        # TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
        # N_IND_SEARCH = 10  # Search index number

        self.DT = 0.2  # [s] time tick

        # Vehicle parameters
        # LENGTH = 4.5  # [m]
        # WIDTH = 2.0  # [m]
        # BACKTOWHEEL = 1.0  # [m]
        # WHEEL_LEN = 0.3  # [m]
        # WHEEL_WIDTH = 0.2  # [m]
        # TREAD = 0.7  # [m]
        self. WB = 2.5  # [m]
        self.MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
        self.MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
        self.MAX_SPEED = 20.0 / 3.6  # maximum speed [m/s]
        self.MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
        self.MAX_ACCEL = 1.0  # maximum accel [m/ss]

    
        # global var by zbl
        # x有T个状态
        TransXMat = np.kron(np.identity(self.T), np.identity(self.NX))
        self.Tx = np.hstack((TransXMat, np.zeros((TransXMat.shape[0], (self.T-1)*self.NU))))
        # print("Tx: %d ,%d" %  (self.Tx.shape[0],self.Tx.shape[1]))
        # u有T-1个状态
        TransUMat = np.kron(np.identity(self.T-1), np.identity(self.NU))
        self.Tu = np.hstack((np.zeros((TransUMat.shape[0], (self.T)*self.NX)), TransUMat))

        # 两个相邻控制量的变化量最小
        self.Pu = np.eye(self.T-1)*(-1) + np.eye(self.T-1, k=1)
        # 删除最后一行
        self.Pu = np.delete(self.Pu, -1, axis=0)       
        self.Pu = np.kron(self.Pu, self.Rd)
        
        # 两个相邻角度控制量的变化量限制
        self.Pc = np.eye(self.T-2, self.T-1)*(-1) + np.eye(self.T-2, self.T-1, k=1)

        # 距离变量的权重
        self.Qx = np.kron(np.identity(self.T), self.Q)
        # 控制变量的权重
        self.Ru = np.kron(np.identity(self.T-1), self.R) 
        # 参考状态
        self.xref_len = self.NX * self.T + self.NU * (self.T-1)
        # self.xref = np.empty(((self.xref_len, -1)))

    def initialize(self, xref, T=None):
        if not T is None:
            self.T = T
        if len(xref) != self.xref_len:
            raise("the length of xref has error!!!")

        self.xref = xref

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        ox = self.Tx @ x
        ou = self.Tu @ x
        '''
        obj: (ox - xref)^T * Q * (ox - xref) + ou^T * Ru *ou + (Pu*ou)^T * (Pu*ou)
        '''
        return 0.5 * ((ox - self.xref).T @ self.Qx @ (ox - self.xref) + ou.T @ (self.Ru + self.Pu.T @ self.Pu) @ ou)
        # return reduce(np.dot, [(ox - self.xref).T, self.Qx, (ox - self.xref)]) \
        #     + reduce(np.dot, [ou, self.Ru, ou]) + reduce(np.dot, [ou.T, self.Pu.T, self.Pu ,ou])

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        ox = self.Tx @ x
        ou = self.Tu @ x

        # return 2*self.Qx @ ox
        return np.concatenate((self.Qx @ ox-self.xref, (self.Ru + self.Pu.T @ self.Pu) @ ou))
        # return np.array([
        #             x[0] * x[3] + x[3] * np.sum(x[0:3]),
        #             x[0] * x[3],
        #             x[0] * x[3] + 1.0,
        #             x[0] * np.sum(x[0:3])
        #             ])

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        od = (self.Tu @ x)[1::2]
        '''
        # 限制Xk与x_(k+1)之间的车辆运动学约束
        dim2 = (T-1) * NX
        constrain2:

        # 限制角速度
        dim1 = (T-2) * NU
        constrain1: self.Pc @ ou
        '''
        a = lambda x, i: [x[i*self.NX] + x[i*self.NX+2]*math.cos(x[i*self.NX+3])*self.DT - x[(i+1)*self.NX], \
                      x[i*self.NX+1] + x[i*self.NX+2]*math.sin(x[i*self.NX+3])*self.DT - x[(i+1)*self.NX+1], \
                      x[i*self.NX+2] + x[self.T*self.NX+i*self.NU]*self.DT - x[(i+1)*self.NX+2], \
                      x[i*self.NX+3] + x[i*self.NX+2]*math.tan(x[self.T*self.NX+i*self.NU+1])*self.DT/self.WB - x[(i+1)*self.NX+3]]
        csts = []
        i = 0
        while i < self.T-1:
            csts = csts + a(x, i)
            i = i + 1
        # print("constraints is \n", np.concatenate((np.array((tuple(csts))), self.Pc @ od)))
        # print("  ")
        # print("constraints1 is \n", np.array((tuple(csts))))
        # print("  ")
        # print("constraints2 is \n", self.Pc @ od)
        # print("  ")
        return np.concatenate((np.array((tuple(csts))), self.Pc @ od))
        
        # return np.array((np.prod(x), np.dot(x, x)))
    def jacobian_i(self, x, i):
        j_i_0 = [0]*(self.T * self.NX + (self.T - 1) * self.NU)

        j_i_0[i*self.NX] = 1
        j_i_0[i*self.NX+2] = math.cos(x[i*self.NX+3])*self.DT
        j_i_0[i*self.NX+3]=  -x[i*self.NX+2]*math.sin(x[i*self.NX+3])*self.DT
        j_i_0[(i+1)*self.NX] = -1

        j_i_1 = [0]*(self.T * self.NX + (self.T - 1) * self.NU)

        j_i_1[i*self.NX+1] = 1
        j_i_1[i*self.NX+2] = math.sin(x[i*self.NX+3])*self.DT
        j_i_1[i*self.NX+3] = x[i*self.NX+2]*math.cos(x[i*self.NX+3])*self.DT
        j_i_1[(i+1)*self.NX+1] = -1

        j_i_2 = [0]*(self.T * self.NX + (self.T - 1) * self.NU)

        j_i_2[i*self.NX+2] = 1
        j_i_2[self.T*self.NX+i*self.NU] = self.DT
        j_i_2[(i+1)*self.NX+2] = -1

        j_i_3 = [0]*(self.T * self.NX + (self.T - 1) * self.NU)

        j_i_3[i*self.NX+3] = 1
        j_i_3[i*self.NX+2] = math.tan(x[self.T*self.NX+i*self.NU+1])*self.DT/self.WB
        j_i_3[i*self.NX+3] = x[i*self.NX+2]*self.DT*(1 + math.pow(math.tan(x[self.T*self.NX+i*self.NU+1]), 2))/self.WB
        j_i_3[(i+1)*self.NX+3] = -1

        return j_i_0 + j_i_1 + j_i_2 + j_i_3

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        '''
        # 每个约束方程
        dim1 = (T * NX + (T-1) * NU) * (T-1) * NX
        '''
        csts = []
        for i in range(self.T-1):
            csts = csts + self.jacobian_i(x, i)

        tmp = self.T * self.NX
        delta_c = np.zeros((self.T-2)* len(x))

        for i in range(self.T-2):
            delta_c[tmp + 1 + self.NU*i] = -1
            delta_c[tmp + 3 + self.NU*i] = 1
            tmp += len(x)

        return np.concatenate((np.array(tuple(csts)), delta_c))

        # return np.concatenate((np.prod(x) / x, 2*x))

    def H_i_k(self, x, i, k):
        H_ = [0]*(self.T * self.NX + (self.T - 1) * self.NU)
        # H_ = tuple(H_)
        H_T = [H_]*(self.T * self.NX + (self.T - 1) * self.NU)

        if k == 0:
            H_T[i*self.NX+3][i*self.NX+2] = -math.sin(x[i*self.NX+3])*self.DT
            H_T[i*self.NX+3][i*self.NX+3] = -x[i*self.NX+2]*math.cos(x[i*self.NX+3])*self.DT
            return np.array(tuple(H_T))
        elif k == 1:
            H_T[i*self.NX+3][i*self.NX+2] = math.cos(x[i*self.NX+3])*self.DT
            H_T[i*self.NX+3][i*self.NX+3] = -x[i*self.NX+2]*math.sin(x[i*self.NX+3])*self.DT
            return np.array(tuple(H_T))
        elif k == 3:
            H_T[self.T*self.NX+i*self.NU+1][i*self.NX+2] = (1 + math.pow(math.tan(x[self.T*self.NX+i*self.NU+1]), 2))*self.DT/self.WB
            H_T[self.T*self.NX+i*self.NU+1][self.T*self.NX+i*self.NU+1] = x[i*self.NX+2]*(2*math.tan(x[self.T*self.NX+i*self.NU+1])*(1 + math.pow(math.tan(x[self.T*self.NX+i*self.NU+1]), 2)))*self.DT/self.WB
            return np.array(tuple(H_T))
        else:
            return np.array(tuple(H_T))

    # def hessianstructure(self):
    #     #
    #     # The structure of the Hessian
    #     # Note:
    #     # The default hessian structure is of a lower triangular matrix. Therefore
    #     # this function is redundant. I include it as an example for structure
    #     # callback.
    #     #
    #     global hs

    #     hs = sps.coo_matrix(np.tril(np.ones((4, 4))))
    #     return (hs.col, hs.row)

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        rigth_bottle = (self.Ru + self.Pu.T @ self.Pu)
        
        top = np.hstack((self.Qx, np.zeros((self.Qx.shape[1],rigth_bottle.shape[1]))))
        # H = obj_factor * top
        battle = np.hstack((np.zeros((rigth_bottle.shape[0], self.Qx.shape[1])), rigth_bottle))
        H = obj_factor*np.vstack((top, battle))
        # print("H is\n", H)
        # print("  ")
        for i in range(self.T-1):
            for k in range(self.NX):
                H += lagrange[i*self.NX+k] * self.H_i_k(x, i, k)
        
        # H = obj_factor*np.array((
        #         (2*x[3], 0, 0, 0),
        #         (x[3],   0, 0, 0),
        #         (x[3],   0, 0, 0),
        #         (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

        # H += lagrange[0]*np.array((
        #         (0, 0, 0, 0),
        #         (x[2]*x[3], 0, 0, 0),
        #         (x[1]*x[3], x[0]*x[3], 0, 0),
        #         (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

        # H += lagrange[1]*2*np.eye(4)

        #
        # Note:
        #
        #
        # return H[len(x), len(x)]
        return H

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):
        pass
        #
        # Example for the use of the intermediate callback.
        #
        # print ("Objective value at iteration #%d is - %g" % (iter_count, obj_value))