"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import matplotlib.pyplot as plt
import cvxpy
from cvxopt import matrix, solvers
import math
import numpy as np
np.set_printoptions(threshold=np.inf)
from functools import reduce
import sys
sys.path.append("../../PathPlanning/CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise


NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 5 # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 5.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.01])  # state cost matrix
wq = 1.0
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 20.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

show_animation = True

# global var by zbl
# x有T+1个状态
TransXMat = np.kron(np.identity(T+1), np.identity(NX))
Tx = np.hstack((TransXMat, np.zeros((TransXMat.shape[0], T*NU))))
# u有T个状态
TransUMat = np.kron(np.identity(T), np.identity(NU))
Tu = np.hstack((np.zeros((TransUMat.shape[0], (T+1)*NX)), TransUMat))

# 两个相邻控制量的变化量最小
Pu = np.eye(T)*(-1) + np.eye(T, k=1)
# 删除最后一行
Pu = np.delete(Pu, -1, axis=0)
Pu = np.kron(Pu, Rd)

# 距离变量的权重
Qx = np.kron(np.identity(T+1), Q)
# 控制变量的权重
Ru = np.kron(np.identity(T), R)

class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, predelta=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = predelta


def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def get_linear_model_matrix(v, phi, delta):

    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB
   

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")


def update_state(state, a, delta):

    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER
    # state.predelta += delta
    # # print(MAX_STEER)
    # if state.predelta >= MAX_STEER:
    #     state.predelta  = MAX_STEER
    # elif state.predelta <= -MAX_STEER:
    #     state.predelta  = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT
    
    if state. v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state. v < MIN_SPEED:
        state.v = MIN_SPEED

    return state


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, cyaw, pind):

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar


def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
    MPC contorl with updating operational point iteraitvely
    """

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control3(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov

# added by zbl
def matrix_pow(base, exp):
    if base.shape[0] != base.shape[1]:
        raise Exception("base matrix has error! perhaps, the number of rows and columns is not equal")
    if exp == 1:
        return base
    if exp == 0:
        return np.eye(base.shape[0])
    res = base
    for i in range(exp-1):
        res = np.dot(res, base)
    return res

def get_ox_oy_from_mpc(x0, oa, od):
    cur = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2], predelta=x0[4])
    ox = np.array(cur.x)
    oy = np.array(cur.y)
    oyaw = np.array(cur.yaw)
    ov = np.array(cur.v)
    n = len(oa)
    for i in range(n):
        cur = update_state(cur, oa[i], od[i])
        ox = np.append(ox, cur.x)
        oy = np.append(oy, cur.y)
        oyaw = np.append(oyaw, cur.yaw)
        ov = np.append(ov, cur.v)
    return ox[1:], oy[1:], oyaw[1:], ov[1:]
    
def get_ox_oy_from_mpc2(J, Z_k, K, u, L, C):
    '''
    状态转移递推公式
    Z = J*Z_k + Ku + LC
    '''
    Z = np.dot(J, np.array(Z_k)) + np.dot(K, u) + np.dot(L, C)
    # print("the shape of np.dot(J, Z_k) is (%f)" % np.dot(J, np.array(Z_k)).shape[0])
    # print("the shape of np.dot(K, u) is (%f)" % np.dot(K, u).shape[0])
    # print("pre Z is \n", Z)
    Z = Z.reshape((-1, NX))
    # print("the shape of Z is (%f, %f)" % (Z.shape[0], Z.shape[1]))
    # print("Z is \n", Z)
    # print("ox is : ", Z[:, 0])
    # print("oy is : ", Z[:, 1])
    # print("ov is : ", Z[:, 2])
    # print("oyaw is : ", Z[:, 3])
    # raise("111")
    return Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]


# def get_M_matrix(T, dt, NU):
#     M = np.zeros((T, T*NU+1))
#     for i in range(T):
#         for j in range(0, i+1, 2):
#             M[i][j] = dt
    
#     # 拼凑一个限制最低速度的状态转移矩阵
#     return np.vstack((M, M*(-1)))

def get_M_matrix(T, dt, NU):
    M = np.zeros((T, T*NU))
    for i in range(T):
        for j in range(0, i+1, 2):
            M[i][j] = dt
    
    # 拼凑一个限制最低速度的状态转移矩阵
    return np.vstack((M, M*(-1)))


def linear_mpc_control2(xref, xbar, x0, dref):
    """
    linear mpc control

    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """
    
    # v, phi(yaw), delta(steer)
    A, B, C = get_linear_model_matrix(
        x0[2], x0[3], dref[0, 0])
    print("%f, %f, %f" % (x0[2], x0[3], dref[0,0]))
    '''
    状态转移递推公式
    Z = J*Z_k + Ku + LC
    其中 Z = [Z_k+1, Z_k+2,...., Z_k+T]
    '''
    J = A
    K = np.hstack((B, np.zeros((B.shape[0], B.shape[1]*(T-1)))))
    L = np.identity(NX)
    for i in range(2, T+1):
        # 初始状态矩阵
        J = np.vstack((J, matrix_pow(A, i)))
        
        # 状态转移矩阵
        tmp = np.dot(matrix_pow(A, i-1), B)
        # print("pow(A, %d)*B" % (i-1))
        for j in range(i-2, 0, -1):
            # print("pow(A, %d)*B" % (j))
            tmp = np.hstack((tmp, np.dot(matrix_pow(A, j), B)))
        tmp = np.hstack((tmp, B))
        tmp = np.hstack((tmp, np.zeros((B.shape[0], B.shape[1]*(T-i)))))
        K = np.vstack((K, tmp))
        # C常数矩阵
        # print("================")
        tmp = np.identity(NX)
        # print("I")
        for j in range(1, i):
            # print("+")
            # print("pow(A, %d)" % (j))
            tmp += matrix_pow(A, j)
        # print("===============")
        L = np.vstack((L, tmp))
    # print("J is: \n", J)
    # print("K is: \n", K)
    # print("L is: \n", L)
    # print("C is: \n", C)
    # print("B is :\n", B)
    # print("AB is : \n", np.dot(A,B))
    np.savetxt("A.txt", A,fmt='%f',delimiter=',')
    np.savetxt("J.txt", J,fmt='%f',delimiter=',')
    np.savetxt("K.txt", K,fmt='%f',delimiter=',')
    np.savetxt("L.txt", L,fmt='%f',delimiter=',')
    np.savetxt("xref.txt", xref,fmt='%f',delimiter=',')
    print("xref[:, :-1] is \n", xref[:, :-1])
    print("xref is \n", xref)
    zref = xref[:, :-1].reshape(-1, 1, order='F')[:, 0]
    print("zref is \n", zref)
    # print(zref.shape[0])
    # raise("1111")
    '''
    E = J*Z_k + LC - Z_ref 
    '''
    # zk = np.array([[x0[0]], [x0[1]], [x0[2]], [x0[3]]])
    zk = np.array([x0[0], x0[1], x0[2], x0[3]])
    # print("zk is: \n ", zk)
    # print("np.dot(J, zk) : \n", np.dot(J, zk))
    # print("np.dot(L, C) : \n", np.dot(L, C))
    # print("J: ", J.shape[0], J.shape[1])
    # print("zref is: ", zref)
    # print("zref: ", zref.shape[0], zref.shape[1])
    # tmp = np.dot(J, zk)
    # print("np.dot(J, zk): ", tmp.shape[0], tmp.shape[1])
    # tmp = np.dot(L, C)
    print("np.dot(J, zk) + np.dot(L, C): ", np.dot(J, zk) + np.dot(L, C))
    # raise("1111")
    E = np.dot(J, zk) + np.dot(L, C) - zref
    # print("E: ", E.shape[0])
    print("E is \n", E)
    # raise("111")
    # 优化距离的权重
    W = np.kron(np.identity(T) * np.array([wq **i for i in range(T)]), Q)
    np.savetxt("W.txt", W,fmt='%f',delimiter=',')
    # print("W is \n", W)

    # 优化控制量的权重
    R = np.kron(np.identity(T), np.diag([0.01, 0.01]))
    # 松弛因子的权重
    p = 10
    # 两个相邻控制量的变化量最小
    P = np.eye(T)*(-1) + np.eye(T, k=1)
    # 删除最后一行
    P = np.delete(P, -1, axis=0)
    P = np.kron(P, Rd)
    np.savetxt("P_Rd.txt", P,fmt='%f',delimiter=',')
    # P = np.kron(, Rd) 
    # print(P)
    # print(P.shape[0], P.shape[1])
    h_tmp = reduce(np.dot, [K.T, W, K]) + R + np.dot(P.T, P)
    # print("reduce(np.dot, [K.T, W]) is1\n", reduce(np.dot, [K.T, W]))
    # print("reduce(np.dot, [K.T, W, K]) is\n", reduce(np.dot, [K.T, W, K]))
    # raise("1111")
    # print(h_tmp.shape[0], P.shape[1])

    p_tmp = np.vstack((np.zeros((h_tmp.shape[0], 1)), np.identity(1)*p))
    # 注意这里拼凑 p-tmp 与 h-tmp的顺序不能颠倒
    h_tmp = np.vstack((h_tmp, np.zeros((1, h_tmp.shape[1]))))
    # print(h_tmp.shape[0], h_tmp.shape[1])
    # print(p_tmp.shape[0])
    h_tmp = np.hstack((h_tmp, p_tmp))
    H = matrix(h_tmp)
    # print(h_tmp.shape[0], h_tmp.shape[1])
    g_tmp = 2*reduce(np.dot, [E.T, W, K])
    g_tmp = np.append(g_tmp, 0)
    # g_tmp = np.hstack((g_tmp, np.zeros((g_tmp.shape[0],1))))
    # 这里需要转置
    g = matrix(g_tmp.T)
    # print(g)
    '''
    min 1/2*X^T*H*X+G^T*X
    S.T.: MX <= b
          NX = h
        A = 状态累加矩阵
        M = [[A， O], [-A, O], [控制量上限矩阵], [控制量下限矩阵]]
        b = [Vmax - Vcur, STEERmax-STEERcur, -Vmin+Vcur, -STEERmin+STEERcur, 
        MAX_ACCEL, MAX_STEER, -MAX_ACCEL, -MAX_STEER]
    '''
    M = get_M_matrix(T, DT, NU)
    # print(M.shape[0], M.shape[1])
    # print(P.shape[0], P.shape[1])
    # raise("1111")
    # M = np.kron(np.tri(T), np.identity(NU))
    # # v和presteer
    # M = np.vstack((np.hstack((M, np.zeros((M.shape[0], 1)))) ,np.hstack((M*(-1), np.zeros((M.shape[0], 1))))))
    # a和steer上限
    # T*NU 是T时域内的控制量U,+1是为了控制松弛因子
    M = np.vstack((M, np.identity(T*NU+1)))
    # 松弛因子不用考虑
    # M[-1, -1] = 0
    # a和steer下限
    M = np.vstack((M, np.identity(T*NU+1)*(-1)))
    # 松弛因子不用考虑
    # M[-1, -1] = 0

    # 限制角速度与jerk（加加速度）
    P = np.eye(T)*(-1) + np.eye(T, k=1)
    # 删除最后一行
    P = np.delete(P, -1, axis=0)
    P = np.kron(P, np.identity(NU))
    np.savetxt("P.txt", P,fmt='%f',delimiter=',')
    # print(M.shape[0], M.shape[1])
    M = np.vstack((M, np.hstack((P, np.zeros((P.shape[0], 1))))))
    M = np.vstack((M, np.hstack((P*(-1), np.zeros((P.shape[0], 1))))))

    # 限制x和y
    # 正方向
    K_tmp = np.hstack((K, np.zeros((K.shape[0],1))))
    M = np.vstack((M, K_tmp))
    #负方向
    M = np.vstack((M, K_tmp * (-1)))
    Z = np.dot(J, np.array(zk))+ np.dot(L, C)
    # print(M.shape[0], M.shape[1])
    b = []

    '''
    x0 is current state: [x, y, v, yaw, predelta]
    '''
    # v和presteer上限
    for i in range(T):
        b.append(MAX_SPEED - x0[2])
    # v和presteer下限
    for i in range(T):
        # STEERmin = -MAX_STEER
        b.append(-MIN_SPEED+ x0[2])
    # a和steer上限
    for i in range(T):
        b.extend([MAX_ACCEL, MAX_STEER])
    # 松弛因子
    b.append(100)
    # a和steer下限
    for i in range(T):
        b.extend([MAX_ACCEL, MAX_STEER])
    # 松弛因子
    b.append(100)

    # jerk与角速度上下限
    for i in range(T-1):
        b.extend([DT * MAX_ACCEL, DT * MAX_DSTEER])
    for i in range(T-1):
        b.extend([DT * MAX_ACCEL, DT * MAX_DSTEER])

    # 限制路径的x与y
    # 正方向
    print("zref is \n", zref)
    print("np.kron is \n", np.kron(np.ones(T), np.array([1, 1, 100, 100])))
    
    tmp = zref + np.kron(np.ones(T), np.array([3, 3, 100, 100])) - Z
    print("tmp is \n", tmp)
    # raise("111")
    b.extend(tmp)
    # 负方向
    tmp = -(zref - np.kron(np.ones(T), np.array([5, 5, 100, 100]))) + Z
    b.extend(tmp)
    # print(b))
    # b = np.array(b)
    M = matrix(M)
    b = matrix(b)
    np.savetxt("H.txt", H,fmt='%f',delimiter=',')
    np.savetxt("g.txt", g,fmt='%f',delimiter=',')
    np.savetxt("M.txt", M,fmt='%f',delimiter=',')
    np.savetxt("b.txt", b,fmt='%f',delimiter=',')
    sol=solvers.qp(H, g, M, b)
    # print(type(sol['x']))
    oa = np.array(sol['x'][0:-1:2]).reshape((1, T))[0]
    od = np.array(sol['x'][1:-1:2]).reshape((1, T))[0]
    u = np.array(sol['x'][0:-1])[:, 0]
    print("sol: \n", sol['x'])
    print("u is: \n", u)
    # print(oa)
    ox, oy, oyaw, ov = get_ox_oy_from_mpc(x0, oa, od)
    # ox, oy, ov, oyaw = get_ox_oy_from_mpc2(J, zk, K, u, L, C)
    # print(ox)
    # print("oa:\n", oa)
    # print("od:\n", od) 
    
    # print(oa[0])
    # print(od[0])
    return oa, od, ox, oy, oyaw, ov

def modify_matrix(ori_mat, t, A, B):
    row = (t+1)*NX
    col = t*NX
    # 从row, col处，将tar_mat赋值到ori_mat中
    if ori_mat.shape[0] < A.shape[0] or ori_mat.shape[1] < A.shape[1]:
        raise Exception("the row or col number of tar_mat is less than the ori_mat !!!")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ori_mat[row+i, col+j] = A[i, j]
    col = (T+1) * NX + t*NU
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            ori_mat[row+i, col+j] = B[i, j]

def linear_mpc_control3(xref, xbar, x0, dref):
    """
    linear mpc control

    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """
    x0 = x0[:NX]
    zref = xref.reshape(-1, 1, order='F')[:, 0]
    H = reduce(np.dot, [Tx.T, Qx, Tx]) + reduce(np.dot, [Tu.T, Ru, Tu]) + reduce(np.dot, [Tu.T, Pu.T, Pu ,Tu])
    g = -2 * reduce(np.dot, [zref, Qx, Tx])
    '''
    min 1/2*X^T*H*X+G^T*X
    S.T.: MX <= b
          NX = h
        A = 状态累加矩阵
        M = [[A， O], [-A, O], [控制量上限矩阵], [控制量下限矩阵]]
        b = [Vmax - Vcur, STEERmax-STEERcur, -Vmin+Vcur, -STEERmin+STEERcur, 
        MAX_ACCEL, MAX_STEER, -MAX_ACCEL, -MAX_STEER]
    '''
    # M矩阵是限制速度上下限
    M = get_M_matrix(T, DT, NU)
    # T*NU 是T时域内的控制量U
    M = np.vstack((M, np.identity(T*NU)))
    # a和steer下限
    M = np.vstack((M, np.identity(T*NU)*(-1)))
    # 限制角速度与jerk（加加速度）
    P = np.eye(T)*(-1) + np.eye(T, k=1)
    # 删除最后一行
    P = np.delete(P, -1, axis=0)
    P = np.kron(P, np.identity(NU))
    # print(M.shape[0], M.shape[1])
    M = np.vstack((M, P))
    M = np.vstack((M, P*(-1)))

    # print(M.shape[0], M.shape[1])
    b = []

    '''
    x0 is current state: [x, y, v, yaw, predelta]
    '''
    # v和presteer上限
    for i in range(T):
        b.append(MAX_SPEED - x0[2])
    # v和presteer下限
    for i in range(T):
        # STEERmin = -MAX_STEER
        b.append(-MIN_SPEED + x0[2])
    # a和steer上限
    for i in range(T):
        b.extend([MAX_ACCEL, MAX_STEER])
    # a和steer下限
    for i in range(T):
        b.extend([MAX_ACCEL, MAX_STEER])

    # jerk与角速度上下限
    for i in range(T-1):
        b.extend([DT * MAX_ACCEL, DT * MAX_DSTEER])
    for i in range(T-1):
        b.extend([DT * MAX_ACCEL, DT * MAX_DSTEER])
    # 行是NX， 列是NU+NX
    N = np.zeros((NX*(T+1), (NU+NX)*T+NX))
    h = []
    # 限制x0
    h.extend(x0)
    for i in range(NX):
        N[i, i] = 1
    for t in range(T):
        A, B, C = get_linear_model_matrix(
            xbar[2, t], xbar[3, t], dref[0, t])
        h.extend(-C)
        # 修改矩阵N,t+1开始
        modify_matrix(N, t, np.hstack((A, (-1)*np.identity(NX))), B)
    
    print("mat N is \n", N)
    print("h is\n", h)
    np.savetxt("N.txt", N,fmt='%f',delimiter=',')  
    H = matrix(H)
    g = matrix(g)
    M = matrix(M @ Tu)
    b = matrix(b)
    N = matrix(N)
    h = matrix(h)
    sol=solvers.qp(H, g, M, b, N, h)
    x = np.array(sol['x'][:(T+1)*NX])[:, 0]
    u = np.array(sol['x'][(T+1)*NX:])[:, 0]
    
    ox = x[0::NX]
    oy = x[1::NX]
    ov = x[2::NX]
    oyaw = x[3::NX]

    oa = u[0::2]
    od = u[1::2]
    print("x is\n", x)
    print("ox is\n", ox)
    print("sol: \n", sol['x'])
    print("u is: \n", u)
    # raise("1111")
    return oa, od, ox, oy, oyaw, ov



def linear_mpc_control(xref, xbar, x0, dref):
    """
    linear mpc control

    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """
    x0 = x0[:NX]
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)
        # v, phi(yaw), delta(steer)
        A, B, C = get_linear_model_matrix(
            xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                            MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None
    # print("oa: ", type(oa))
    return oa, odelta, ox, oy, oyaw, ov


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0
    # if state.v < 0.5:
    #     state.v = 0.5
    cur = ind
    for i in range(T + 1):
        travel += abs(state.v) * DT
        # print("travel: ", travel)
        # 从连续的路径上，取v*dt间隔路径点
        dind = int(round(travel / dl))
        # if dind == 0:
        #     dind = cur + 1 
        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0
        cur = dind
    return xref, ind, dref


def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False


def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    Simulation

    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]

    """

    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None
    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind)

        x0 = [state.x, state.y, state.v, state.yaw]  # current state
        # dref[0, 0] = d[-1]
        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta)

        if odelta is not None:
            di, ai = odelta[0], oa[0]

        state = update_state(state, ai, di)
        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        d.append(di)
        a.append(ai)
        # added by zbl
        # self.predelta += di
        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
            plt.pause(0.0001)

    return t, x, y, yaw, v, d, a


def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def get_straight_course(dl):
    ax = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course2(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course3(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    cyaw = [i - math.pi for i in cyaw]

    return cx, cy, cyaw, ck


def get_forward_course(dl):
    ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_switch_back_course(dl):
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [35.0, 10.0, 0.0, 0.0]
    ay = [20.0, 30.0, 5.0, 0.0]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw2 = [i - math.pi for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)

    return cx, cy, cyaw, ck


def main():
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    # cx, cy, cyaw, ck = get_straight_course(dl)
    # cx, cy, cyaw, ck = get_straight_course2(dl)
    # cx, cy, cyaw, ck = get_straight_course3(dl)
    # cx, cy, cyaw, ck = get_forward_course(dl)
    cx, cy, cyaw, ck = get_switch_back_course(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.3)

    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, d, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


def main2():
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    cx, cy, cyaw, ck = get_straight_course3(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=0.0, v=0.0)

    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


if __name__ == '__main__':
    main()
    # main2()
