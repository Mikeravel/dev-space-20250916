"""
根据初始化的无人机初始位置、目标位置，计算无人机和目标之间、各目标之间的距离
"""

import math
import initial as ini

# 确定无人机到各目标点欧式距离
UtoT = [[0.0 for k in range(ini.nt)] for t in range(ini.nu)]
# 生成一个nu行、nt列的列表，元素全部为0,nu=无人机数目,nt=目标的数目
for i in range(ini.nu):
    for j in range(ini.nt):
        d_x = ini.u_start_loc[i][0] - ini.t_loc[j][0]   # 第i架无人机位置的横坐标-第j个目标的横坐标
        d_y = ini.u_start_loc[i][1] - ini.t_loc[j][1]   # 第i架无人机位置的纵坐标-第j个目标的纵坐标
        UtoT[i][j] = math.sqrt(d_x**2 + d_y**2)    # 计算两点之间的欧式距离

# 各目标点之间的距离
TtoT = [[0.0 for k in range(ini.nt)] for t in range(ini.nt)]
# 生成一个nt行、nt列的列表，元素全部为0,column=目标的数目
for i in range(ini.nt):
    for j in range(ini.nt):
        d_x = ini.t_loc[i][0] - ini.t_loc[j][0]
        d_y = ini.t_loc[i][1] - ini.t_loc[j][1]
        TtoT[i][j] = math.sqrt(d_x**2 + d_y**2)

# fitness和limit两个程序都需要调用该函数
# 计算某个粒子解码后的分配方案中各个无人机的航行距离
def dis_U(UdoT):                   # UdoT表示某个粒子解码后的任务分配方案
    U_range = [0.0 for k in range(ini.nu)]      # U_range为一个1维矩阵，有nu各元素，分别用来存储各无人机航程
    U_complete = [0.0 for k in range(ini.nu)]    # U_complete为一个1维矩阵，有nu各元素，分别用来存储各无人机完成侦察任务的距离
    for i in range(len(UdoT)):
        tasklist = UdoT[i].copy()
        if len(tasklist) == 0:
            U_range[i] = 0          # 无人机未分配任务则航程为零
        elif len(tasklist) == 1:
            U_range[i] = 2*UtoT[i][tasklist[0]] + ini.tar_range[tasklist[0]]
            # 无人机从出发点前往任务地点执行任务并返回
        else:
            U_range[i] = UtoT[i][tasklist[0]]
            for j in range(len(tasklist) - 1):  # 计算在各任务点间转移、对各任务点实施侦察行动中的航程
                U_range[i] += TtoT[tasklist[j]][tasklist[j + 1]] + ini.tar_range[tasklist[j]]
            U_complete[i] = U_range[i] + ini.tar_range[tasklist[-1]]
            U_range[i] += ini.tar_range[tasklist[-1]] + UtoT[i][tasklist[-1]]  # 加上最后一个任务的侦察距离和返回距离
    return U_range, U_complete
