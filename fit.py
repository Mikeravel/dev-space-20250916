"""
计算某个粒子的适应度值，考虑：1.时间代价；2.航程代价
"""
import distence as dis
import initial as ini
import numpy as np

w1 = 0.3  # 适应度中侦察完成时间所占权重
w2 = 0.3  # 适应度中无人机平均飞行距离所占权重
w3 = 0.4  # 适应度中目标价值大小



def fit_cal(UdoT):
    CT, AT = timecost(UdoT)
    VT = calculate_custom_score(ini.t_value)
    fitness = w1 * CT + w2 * AT + w3 * VT
    return fitness

def calculate_custom_score(int_list: list[int]) -> int:
    """
    根据新的计分规则计算列表的分数。

    新计分规则:
    1. 列表的逆序数最大时（完全降序），基础分数为 100。
    2. 逆序数从最大值每减少 1，分数增加 50。
    
    公式:
    分数 = 100 + (最大逆序数 - 实际逆序数) * 50

    Args:
        int_list: 一个只包含整数的列表。

    Returns:
        根据新规则计算出的分数。
    """
    n = len(int_list)
    # 如果列表为空或只有一个元素，没有逆序，分数应为最高
    # 按照公式，最大逆序数为0，实际逆序数也为0，分数为100
    if n < 2:
        return 100

    # 1. 计算实际的逆序数 (这部分逻辑不变)
    inversion_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if int_list[i] > int_list[j]:
                inversion_count += 1

    # 2. 计算理论上的最大逆序数 (这部分逻辑不变)
    max_inversions = n * (n - 1) // 2

    # 3. 根据新规则计算最终分数 (这部分逻辑已修改)
    score = 100 + (max_inversions - inversion_count) * 50
    
    return score

# 计算所有无人机侦察任务完成时间，最长的时间作为时间代价
def timecost(UdoT):
    time_uav = [0.0 for k in range(ini.nu)]             # 存储各无人机飞行时间
    time_uav_complete = [0.0 for k in range(ini.nu)]    # 存储各无人机侦察完成时间
    for i in range(ini.nu):
        tasklist = UdoT[i].copy()
        if len(tasklist) == 0:
            time_uav[i] = 0.0              # 无人机未分配任务则耗时为零
            time_uav_complete[i] = 0.0
        elif len(tasklist) == 1:
            time_uav[i] = (2 * dis.UtoT[i][tasklist[0]] + ini.tar_range[tasklist[0]]) / ini.uav_vc[i]
            time_uav_complete[i] = (dis.UtoT[i][tasklist[0]] + ini.tar_range[tasklist[0]]) / ini.uav_vc[i]
        else:
            range1 = 0.0    # 巡航过程航行距离
            range2 = 0.0    # 侦察过程航行距离
            range1 += dis.UtoT[i][tasklist[0]]
            for j in range(len(tasklist)-1):
                range1 += dis.TtoT[tasklist[j]][tasklist[j + 1]]
                range2 += ini.tar_range[tasklist[j]]
            range2 += ini.tar_range[tasklist[-1]]
            time_uav_complete[i] = (range1 + range2) / ini.uav_vc[i]
            time_uav[i] = (range1 + range2 + dis.UtoT[i][tasklist[-1]]) / ini.uav_vc[i]
    return max(time_uav_complete), np.mean(time_uav)

# 计算所有无人机的总航程和，作为航程代价
def trackcost(UdoT):
    sum_track = 0.0
    U_range = dis.dis_U(UdoT)
    for i in range(ini.nu):
        sum_track += U_range[i]
    return sum_track

