"""
对粒子的编码进行解码，得到各粒子对应的解
"""

import initial as ini

def deco(xcode):  # 浮点型编码的解码方案
    UdoT = [[] for k in range(ini.nu)]         # 使用一个列表存储每架无人机需要执行的任务
    for i in range(ini.nt):
        UdoT[int(xcode[i])].append(i)          # 编码每一维的整数部分代表由第几架无人机执行该任务
    for k in range(ini.nu):                    # 对每架无人机分配到的目标，按照编码由小到大的顺序排序
        for l in range(len(UdoT[k])-1):
            for j in range(l, len(UdoT[k])-1):
                if xcode[UdoT[k][j]] > xcode[UdoT[k][j+1]]:
                    UdoT[k][j], UdoT[k][j+1] = UdoT[k][j+1], UdoT[k][j]
    return UdoT