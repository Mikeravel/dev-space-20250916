import numpy as np

from MPPWPA import mppwpa
import initial as ini
import matplotlib.pyplot as plt
#from original_wolf import original_wolf
from mycode import mdwpa
#from LWPA import mpwpa
from NewWPA import nlwpa


def main():
    plan, task_complete_time, average_time, gbest_fitness, best_fitness_mat = nlwpa().run()

    print('改进狼群算法求解情况：')
    print('分配方案：', plan)
    print('分配方案适应度值：', gbest_fitness)
    print('方案任务完成时间：', task_complete_time)
    print('方案无人机平均飞行时间：', average_time)

    plt.figure('适应度值变化曲线')
    x = np.arange(0, ini.max_iteration, 1)
    y = best_fitness_mat
    plt.plot(x, y, label='WPA')
    plt.legend()
    plt.xlabel('Iter')
    plt.ylabel('Fintess')
    plt.show()

    # 画出目标和我方编队分布示意图
    plt.figure('任务分配情况示意图')
    marker = ['o', 'v']  # 图标，分别表示目标和我方兵力
    color = ['b', 'r']  # 颜色，分别表示目标和我方兵力
    area = [i * 10 for i in ini.tar_range]
    x_targets = [0.0 for k in range(ini.nt)]
    y_targets = [0.0 for k in range(ini.nt)]
    for k in range(ini.nt):
        x_targets[k] = ini.t_loc[k][0]
        y_targets[k] = ini.t_loc[k][1]
    plt.scatter(x_targets, y_targets, s=area, marker=marker[0], color=color[0])
    plt.xlim(0, 400)  # 坐标轴范围
    plt.ylim(0, 400)
    x_uavs = [0.0 for k in range(ini.nu)]
    y_uavs = [0.0 for k in range(ini.nu)]
    for i in range(ini.nu):
        x_uavs[i] = ini.u_start_loc[i][0]
        y_uavs[i] = ini.u_start_loc[i][1]
    plt.scatter(x_uavs, y_uavs, marker=marker[1], color=color[1])
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=10, loc='upper right', labels=['Targets', 'Ships equipped with UAV'])  # 打印图例
    plt.show()

    # 画出目标和我方编队分布示意图
    plt.figure('任务分配情况示意图')
    marker = ['o', 'v']  # 图标，分别表示目标和我方兵力
    color = ['b', 'r']  # 颜色，分别表示目标和我方兵力
    ucolor = ['brown', 'sienna', 'peru', 'gold', 'lawngreen',
              'darkgreen', 'turquoise', 'steelblue', 'slateblue', 'deeppink']  # 颜色，用于区分各无人机航线
    plt.xlim(0, 400)  # 坐标轴范围
    plt.ylim(0, 400)
    # 目标散点图

    for k in range(ini.nt):
        x_targets[k] = ini.t_loc[k][0]
        y_targets[k] = ini.t_loc[k][1]
    plt.scatter(x_targets, y_targets, s=area, marker=marker[0], color=color[0], label='Targets')
    x_uavs = [0.0 for k in range(ini.nu)]
    y_uavs = [0.0 for k in range(ini.nu)]
    # 无人机平台散点图
    for i in range(ini.nu):
        x_uavs[i] = ini.u_start_loc[i][0]
        y_uavs[i] = ini.u_start_loc[i][1]
    plt.scatter(x_uavs, y_uavs, marker=marker[1], color=color[1], label='Ships')
    # 无人机航线图
    for i in range(ini.nu):
        if len(plan[i]) != 0:
            tar = plan[i]  # 记录第i架无人机的任务
            x = [0.0 for k in range(len(tar) + 2)]  # 用来记录无人机航路点
            y = [0.0 for k in range(len(tar) + 2)]
            x[0] = ini.u_start_loc[i][0]  # 无人机初始位置
            y[0] = ini.u_start_loc[i][1]
            for k in range(1, len(tar) + 1):  # 无人机各目标点
                t = tar[k - 1]
                x[k] = ini.t_loc[t][0]
                y[k] = ini.t_loc[t][1]
            x[len(tar) + 1] = ini.u_start_loc[i][0]  # 返回初始位置
            y[len(tar) + 1] = ini.u_start_loc[i][1]
            plt.plot(x, y, linewidth=0.8, color=ucolor[i], label='U$_%.0f$' % i)
            for j in range(len(tar)):
                plt.arrow(x[j], y[j], x[j + 1] - x[j], y[j + 1] - y[j], head_width=1, width=0.05, color=ucolor[i])
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=10, loc='upper right')  # 打印图例
    plt.show()


if __name__ == "__main__":
    main()