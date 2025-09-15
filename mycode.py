import math
import fit
import initial as ini
import numpy as np
import random
import deco
import limit
import random
import re_deco
import possible as poss
import time

# 自适应步长+随机召唤+个体生成

def mdwpa():
    #start_time = time.time()
    population_size = 100   # 狼群数量
    max_iteration = 1000     # 最大迭代次数
    a = 2                   # 探狼比例因子。决定每轮迭代中探狼的数量
    b = 7                   # 更新比例因子，每轮迭代淘汰/重新生成狼群的数量
    q = 2                   # 围捕因子，决定随机选择的位置
    Tmax = 10              # 每轮迭代中的随机游走次数
    iter = 0                # 实际迭代次数
    T = 100
    mn = [[]for k in range(population_size)]    # 表示要执行的任务序列，任务序号出现的顺序表示任务执行的顺序。同时也是人工狼群的坐标
    un = [[]for k in range(population_size)]    # 表示该任务分配给第几架无人机完成
    # mn = [[11,13,17,12,6,3,20,4,21,22,14,5,24,18,19,0,7,1,16,9,23,8,10,15,2],[20,18,7,23,8,14,19,0,2,22,24,21,16,13,3,17,11,5,4,9,1,10,15,12,6],[19,23,16,0,5,11,12,24,18,8,13,1,2,15,6,20,22,7,9,14,17,4,10,3,21],[10,2,18,0,16,20,12,11,24,15,9,17,1,21,8,13,3,22,7,23,4,6,19,5,14],[18,5,23,12,2,16,20,21,3,8,15,22,13,19,24,7,11,14,17,1,10,6,9,4,0]]
    # un = [[9,6,3,3,4,0,5,3,6,6,3,1,0,2,4,2,2,6,4,1,8,9,0,5,0],[7,3,9,9,6,9,3,3,0,4,0,3,9,3,7,1,1,6,0,2,2,6,3,1,1],[0,9,5,0,6,4,3,3,1,0,9,7,2,0,1,5,0,6,2,7,6,4,3,3,2],[8,7,0,4,3,9,5,2,9,5,6,3,4,1,5,0,5,5,4,2,8,4,9,9,9],[8,7,2,6,6,7,7,4,8,6,1,1,3,9,8,0,1,8,6,7,3,3,7,2,3]]
    UdoT_list = [[] for k in range(population_size)]
    fitness = np.zeros(population_size)
    fit_record = [[] for i in range(max_iteration)]                                     # 记录每次迭代最优的适应度值
    plan_record =[[0.0 for i in range(ini.nu)] for j in range(max_iteration)]           # 记录每次迭代最好的分配方案

    # 狼群的初始化
    for i in range(population_size):
        mn[i] = random.sample(range(0,ini.nt),ini.nt)
        un[i] = [random.randint(0,ini.nu - 1) for i in range(ini.nt)]
        UdoT_list[i] = deco.deco(mn[i],un[i])
        if limit.limit_verify(UdoT_list[i]) == 0:
            fitness[i] = float("inf")
        else:
            fitness[i] = fit.fit_cal(UdoT_list[i])
    gbest_fitness = min(fitness)
    sort_fitness, sort_mn, sort_un = zip(*sorted(zip(fitness, mn, un)))  # 按照适应度值由小到大对狼群进行排序
    # 排序完成后原来的列表变成了元组的格式，因此需要往列表中复制一遍
    for i in range(population_size):
        # mn[i] = sort_mn[i].copy()
        # un[i] = sort_un[i].copy()
        fitness[i] = sort_fitness[i]
        UdoT_list[i] = deco.deco(mn[i],un[i])
    print(ini.tar_range)
    while iter < max_iteration:
        # walktime = 0            # 实际游走的次数
        # 游走行为
        # while walktime < Tmax:                                          # 小于最大游走次数
        snum = random.randint(int(population_size / (1 + a)), int(population_size / a))
        for e in range(1,snum + 1):                                 # 遍历所有探狼
            walktime = np.zeros(ini.nu)
            for i in range(ini.nu):                                 # 遍历所有要执行任务的无人机
                if len(UdoT_list[e][i]) > 2 :                        # 只有当无人机执行任务数量>1时，游走任务才有意义
                    record_task = []                                # 用于存储各无人机的任务标号
                    record_order = []                               # 用于存储原先无人机执行的任务在原序列中的位置
                    tempwolf_mn = mn[e].copy()
                    tempwolf_un = un[e].copy()
                    for j in range(ini.nt):
                        if un[e][j] == i:
                            record_task.append(mn[e][j])
                            record_order.append(j)
                    task_number = len(record_task)
                    possible = poss.factorial(task_number)
                    while walktime[i] < possible:
                        random.shuffle(record_task)
                        for t in range(len(record_task)):
                            tempwolf_mn[record_order[t]] = record_task[t]
                        temp_UdoT = deco.deco(tempwolf_mn,tempwolf_un)
                        if limit.limit_verify(temp_UdoT) == 0:
                            tempfit = float('inf')
                        else:
                            tempfit = fit.fit_cal(temp_UdoT)
                        # 比较探索后，与原位置比较是否更好
                        if tempfit < fitness[e]:
                            fitness[e] = tempfit
                            mn[e] = tempwolf_mn.copy()
                            un[e] = tempwolf_un.copy()
                            UdoT_list[e] = temp_UdoT.copy()

                        walktime[i] += 1
        sort_fitness, sort_mn, sort_un = zip(*sorted(zip(fitness, mn, un)))  # 按照适应度值由小到大对狼群进行排序
        # 排序完成后原来的列表变成了元组的格式，因此需要往列表中复制一遍
        for i in range(population_size):
            mn[i] = sort_mn[i].copy()
            un[i] = sort_un[i].copy()
            fitness[i] = sort_fitness[i]
            UdoT_list[i] = deco.deco(mn[i],un[i])

        # 召唤行为
        # for w in range(int(population_size / 2) , population_size):
        for w in range( snum + 1 , population_size):
            # TforU:表示的是0-24个任务是由第x架无人机完成
            leader_TforU = re_deco.re_deco(mn[0],un[0])
            tempwolf_mn = mn[w].copy()
            tempwolf_un = un[w].copy()
            column_number = random.randint(1,9)            # 复制头狼un的列的总数
            for j in range(column_number):
                select_column = random.randint(0,9)         # 具体复制头狼的哪一列
                t_number = tempwolf_mn[select_column]                   # t_number:第i个任务编号
                tempwolf_un[select_column] = leader_TforU[t_number]     # un和mn垂直对应的关系，因此tempwolf_un[j] = leader_TforU内执行t_number任务的无人机编号
            temp_UdoT = deco.deco(tempwolf_mn,tempwolf_un)
            if limit.limit_verify(temp_UdoT) == 0:
                tempfit = float('inf')
            else:
                tempfit = fit.fit_cal(temp_UdoT)
            # 比较召唤行为后的适应度值是否比之前好
            if tempfit < fitness[w]:
                fitness[w] = tempfit
                mn[w] = tempwolf_mn.copy()
                un[w] = tempwolf_un.copy()
                UdoT_list[w] = temp_UdoT.copy()
            # 比较召唤行为后的适应度值是否比头狼好,若好，则交换二者的位置，最后重新排序
            if tempfit < fitness[0]:
                f_exchange = fitness[0]
                mn_exchange = mn[0].copy()
                un_exchange = un[0].copy()
                UdoT_list_exchange = UdoT_list[0].copy()
                fitness[0] = fitness[w]
                mn[0] = mn[w].copy()
                un[0] = un[w].copy()
                UdoT_list[0] = UdoT_list[w].copy()
                fitness[w] = f_exchange
                mn[w] = mn_exchange.copy()
                un[w] = un_exchange.copy()
                UdoT_list[w] = UdoT_list_exchange.copy()
        sort_fitness, sort_mn, sort_un = zip(*sorted(zip(fitness, mn, un)))  # 按照适应度值由小到大对狼群进行排序
        # 排序完成后原来的列表变成了元组的格式，因此需要往列表中复制一遍
        for i in range(population_size):
            mn[i] = sort_mn[i].copy()
            un[i] = sort_un[i].copy()
            fitness[i] = sort_fitness[i]
            UdoT_list[i] = deco.deco(mn[i], un[i])

        # 围攻行为
        for w in range(1,population_size):
            r = random.random()
            # 自适应概率
            p = math.exp((fitness[0] - fitness[w]) / T * 0.3)
            if p > r:
                tempwolf_mn = mn[w].copy()
                tempwolf_un = un[w].copy()
                for i in range(2):
                    selected_column = random.randint(0,ini.nt - 1)
                    tempwolf_un[selected_column] = random.randint(0,ini.nu - 1)
                temp_UdoT = deco.deco(tempwolf_mn,tempwolf_un)
                if limit.limit_verify(temp_UdoT) == 0:
                    tempfit = float('inf')
                else:
                    tempfit = fit.fit_cal(temp_UdoT)
                if tempfit < fitness[w]:
                    fitness[w] = tempfit
                    mn[w] = tempwolf_mn.copy()
                    un[w] = tempwolf_un.copy()
                    UdoT_list[w] = temp_UdoT.copy()
            else:
                rw = random.randint(1,w)
                l = random.randint(0,ini.nt - 2)
                m = random.randint(l,ini.nt - 1)
                tempwolf_mn = mn[rw].copy()
                tempwolf_un = un[rw].copy()
                tempwolf_un[l] = random.randint(0,ini.nu - 1)
                tempwolf_un[m] = random.randint(0,ini.nu - 1)
                temp_UdoT = deco.deco(tempwolf_mn,tempwolf_un)
                if limit.limit_verify(temp_UdoT) == 0:
                    tempfit = float('inf')
                else:
                    tempfit = fit.fit_cal(temp_UdoT)
                if tempfit < fitness[w]:
                    fitness[w] = tempfit
                    mn[w] = tempwolf_mn.copy()
                    un[w] = tempwolf_un.copy()
                    UdoT_list[w] = temp_UdoT.copy()
        sort_fitness, sort_mn, sort_un = zip(*sorted(zip(fitness, mn, un)))  # 按照适应度值由小到大对狼群进行排序
        # 排序完成后原来的列表变成了元组的格式，因此需要往列表中复制一遍
        for i in range(population_size):
            mn[i] = sort_mn[i].copy()
            un[i] = sort_un[i].copy()
            fitness[i] = sort_fitness[i]
            UdoT_list[i] = deco.deco(mn[i], un[i])

        # 新个体生成行为
        R = random.randint(int(population_size / (2 * b)), int(population_size / b))
        # 构建经验池
        exper_pool = [[0] * ini.nu for _ in range(ini.nt)]
        for i in range(int(population_size * 0.2)):
            for j in range(ini.nt):
                task_location = mn[i].index(j)
                uav_id = un[i][task_location]
                exper_pool[j][uav_id] += 1
        # 个体重构
        for w in range(population_size - R, population_size):
            tempwolf_mn = mn[w].copy()
            tempwolf_un = un[w].copy()
            for j in range(ini.nt):
                max_cnt = max(exper_pool[j])
                best_task = exper_pool[j].index(max_cnt)
                location_un = tempwolf_mn.index(j)
                tempwolf_un[location_un] = best_task
            temp_UdoT = deco.deco(tempwolf_mn, tempwolf_un)
            if limit.limit_verify(temp_UdoT) == 0:
                tempfit = float('inf')
            else:
                tempfit = fit.fit_cal(temp_UdoT)
            if tempfit < fitness[w]:
                fitness[w] = tempfit
                mn[w] = tempwolf_mn.copy()
                un[w] = tempwolf_un.copy()
                UdoT_list[w] = temp_UdoT.copy()


        # # 构建经验池
        # for w in range(population_size - R, population_size):
        #     for i in range(ini.nt):
        #         tempwolf_mn = mn[w].copy()
        #         tempwolf_un = un[w].copy()
        #         task_cnt = [0] * ini.nu
        #         for j in range(int(population_size * 0.2)):
        #             task_location = mn[j].index(i)
        #             uav_id = un[j][task_location]
        #             task_cnt[uav_id] += 1
        #         max_cnt = max(task_cnt)
        #         best_task = task_cnt.index(max_cnt)
        #         original_task_location = tempwolf_mn.index(i)
        #         tempwolf_un[original_task_location] = best_task
        #         # original_task_location = mn[w].index(i)
        #         # un[w][original_task_location] = best_task
        #     #加UdoT_list,然后比较适应度，好则保留，坏则不接受
        #     temp_UdoT = deco.deco(tempwolf_mn,tempwolf_un)
        #     if limit.limit_verify(temp_UdoT) == 0:
        #         tempfit = float('inf')
        #     else:
        #         tempfit = fit.fit_cal(temp_UdoT)
        #     if tempfit < fitness[w]:
        #         fitness[w] = tempfit
        #         mn[w] = tempwolf_mn.copy()
        #         un[w] = tempwolf_un.copy()
        #         UdoT_list[w] = temp_UdoT.copy()

        # for w in range(population_size - R,population_size):
        #     r = random.random()
        #     tempwolf_mn = mn[0].copy()
        #     tempwolf_un = un[0].copy()
        #     if r < 0.5:
        #         l = random.randint(0,ini.nt - 2)
        #         m = random.randint(l,ini.nt - 1)
        #         exchange = tempwolf_un[l]
        #         tempwolf_un[l] = tempwolf_un[m]
        #         tempwolf_un[m] = exchange
        #     else:
        #         l = random.randint(0,ini.nt - 2)
        #         m = random.randint(l,ini.nt - 1)
        #         tempwolf_un[l] = random.randint(0,ini.nu - 1)
        #         tempwolf_un[m] = random.randint(0,ini.nu - 1)
        #     temp_UdoT = deco.deco(tempwolf_mn,tempwolf_un)
        #     if limit.limit_verify(temp_UdoT) == 0:
        #         tempfit = float('inf')
        #     else:
        #         tempfit = fit.fit_cal(temp_UdoT)
        #     if tempfit < fitness[w]:
        #         fitness[w] = tempfit
        #         mn[w] = tempwolf_mn.copy()
        #         un[w] = tempwolf_un.copy()
        #         UdoT_list[w] = temp_UdoT.copy()
        sort_fitness, sort_mn, sort_un = zip(*sorted(zip(fitness, mn, un)))  # 按照适应度值由小到大对狼群进行排序
        # 排序完成后原来的列表变成了元组的格式，因此需要往列表中复制一遍
        for i in range(population_size):
            mn[i] = sort_mn[i].copy()
            un[i] = sort_un[i].copy()
            fitness[i] = sort_fitness[i]
            UdoT_list[i] = deco.deco(mn[i], un[i])
        gbest_fitness = fitness[0]
        fit_record[iter] = gbest_fitness
        plan_record[iter] = deco.deco(mn[0],un[0])
        iter += 1
        # 自适应温度参数
        T = T - 0.125
    min_best_fit = min(fit_record)
    min_index = fit_record.index(min_best_fit)
    min_best_plan = plan_record[min_index]
    task_complete_time,average_time = fit.timecost(min_best_plan)
    # end_time = time.time()
    #print(f'程序运行时间为:{end_time - start_time}')
    return min_best_plan,task_complete_time,average_time,min_best_fit,fit_record


# print(min_best_plan)
# print(min_best_fit)
# def deco(mn,un):
#     UdoT_list = [[] for k in range(ini.nu)]
#     for i in range(len(un)):
#         UdoT_list[un[i]].append(i)


if __name__ == '__main__':

    print("开始单次运行算法...")
    start_time = time.time()
    plan, total_time, avg_time, best_fit, fit_history = mdwpa()

    print("\n--- 单次运行结果 ---")
    print(f"最优适应度: {best_fit}")
    print(f"总完成时间: {total_time}")
    end_time = time.time()
    comsued_time = (end_time - start_time) 