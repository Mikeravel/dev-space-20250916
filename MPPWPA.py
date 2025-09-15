import math
import fit
import initial as ini
import numpy as np
import random
import deco
import limit
import random
import re_deco
import possible
import time

def search_repeat(list,column_id):
    repeat_list = []
    for index,num in enumerate(list):
        if num in repeat_list:
            repeat_index = index
            exist_index = repeat_list.index(num)    # 查找到的索引值不对
            if repeat_index == column_id:
                return exist_index
            else:
                return repeat_index
        else:
            repeat_list.append(num)
    return -1
def judge_repeat(list):
    repeat_list = []
    for num in list:
        if num in repeat_list:
            return 0
        repeat_list.append(num)
    return 1
def mppwpa():
    population_size = 100   # 狼群数量
    max_iteration = 800     # 最大迭代次数
    a = 4                   # 探狼比例因子。决定每轮迭代中探狼的数量
    b = 7                   # 更新比例因子，每轮迭代淘汰/重新生成狼群的数量
    q = 2                   # 围捕因子，决定随机选择的位置
    Tmax = 10                # 每轮迭代中的随机游走次数
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
        UdoT_list[i] = deco.deco(mn[i], un[i])
    while iter < max_iteration:
        # 游走行为
        do_yes = 1
        for i in range(Tmax):
            snum = random.randint(int(population_size / (a + 1)), int(population_size / a))     # 随机生成探狼个数
            for j in range(1,snum + 1):
                temp_un = un[j].copy()
                temp_mn = mn[j].copy()
                selected_column = random.randint(0,ini.nt - 1)              # 挑选变异的un列
                exchanged_column = random.randint(0,ini.nt - 1)             # 挑选与之交换的mn列
                temp_un[selected_column] = random.randint(0,ini.nu - 1)
                mid_change = temp_mn[selected_column]                   # 交换的中间变量
                temp_mn[selected_column] = temp_mn[exchanged_column]
                temp_mn[exchanged_column] = mid_change
                temp_UdoT = deco.deco(temp_mn,temp_un)
                if limit.limit_verify(temp_UdoT) == 0:
                    temp_fit = float('inf')
                else:
                    temp_fit = fit.fit_cal(temp_UdoT)
                if temp_fit < fitness[j]:
                    fitness[j] = temp_fit
                    mn[j] = temp_mn.copy()
                    un[j] = temp_un.copy()
                    UdoT_list[j] = temp_UdoT
                if temp_fit < fitness[0]:
                    exchange_fitness = fitness[0]
                    exchange_mn = mn[0].copy()
                    exchange_un = un[0].copy()
                    exchange_ul = UdoT_list[0].copy()
                    fitness[0] = temp_fit
                    mn[0] = mn[j]
                    un[0] = un[j]
                    UdoT_list[0] = UdoT_list[j]
                    fitness[j] = exchange_fitness
                    mn[j] = exchange_mn
                    un[j] = exchange_un
                    UdoT_list[j] = exchange_ul
                    do_yes = 0
                    break
            if do_yes == 0:
                break
        sort_fitness, sort_mn, sort_un = zip(*sorted(zip(fitness, mn, un)))  # 按照适应度值由小到大对狼群进行排序
        # 排序完成后原来的列表变成了元组的格式，因此需要往列表中复制一遍
        for i in range(population_size):
            mn[i] = sort_mn[i].copy()
            un[i] = sort_un[i].copy()
            fitness[i] = sort_fitness[i]
            UdoT_list[i] = deco.deco(mn[i], un[i])
        # 召唤行为
        step_b = random.randint(1, ini.nt - 1)  # 随机生成要变化列的个数
        for i in range(snum + 1 ,population_size):
            temp_mn = mn[i].copy()
            temp_un = un[i].copy()
            leader_mn = mn[0].copy()
            leader_un = un[0].copy()
            for j in range(step_b):
                selected_column1 = random.randint(0,ini.nt - 1)                # 随机生成变化的列
                middle_mn = temp_mn[selected_column1]
                temp_mn[selected_column1] = leader_mn[selected_column1]
                temp_un[selected_column1] = leader_un[selected_column1]
                if search_repeat(temp_mn,selected_column1) != -1:
                    repeat_index = search_repeat(temp_mn,selected_column1)
                    temp_mn[repeat_index] = middle_mn
            temp_UdoT = deco.deco(temp_mn,temp_un)
            if limit.limit_verify(temp_UdoT) == 0:
                temp_fit = float('inf')
            else:
                temp_fit = fit.fit_cal(temp_UdoT)
            if temp_fit < fitness[i]:
                fitness[i] = temp_fit
                mn[i] = temp_mn.copy()
                un[i] = temp_un.copy()
                UdoT_list[i] = temp_UdoT.copy()
            if temp_fit < fitness[0]:
                f_exchange = fitness[0]
                mn_exchange = mn[0].copy()
                un_exchange = un[0].copy()
                UdoT_list_exchange = UdoT_list[0].copy()
                fitness[0] = fitness[i]
                mn[0] = mn[i].copy()
                un[0] = un[i].copy()
                UdoT_list[0] = UdoT_list[i].copy()
                fitness[i] = f_exchange
                mn[i] = mn_exchange.copy()
                un[i] = un_exchange.copy()
                UdoT_list[i] = UdoT_list_exchange.copy()
        sort_fitness, sort_mn, sort_un = zip(*sorted(zip(fitness, mn, un)))  # 按照适应度值由小到大对狼群进行排序
        # 排序完成后原来的列表变成了元组的格式，因此需要往列表中复制一遍
        for i in range(population_size):
            # mn[i] = sort_mn[i].copy()
            # un[i] = sort_un[i].copy()
            fitness[i] = sort_fitness[i]
            UdoT_list[i] = deco.deco(mn[i], un[i])

        # 围攻行为
        step_c = int(step_b / 4)
        for i in range(1,population_size):
            leader_mn = mn[0].copy()
            leader_un = un[0].copy()
            for j in range(step_c):
                temp_mn = mn[i].copy()
                temp_un = un[i].copy()
                selected_column2 = random.randint(0,ini.nt - 1)
                middle_mn = temp_mn[selected_column2]
                temp_mn[selected_column2] = leader_mn[selected_column2]
                temp_un[selected_column2] = leader_un[selected_column2]
                if search_repeat(temp_mn,selected_column2) != -1:
                    repeat_index = search_repeat(temp_mn, selected_column2)
                    temp_mn[repeat_index] = middle_mn
                temp_UdoT = deco.deco(temp_mn, temp_un)
                if limit.limit_verify(temp_UdoT) == 0:
                    temp_fit = float('inf')
                else:
                    temp_fit = fit.fit_cal(temp_UdoT)
                if temp_fit < fitness[i]:
                    fitness[i] = temp_fit
                    mn[i] = temp_mn.copy()
                    un[i] = temp_un.copy()
                    UdoT_list[i] = temp_UdoT.copy()
        sort_fitness, sort_mn, sort_un = zip(*sorted(zip(fitness, mn, un)))  # 按照适应度值由小到大对狼群进行排序
        # 排序完成后原来的列表变成了元组的格式，因此需要往列表中复制一遍
        for i in range(population_size):
            mn[i] = sort_mn[i].copy()
            un[i] = sort_un[i].copy()
            fitness[i] = sort_fitness[i]
            UdoT_list[i] = deco.deco(mn[i], un[i])

        #个体重新生成
        R = random.randint(int(population_size / (b + 1)) , int(population_size / b))
        for i in range(population_size - R , population_size):
            mn[i] = random.sample(range(0, ini.nt), ini.nt)
            un[i] = [random.randint(0, ini.nu - 1) for i in range(ini.nt)]
            UdoT_list[i] = deco.deco(mn[i], un[i])
            if limit.limit_verify(UdoT_list[i]) == 0:
                fitness[i] = float("inf")
            else:
                fitness[i] = fit.fit_cal(UdoT_list[i])
        gbest_fitness = fitness[0]
        fit_record[iter] = gbest_fitness
        plan_record[iter] = deco.deco(mn[0], un[0])
        iter += 1
    min_best_fit = min(fit_record)
    min_index = fit_record.index(min_best_fit)
    min_best_plan = plan_record[min_index]
    task_complete_time, average_time = fit.timecost(min_best_plan)
    #end_time = time.time()
    print(min_best_plan)
    return min_best_plan, task_complete_time, average_time, min_best_fit, fit_record



