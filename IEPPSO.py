import numpy as np
import initial as ini
import random, time
import limit, fit

def ieppso(u_loc, t_loc, t_range, t_window):
    # 算法初始化
    population_size = 200   # 种群数量
    dim = ini.nt   # 解空间维度 = 目标数目
    max_iteration = ini.max_iteration   # 最大迭代次数，迭代次数达到最大限制，算法退出迭代
    max_w = 0.9    # 惯性权重范围，后期根据该范围利用自适应调整
    min_w = 0.4
    max_c1 = 2.5   # 个体、群体学习因子，设置最大最小值可以后期自适应调整
    min_c1 = 0.5
    max_c2 = 2.0
    min_c2 = 0.5
    exper_size = 40   # 经验池大小
    pv = 0.2  # 粒子初始化阶段约束验证概率
    max_unchange = 40  # 当连续max次迭代群体最佳适应度值都没有改变时则对经验池执行突变操作
    # 初始化粒子群粒子位置  解空间范围，对应无人机数目
    x = np.random.uniform(0, ini.nu, (population_size, dim))
    # 初始化粒子群速度
    v = np.random.rand(population_size, dim)
    fitness = [0.0 for k in range(population_size)]     # 适应度值初始化
    UdoT_list = [[] for k in range(population_size)]    # 存储粒子对应解
    time1 = time.time()
    for i in range(population_size):
        UdoT_list[i] = deco(x[i])
        if random.random() <= pv:
            reini = 0
            while limit.limit_verify(UdoT_list[i], u_loc, t_loc, t_range, t_window) == 0 and reini < 50:
                x[i] = np.random.uniform(0, ini.nu, ini.nt)
                UdoT_list[i] = deco(x[i])
                reini += 1
        fitness[i] = fit.fit_cal(UdoT_list[i], u_loc, t_loc, t_range, t_window)
    ini_time = time.time() - time1
    p = x.copy()                          # 初始化个体最优粒子
    pg = x[np.argmin(fitness)].copy()     # 初始化种群最优粒子
    pbest_fitness = fitness.copy()        # 初始化个体最优适应度值
    gbest_fitness = min(fitness)          # 初始化种群最优适应度值
    best_fitness_mat = [gbest_fitness]                 # 记录历史最优适应度值
    experiencex = np.random.uniform(0, 0, (exper_size, dim))  # 经验池
    exp_fitness = [0.0 for k in range(exper_size)]  # 记录经验池内粒子适应度值
    for j in range(exper_size):
        experiencex[j] = x[j].copy()  # 初始化经验池
        exp_fitness[j] = fitness[j]
    iter = 0                  # 初始化迭代次数
    unchange_ct = 0   # 用来统计群体最佳适应度值没有改变的次数

    while iter < max_iteration:
        # 区间 [0,1] 内的随机数，增加搜索的随机性
        r1 = np.random.rand(population_size, dim)
        r2 = np.random.rand(population_size, dim)
        w = max_w - (max_w - min_w) * iter / max_iteration  # 自适应调整惯性权重
        c1 = max_c1 - (max_c1 - min_c1) * iter / max_iteration  # 自适应调整个体学习因子
        c2 = min_c2 + (max_c2 - min_c2) * iter / max_iteration  # 自适应调整群体学习因子
        v = w * v.copy() + c1 * r1 * (p.copy() - x.copy()) + c2 * r2 * (pg.copy() - x.copy())  # 更新速度
        v = np.clip(v, -2 + 1 * iter / max_iteration,
                    2 - 1 * iter / max_iteration)  # 自适应调整步长
        x = v.copy() + x.copy()    # 更新位置
        x = np.clip(x, 0, ini.nu - 0.0001)
        fitness = [0.0 for k in range(population_size)]
        UdoT_list = [[] for k in range(population_size)]
        # 基于经验池，对不满足约束粒子重新生成
        for i in range(population_size):
            UdoT_list[i] = deco(x[i])
            fitness[i] = fit.fit_cal(UdoT_list[i], u_loc, t_loc, t_range, t_window)
            pro = 0.1  # 突变概率
            # 不满足约束粒子根据经验池重新生成
            while limit.limit_verify(UdoT_list[i], u_loc, t_loc, t_range, t_window) == 0:
                for d in range(dim):
                    m1 = random.randint(0, exper_size - 2)
                    m2 = random.randint(m1 + 1, exper_size - 1)
                    if exp_fitness[m1] <= exp_fitness[m2]:
                        x[i][d] = experiencex[m1][d]
                    else:
                        x[i][d] = experiencex[m2][d]
                    if pro > random.random():
                        x[i][d] = random.uniform(0, ini.nu)
                UdoT_list[i] = deco(x[i])
                fitness[i] = fit.fit_cal(UdoT_list[i], u_loc, t_loc, t_range, t_window)
            if pbest_fitness[i] > fitness[i]:
                pbest_fitness[i] = fitness[i]
                p[i] = x[i].copy()
            # 更新经验池
            for k in range(exper_size):
                if exp_fitness[k] > pbest_fitness[i]:
                    experiencex[k] = p[i].copy()
                    exp_fitness[k] = pbest_fitness[i]
                    break
        if min(pbest_fitness) < gbest_fitness:
            best_p_index = pbest_fitness.index(min(pbest_fitness))     # 记录种群最优粒子对应的编号
            pg = p[best_p_index].copy()            # 更新种群最优粒子
            gbest_fitness = min(pbest_fitness)     # 更新种群最优粒子适应度值
        else:
            unchange_ct += 1
        best_fitness_mat.append(gbest_fitness)
        if unchange_ct >= max_unchange:    # 当连续max次群体最佳适应度值没有发生变化时则对经验池进行更新
            pm1 = 0.2    # 突变概率
            pm2 = 0.3    # 学习概率
            pt = 50      # 每个粒子更新次数
            for t in range(exper_size):
                # rex = random.randint(0, exper_size - 1)
                tempx = experiencex[t].copy()
                for n in range(pt):
                    for l in range(dim):
                        r = random.random()
                        if r <= pm1:
                            # tempx[l] = tempx[l] + (max_iteration - iter) / max_iteration * random.uniform(-4, 4)
                            tempx[l] = random.uniform(0, ini.nu)
                            tempx = np.clip(tempx, 0, ini.nu - 0.0001)
                        elif pm1 < r <= pm2:
                            tempx[l] = pg[l]
                            tempx = np.clip(tempx, 0, ini.nu - 0.0001)
                    UdoT = deco(tempx)
                    fit_temp = fit.fit_cal(UdoT, u_loc, t_loc, t_range, t_window)
                    if fit_temp <= exp_fitness[t]:
                        exp_fitness[t] = fit_temp
                        experiencex[t] = tempx.copy()
                        break
            min_exfit = min(exp_fitness)
            min_exfit_index = exp_fitness.index(min(exp_fitness))
            if min_exfit < gbest_fitness:
                best_fitness_mat[-1] = min_exfit
                pg = experiencex[min_exfit_index].copy()
            unchange_ct = 0
        iter = iter + 1
    print('IEPPSO算法完成迭代')
    # plan = deco(pg)      # 最终分配方案
    # task_complete_time, average_time = fit.timecost(plan, u_loc, t_loc, t_range)      # 记录任务完成时间，无人机平均飞行时间
    return gbest_fitness, best_fitness_mat

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