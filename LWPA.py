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

class lwpa():
    def __init__(self, population_size=100, max_iterations=800, a=4, b=7, q=2, Tmax=10):
        self.population_size = population_size   # 种群规模
        self.max_iterations = max_iterations     # 最大迭代次数
        self.a = a                      # 探狼比例因子，决定每轮迭代探狼数量
        self.b = b                      # 比例因子，每轮迭代淘汰数量
        self.q = q
        self.Tmax = Tmax
        self.nt = ini.nt
        self.nu = ini.nu
        self.fitness = np.zeros(self.population_size) 
        self.fit_record = [[] for _ in range(max_iterations)]
        self.plan_record = [[0.0 for i in range(ini.nu)] for j in range(max_iterations)]  # 记录每次迭代的任务分配情况
        self.iter = 0
        self.mn = [[] for k in range(population_size)]
        self.un = [[] for k in range(population_size)]
        self.mn_opl = [[] for k in range(population_size)]
        self.un_opl = [[] for k in range(population_size)]
        self.UdoT_list = [[] for k in range(population_size)]
        self.UdoT_list_opl = [[] for k in range(population_size)]
    
    def opposite_learning(self, solution, lower, upper):
        opposite_solution = [lower + upper -x for x in solution]
        return opposite_solution  # 计算反向解

    def _generate_levy_step(self, beta=1.5 ):
        """生成一个莱ви步长并映射到离散的变异步数"""
        # 计算 sigma
        sigma_num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        sigma_den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma = (sigma_num / sigma_den) ** (1 / beta)
    
        # 生成 u 和 v
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
    
        # 计算莱维步长
        levy_step = u / (abs(v) ** (1 / beta))
    
        # 将连续步长映射为离散的变异步数
        # 保证至少有1次变异，并且不会过多，这里限制在 nt/4 以内
        num_mutations = int(1 + abs(levy_step) % (self.nt // 4))
        return num_mutations
    def update(self):
        #初始化种群
        for i in range(self.population_size):
            self.mn[i] = random.sample(range(0, ini.nt), ini.nt)
            self.un[i] = [random.randint(0, ini.nu - 1) for _ in range(ini.nt)]
            self.UdoT_list[i] = deco.deco(self.mn[i], self.un[i])
            #反向学习策略
            self.mn_opl[i] = self.opposite_learning(self.mn[i], 0, self.nt -1)
            self.un_opl[i] = self.opposite_learning(self.un[i], 0, self.nu -1)
            self.UdoT_list_opl[i] = deco.deco(self.mn_opl[i], self.un_opl[i])
        
        # 计算适应度值
            if limit.limit_verify(self.UdoT_list[i]) == 0 and limit.limit_verify(self.UdoT_list_opl[i]) == 0:
                self.fitness[i] = float("inf")
            else:
                if fit.fit_cal(self.UdoT_list[i]) < fit.fit_cal(self.UdoT_list_opl[i]):
                    self.fitness[i] = fit.fit_cal(self.UdoT_list[i])
                else:
                    self.mn[i] = self.mn_opl[i]
                    self.un[i] = self.un_opl[i]
                    self.fitness[i] = fit.fit_cal(self.UdoT_list_opl[i])
                
        
        self.gbest_fitness = min(self.fitness)
        self.sort_population()

    def judge_repeat(self, list):
        repeat_list = []
        for num in list:
            if num in repeat_list:
                return 0
            repeat_list.append(num)
        return 1
    

    def search_repeat(self, list_, column_id):
        """查找重复元素索引"""
        repeat_list = []
        for index, num in enumerate(list_):
            if num in repeat_list:
                repeat_index = index
                exist_index = repeat_list.index(num)
                return exist_index if repeat_index == column_id else repeat_index
            else:
                repeat_list.append(num)
        return -1
    

    def sort_population(self):
        """根据适应度对狼群排序"""
        sort_fitness, sort_mn, sort_un = zip(*sorted(zip(self.fitness, self.mn, self.un)))  # 按照适应度值由小到大对狼群进行排序
        # 排序完成后原来的列表变成了元组的格式，因此需要往列表中复制一遍
        for i in range(self.population_size):
            self.mn[i] = sort_mn[i].copy()
            self.un[i] = sort_un[i].copy()
            self.fitness[i] = sort_fitness[i]
            self.UdoT_list[i] = deco.deco(self.mn[i], self.un[i])
        # sorted_data = sorted(zip(self.fitness, self.mn, self.un))
        # self.fitness, self.mn, self.un = zip(*sorted_data)
        # self.fitness = list(self.fitness)
        # self.mn = list(self.mn)
        # self.un = list(self.un)
        # self.UdoT_list = [deco.deco(mn, un) for mn, un in zip(self.mn, self.un)]
    

    def wandering(self):
        """游走行为"""
        do_yes = 1
        # --- 改进点2：自适应探狼比例 ---
        # 探狼比例从 a 线性/非线性地增加到 a+1，使得早期探索性更强
        # 这里我们用一个简单的非线性递减函数
        # ratio_decay = (1 - (self.iter / self.max_iterations))**2
        # current_a = self.a + (1 - ratio_decay)  # a 从 a -> a+1
        # self.snum = random.randint(int(self.population_size / (current_a + 1)), int(self.population_size / current_a))
        # --- 结束改进点2 ---
        for i in range(self.Tmax):
            self.snum = random.randint(int(self.population_size / (self.a + 1)), int(self.population_size / self.a))
            for j in range(1, self.snum + 1):
                temp_un = self.un[j].copy()
                temp_mn = self.mn[j].copy()

                # --- 改进点1：使用莱维飞行确定变异步数 ---
                num_mutations = self._generate_levy_step()
                for _ in range(num_mutations):
                    # 随机选择两种变异操作：UAV分配变异 或 任务顺序变异
                    if random.random() < 0.5:
                        # UAV分配变异
                        col = random.randint(0, self.nt - 1)
                        temp_un[col] = random.randint(0, self.nu - 1)
                    else:
                        # 任务顺序交换变异 (Swap)
                        col1, col2 = random.sample(range(self.nt), 2)
                        temp_mn[col1], temp_mn[col2] = temp_mn[col2], temp_mn[col1]
                # --- 结束改进点1 ---
                
                
                # 计算新适应度
                temp_UdoT = deco.deco(temp_mn, temp_un)
                #反向学习
                # temp_mn_opl = self.opposite_learning(temp_mn, 0, 24)
                # temp_un_opl = self.opposite_learning(temp_un, 0, 9)
                # temp_UdoT_opl = deco.deco(temp_mn_opl, temp_un_opl)
                # if limit.limit_verify(temp_UdoT_opl) == 0 and limit.limit_verify(temp_UdoT) == 0:
                #     temp_fit = float('inf')
                # else:
                #     if fit.fit_cal(temp_UdoT) < fit.fit_cal(temp_UdoT_opl):
                #         temp_fit = fit.fit_cal(temp_UdoT)
                #     else:
                #         temp_mn = temp_mn_opl
                #         temp_un = temp_un_opl
                #         temp_fit = fit.fit_cal(temp_UdoT_opl)
                temp_fit = float('inf') if limit.limit_verify(temp_UdoT) == 0 else fit.fit_cal(temp_UdoT)
                
                if temp_fit < self.fitness[j]:
                    self.fitness[j] = temp_fit
                    self.mn[j], self.un[j], self.UdoT_list[j] = temp_mn.copy(), temp_un.copy(), temp_UdoT
                
                if temp_fit < self.fitness[0]:  # 更新领导者
                    self.swap_leader(j, temp_fit, temp_mn, temp_un, temp_UdoT)
                    do_yes = 0
                    #break
            # if  do_yes ==0:
            #     break
        self.sort_population()


    def swap_leader(self, follower_idx, temp_fit, new_mn, new_un, new_UdoT):
        """交换领导者与跟随者的位置"""
        exchange_fitness = self.fitness[0]
        exchange_mn = self.mn[0].copy()
        exchange_un = self.un[0].copy()
        exchange_ul = self.UdoT_list[0].copy()
        self.fitness[0] = temp_fit
        self.mn[0] = new_mn
        self.un[0] = new_un
        self.UdoT_list[0] = new_UdoT
        self.fitness[follower_idx] = exchange_fitness
        self.mn[follower_idx] = exchange_mn
        self.un[follower_idx] = exchange_un
        self.UdoT_list[follower_idx] = exchange_ul
        # self.fitness[leader_idx], self.fitness[follower_idx] = new_fit, self.fitness[leader_idx]
        # self.mn[leader_idx], self.mn[follower_idx] = new_mn, self.mn[leader_idx]
        # self.un[leader_idx], self.un[follower_idx] = new_un, self.un[leader_idx]
        # self.UdoT_list[leader_idx], self.UdoT_list[follower_idx] = new_UdoT, self.UdoT_list[leader_idx]


    def call_followers(self):
        """召唤行为"""
        
        #leader_mn, leader_un = self.mn[0].copy(), self.un[0].copy()
        self.step_b = random.randint(1, self.nt - 1)
        for i in range( self.snum + 1, self.population_size):
            temp_mn, temp_un = self.mn[i].copy(), self.un[i].copy()
            leader_mn, leader_un = self.mn[0].copy(), self.un[0].copy()
            for j in range(self.step_b):
                col = random.randint(0, self.nt - 1)
                middle_mn = temp_mn[col]
                temp_mn[col], temp_un[col] = leader_mn[col], leader_un[col]
                if self.search_repeat(temp_mn,col) != -1:
                    repeat_index = self.search_repeat(temp_mn,col)
                    temp_mn[repeat_index] = middle_mn
            
            temp_UdoT = deco.deco(temp_mn, temp_un)
            #反向学习
            # temp_mn_opl = self.opposite_learning(temp_mn, 0, 24)
            # temp_un_opl = self.opposite_learning(temp_un, 0, 9)
            # temp_UdoT_opl = deco.deco(temp_mn_opl, temp_un_opl)
            # if limit.limit_verify(temp_UdoT_opl) == 0 and limit.limit_verify(temp_UdoT) == 0:
            #     temp_fit = float('inf')
            # else:
            #     if fit.fit_cal(temp_UdoT) < fit.fit_cal(temp_UdoT_opl):
            #         temp_fit = fit.fit_cal(temp_UdoT)
            #     else:
            #         temp_mn = temp_mn_opl
            #         temp_un = temp_un_opl
            #         temp_fit = fit.fit_cal(temp_UdoT_opl)
            temp_fit = float('inf') if limit.limit_verify(temp_UdoT) == 0 else fit.fit_cal(temp_UdoT)
            
            if temp_fit < self.fitness[i]:
                self.fitness[i], self.mn[i], self.un[i], self.UdoT_list[i] = temp_fit, temp_mn, temp_un, temp_UdoT
            
            if temp_fit < self.fitness[0]:
                self.swap_leader(i, temp_fit, temp_mn, temp_un, temp_UdoT)
        
        self.sort_population()


    def siege(self):
        """围攻行为"""
        step_c = int(self.step_b / 4)

        
        for i in range(1, self.population_size):
            leader_mn, leader_un = self.mn[0].copy(), self.un[0].copy()

            for j in range(step_c):
                temp_mn, temp_un = self.mn[i].copy(), self.un[i].copy()
                col = random.randint(0, self.nt - 1)
                middle_mn = temp_mn[col]
                temp_mn[col], temp_un[col] = leader_mn[col], leader_un[col]
                if self.search_repeat(temp_mn,col) != -1:
                    repeat_idx = self.search_repeat(temp_mn, col)
                    temp_mn[repeat_idx] = middle_mn
            
                temp_UdoT = deco.deco(temp_mn, temp_un)
                #反向学习
                # temp_mn_opl = self.opposite_learning(temp_mn, 0, 24)
                # temp_un_opl = self.opposite_learning(temp_un, 0, 9)
                # temp_UdoT_opl = deco.deco(temp_mn_opl, temp_un_opl)
                # if limit.limit_verify(temp_UdoT_opl) == 0 and limit.limit_verify(temp_UdoT) == 0:
                #     temp_fit = float('inf')
                # else:
                #     if fit.fit_cal(temp_UdoT) < fit.fit_cal(temp_UdoT_opl):
                #         temp_fit = fit.fit_cal(temp_UdoT)
                #     else:
                #         temp_mn = temp_mn_opl
                #         temp_un = temp_un_opl
                #         temp_fit = fit.fit_cal(temp_UdoT_opl)
                temp_fit = float('inf') if limit.limit_verify(temp_UdoT) == 0 else fit.fit_cal(temp_UdoT)
            
                if temp_fit < self.fitness[i]:
                    self.fitness[i], self.mn[i], self.un[i], self.UdoT_list[i] = temp_fit, temp_mn, temp_un, temp_UdoT
        
        self.sort_population()
    

    def live(self):
        """个体重新生成"""
        R = random.randint(int(self.population_size / (self.b + 1)), int(self.population_size / self.b))
        for i in range(self.population_size - R, self.population_size):
            self.mn[i] = random.sample(range(0, self.nt), self.nt)
            self.un[i] = [random.randint(0, self.nu - 1) for _ in range(self.nt)]
            self.UdoT_list[i] = deco.deco(self.mn[i], self.un[i])
            #反向学习
            # temp_mn_opl = self.opposite_learning(self.mn[i], 0, 24)
            # temp_un_opl = self.opposite_learning(self.un[i], 0, 9)
            # temp_UdoT_opl = deco.deco(temp_mn_opl, temp_un_opl)
            # if limit.limit_verify(temp_UdoT_opl) == 0 and limit.limit_verify(self.UdoT_list[i]) == 0:
            #     self.fitness[i] = float('inf')
            # else:
            #     if fit.fit_cal(self.UdoT_list[i]) < fit.fit_cal(temp_UdoT_opl):
            #         self.fitness[i] = fit.fit_cal(self.UdoT_list[i])
            #     else:
            #         self.mn[i] = temp_mn_opl
            #         self.un[i] = temp_un_opl
            #         self.fitness[i] = fit.fit_cal(temp_UdoT_opl)
            self.fitness[i] = float("inf") if limit.limit_verify(self.UdoT_list[i]) == 0 else fit.fit_cal(self.UdoT_list[i])


    def run(self):
        """运行狼群算法"""
        start_time = time.time()
        self.update()
        iter_count = 0
        while iter_count < self.max_iterations:
            self.wandering()
            self.call_followers()
            self.siege()
            self.live()
            
            self.gbest_fitness = self.fitness[0]
            self.fit_record[iter_count] = self.gbest_fitness
            self.plan_record[iter_count] = deco.deco(self.mn[0], self.un[0])

            # 增加进度打印
            if (iter_count + 1) % 50 == 0:
                 print(f"迭代 {iter_count + 1}/{self.max_iterations}, 当前最优适应度: {self.fitness[0]}")
            iter_count += 1

            
        end_time = time.time()
        print(f"\n算法运行结束，耗时 {end_time - start_time:.2f} 秒。")


        min_best_fit = min(self.fit_record)
        min_index = self.fit_record.index(min_best_fit)
        min_best_plan = self.plan_record[min_index]
        task_complete_time, average_time = fit.timecost(min_best_plan)
        
        print(min_best_plan)
        return min_best_plan, task_complete_time, average_time, min_best_fit, self.fit_record