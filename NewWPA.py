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

class nlwpa():
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
        self.UdoT_list = [[] for k in range(population_size)]

    def opposite_learning(self, solution, lower, upper):
        # 计算反向解
        opposite_solution = [lower + upper - x for x in solution]
        return opposite_solution

    def _generate_levy_step(self, beta=1.5 ):
        """生成一个莱维步长并映射到离散的变异步数"""
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

    def _generate_levy_crossover_step(self, beta=1.5):
        """【新增】生成一个用于交叉/学习的莱维步长"""
        # 计算 sigma
        sigma_num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        sigma_den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma = (sigma_num / sigma_den) ** (1 / beta)
    
        # 生成 u 和 v
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
    
        # 计算莱维步长
        levy_step = u / (abs(v) ** (1 / beta))

        # 将连续步长映射为离散的交叉点数
        # 步长与任务数量相关，并限制在合理范围内
        step_size = int(1 + abs(levy_step) * self.nt * 0.1) # 0.1为启发式缩放因子
        return max(1, min(step_size, self.nt - 1))  # 确保步长在 [1, nt-1] 之间

    def _apply_obl_and_get_fitness(self, mn_sol, un_sol):
        """【新增】对给定的解应用反向学习策略，并返回更优的解及其适应度值"""
        # 计算原始解的适应度
        udot_original = deco.deco(mn_sol, un_sol)
        fit_original = float('inf') if not limit.limit_verify(udot_original) else fit.fit_cal(udot_original)

        # 计算反向解的适应度 (修正了硬编码的边界)
        mn_opposite = self.opposite_learning(mn_sol, 0, self.nt - 1)
        un_opposite = self.opposite_learning(un_sol, 0, self.nu - 1)
        udot_opposite = deco.deco(mn_opposite, un_opposite)
        fit_opposite = float('inf') if not limit.limit_verify(udot_opposite) else fit.fit_cal(udot_opposite)

        # 比较并返回更优的解
        if fit_original < fit_opposite:
            return mn_sol, un_sol, udot_original, fit_original
        else:
            return mn_opposite, un_opposite, udot_opposite, fit_opposite

    def update(self):
        """【修改】初始化种群"""
        for i in range(self.population_size):
            # 随机生成一个临时解
            mn_temp = random.sample(range(0, self.nt), self.nt)
            un_temp = [random.randint(0, self.nu - 1) for _ in range(self.nt)]
            
            # 应用OBL并获取最优解及其适应度
            self.mn[i], self.un[i], self.UdoT_list[i], self.fitness[i] = self._apply_obl_and_get_fitness(mn_temp, un_temp)
                
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
        sorted_data = sorted(zip(self.fitness, self.mn, self.un, self.UdoT_list))
        self.fitness, self.mn, self.un, self.UdoT_list = (list(t) for t in zip(*sorted_data))

    def wandering(self):
        """【修改】游走行为"""
        do_yes = 1
        for i in range(self.Tmax):
            self.snum = random.randint(int(self.population_size / (self.a + 1)), int(self.population_size / self.a))
            for j in range(1, self.snum + 1):
                temp_un = self.un[j].copy()
                temp_mn = self.mn[j].copy()

                num_mutations = self._generate_levy_step()
                for _ in range(num_mutations):
                    if random.random() < 0.5:
                        col = random.randint(0, self.nt - 1)
                        temp_un[col] = random.randint(0, self.nu - 1)
                    else:
                        col1, col2 = random.sample(range(self.nt), 2)
                        temp_mn[col1], temp_mn[col2] = temp_mn[col2], temp_mn[col1]
                
                # 应用OBL并计算新适应度
                temp_mn_final, temp_un_final, temp_UdoT, temp_fit = self._apply_obl_and_get_fitness(temp_mn, temp_un)
                
                if temp_fit < self.fitness[j]:
                    self.fitness[j] = temp_fit
                    self.mn[j], self.un[j], self.UdoT_list[j] = temp_mn_final.copy(), temp_un_final.copy(), temp_UdoT
                
                if temp_fit < self.fitness[0]:  # 更新领导者
                    self.swap_leader(j, temp_fit, temp_mn_final, temp_un_final, temp_UdoT)
                    do_yes = 0

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

    def call_followers(self):
        """【修改】召唤行为"""
        # 使用莱维飞行决定学习步长
        self.step_b = self._generate_levy_crossover_step()
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
            
            # 应用OBL并计算适应度
            temp_mn_final, temp_un_final, temp_UdoT, temp_fit = self._apply_obl_and_get_fitness(temp_mn, temp_un)
            
            if temp_fit < self.fitness[i]:
                self.fitness[i], self.mn[i], self.un[i], self.UdoT_list[i] = temp_fit, temp_mn_final, temp_un_final, temp_UdoT
            
            if temp_fit < self.fitness[0]:
                self.swap_leader(i, temp_fit, temp_mn_final, temp_un_final, temp_UdoT)
        
        self.sort_population()

    def siege(self):
        """【修改】围攻行为"""
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
            
                # 应用OBL并计算适应度
                temp_mn_final, temp_un_final, temp_UdoT, temp_fit = self._apply_obl_and_get_fitness(temp_mn, temp_un)
            
                if temp_fit < self.fitness[i]:
                    self.fitness[i], self.mn[i], self.un[i], self.UdoT_list[i] = temp_fit, temp_mn_final, temp_un_final, temp_UdoT
        
        self.sort_population()

    def live(self):
        """【修改】个体重新生成"""
        R = random.randint(int(self.population_size / (self.b + 1)), int(self.population_size / self.b))
        for i in range(self.population_size - R, self.population_size):
            mn_temp = random.sample(range(0, self.nt), self.nt)
            un_temp = [random.randint(0, self.nu - 1) for _ in range(self.nt)]
            
            # 应用OBL并直接更新种群
            self.mn[i], self.un[i], self.UdoT_list[i], self.fitness[i] = self._apply_obl_and_get_fitness(mn_temp, un_temp)

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
            self.plan_record[iter_count] = self.UdoT_list[0]

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