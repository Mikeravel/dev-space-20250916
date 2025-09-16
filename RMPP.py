import math
import fit
import initial as ini
import numpy as np
import random
import deco
import limit
import time
from copy import deepcopy

class MPPWPA_reproduced:
    """
    根据论文 'Multi-Population Parallel Wolf Pack Algorithm for Task Assignment of UAV Swarm' (applsci-11-11996.pdf)
    复现的多群体并行狼群算法 (MPPWPA)。
    该实现严格遵循用户提供的项目结构和编码方式 (mn, un)。
    """
    def __init__(self, population_size=160, max_iterations=800, num_sub_pops=4, migration_prob=0.8, mutation_ratio=0.2, redundancy_interval=5):
        # 核心参数
        self.N = population_size              # 种群数量 (N)
        self.max_iter = max_iterations        # 最大迭代次数
        self.nt = ini.nt                      # 目标数量
        self.nu = ini.nu                      # 无人机数量

        # MPPWPA 特定参数
        self.Num = num_sub_pops               # 质量子种群数量
        self.Pm = migration_prob              # 种群迁移概率
        self.tau = mutation_ratio             # 种群预处理中的变异比例
        self.AI = redundancy_interval         # 种群预处理中移除冗余个体的间隔

        # WPA 内部参数 (参考论文 Table 2, Scenario 2)
        self.alpha = 4                        # 探狼比例因子
        self.beta = 5                         # 更新比例因子
        self.T_max = 10                       # 行走行为最大次数
        self.step_a = 2                       # 行走行为步长
        self.step_b = 14                      # 召唤行为步长
        self.step_c = 1                       # 围攻行为步长

        # 种群数据结构
        self.mn_pop = [[] for _ in range(self.N)] # 任务序列种群
        self.un_pop = [[] for _ in range(self.N)] # 无人机分配种群
        self.fitness = np.zeros(self.N)           # 适应度值

        # 最优解记录
        self.gbest_mn = []
        self.gbest_un = []
        self.gbest_fitness = float('inf')
        self.fit_record = []

    def _calculate_fitness(self, mn, un):
        """计算单个解的适应度值"""
        udot = deco.deco(mn, un)
        if not limit.limit_verify(udot):
            return float('inf')
        return fit.fit_cal(udot)

    def _initialize_population(self):
        """初始化整个狼群"""
        for i in range(self.N):
            self.mn_pop[i] = random.sample(range(self.nt), self.nt)
            self.un_pop[i] = [random.randint(0, self.nu - 1) for _ in range(self.nt)]
            self.fitness[i] = self._calculate_fitness(self.mn_pop[i], self.un_pop[i])
        
        self._sort_population(self.fitness, self.mn_pop, self.un_pop)
        self.gbest_mn = self.mn_pop[0][:]
        self.gbest_un = self.un_pop[0][:]
        self.gbest_fitness = self.fitness[0]

    def _sort_population(self, fitness, mn_pop, un_pop):
        """对种群及其相关列表进行排序"""
        sorted_indices = np.argsort(fitness)
        self.fitness = fitness[sorted_indices]
        self.mn_pop = [mn_pop[i] for i in sorted_indices]
        self.un_pop = [un_pop[i] for i in sorted_indices]
        return self.fitness, self.mn_pop, self.un_pop
        
    def _wpa_operator(self, mn_sub_pop, un_sub_pop, fitness_sub):
        """对一个子种群执行完整的WPA单次迭代操作"""
        sub_pop_size = len(mn_sub_pop)
        
        # 1. 确定头狼
        leader_idx = np.argmin(fitness_sub)
        leader_mn, leader_un = mn_sub_pop[leader_idx][:], un_sub_pop[leader_idx][:]

        # 2. 行走行为 (Walking Behavior)
        snum = random.randint(int(sub_pop_size / (self.alpha + 1)), int(sub_pop_size / self.alpha))
        sorted_indices = np.argsort(fitness_sub)

        for i in sorted_indices[1:snum+1]: # 头狼不参与
            for _ in range(self.T_max):
                # 随机变异
                temp_mn, temp_un = mn_sub_pop[i][:], un_sub_pop[i][:]
                cols_to_mutate = random.sample(range(self.nt), self.step_a)
                for col in cols_to_mutate:
                    temp_un[col] = random.randint(0, self.nu - 1)
                
                # 交换mn中的任务
                if self.step_a > 1:
                    t1_idx, t2_idx = cols_to_mutate[0], cols_to_mutate[1]
                    temp_mn[t1_idx], temp_mn[t2_idx] = temp_mn[t2_idx], temp_mn[t1_idx]
                
                new_fit = self._calculate_fitness(temp_mn, temp_un)
                if new_fit < fitness_sub[i]:
                    fitness_sub[i], mn_sub_pop[i], un_sub_pop[i] = new_fit, temp_mn, temp_un
                    if new_fit < fitness_sub[leader_idx]:
                        leader_idx = i
                        leader_mn, leader_un = mn_sub_pop[i][:], un_sub_pop[i][:]
                        break 
            if fitness_sub[i] < fitness_sub[leader_idx]:
                break

        # 3. 召唤行为 (Calling Behavior)
        for i in sorted_indices[snum+1:]:
            temp_mn, temp_un = mn_sub_pop[i][:], un_sub_pop[i][:]
            
            # 从头狼复制基因片段
            start_idx = random.randint(0, self.nt - self.step_b)
            copied_tasks = leader_mn[start_idx : start_idx + self.step_b]
            
            # 替换 un
            for k in range(self.step_b):
                idx = start_idx + k
                temp_un[idx] = leader_un[idx]

            # 替换 mn 并修复冲突
            current_tasks_in_segment = temp_mn[start_idx : start_idx + self.step_b]
            tasks_to_replace = [t for t in copied_tasks if t not in current_tasks_in_segment]
            
            # 找到被复制过来的任务在原个体中的位置，并用被挤出的任务替换它们
            if tasks_to_replace:
                original_indices_of_new_tasks = [temp_mn.index(t) for t in tasks_to_replace]
                
                temp_mn[start_idx : start_idx + self.step_b] = copied_tasks

                displaced_tasks = [t for t in current_tasks_in_segment if t not in copied_tasks]

                for idx, task in zip(original_indices_of_new_tasks, displaced_tasks):
                    temp_mn[idx] = task

            new_fit = self._calculate_fitness(temp_mn, temp_un)
            if new_fit < fitness_sub[i]:
                fitness_sub[i], mn_sub_pop[i], un_sub_pop[i] = new_fit, temp_mn, temp_un
                if new_fit < fitness_sub[leader_idx]:
                    leader_idx = i
                    leader_mn, leader_un = mn_sub_pop[i][:], un_sub_pop[i][:]
                    break

        # 4. 围攻行为 (Sieging Behavior)
        for i in range(sub_pop_size):
            if i == leader_idx: continue
            
            temp_mn, temp_un = mn_sub_pop[i][:], un_sub_pop[i][:]
            cols_to_learn = random.sample(range(self.nt), self.step_c)
            
            for col in cols_to_learn:
                task_in_leader = leader_mn[col]
                # 在当前个体中找到这个任务，并学习其UAV分配
                try:
                    current_task_idx = temp_mn.index(task_in_leader)
                    temp_un[current_task_idx] = leader_un[col]
                except ValueError:
                    continue # 如果任务不在当前个体中（不太可能发生），则跳过

            new_fit = self._calculate_fitness(temp_mn, temp_un)
            if new_fit < fitness_sub[i]:
                fitness_sub[i], mn_sub_pop[i], un_sub_pop[i] = new_fit, temp_mn, temp_un

        # 5. 更新行为 (Update)
        r = random.randint(int(sub_pop_size / (self.beta + 1)), int(sub_pop_size / self.beta))
        sorted_indices = np.argsort(fitness_sub)
        for i in sorted_indices[-r:]:
            mn_sub_pop[i] = random.sample(range(self.nt), self.nt)
            un_sub_pop[i] = [random.randint(0, self.nu - 1) for _ in range(self.nt)]
            fitness_sub[i] = self._calculate_fitness(mn_sub_pop[i], un_sub_pop[i])
            
        return mn_sub_pop, un_sub_pop, fitness_sub

    def _pretreatment(self, iteration):
        """种群预处理"""
        # Step 1 & 2: 变异最优个体
        num_mutate = int(self.tau * self.N)
        mutant_mn, mutant_un, mutant_fitness = [], [], []
        
        for i in range(num_mutate):
            temp_mn, temp_un = self.mn_pop[i][:], self.un_pop[i][:]
            # 轻微变异
            col1, col2 = random.sample(range(self.nt), 2)
            temp_mn[col1], temp_mn[col2] = temp_mn[col2], temp_mn[col1]
            temp_un[col1] = random.randint(0, self.nu - 1)
            
            mutant_mn.append(temp_mn)
            mutant_un.append(temp_un)
            mutant_fitness.append(self._calculate_fitness(temp_mn, temp_un))

        # Step 3: 合并并移除冗余
        temp_mn_pop = self.mn_pop + mutant_mn
        temp_un_pop = self.un_pop + mutant_un
        temp_fitness = np.concatenate([self.fitness, np.array(mutant_fitness)])

        if iteration % self.AI == 0:
            unique_indices = []
            seen = set()
            for i, mn in enumerate(temp_mn_pop):
                # 将解转换为可哈希的元组以便于去重
                solution_tuple = (tuple(mn), tuple(temp_un_pop[i]))
                if solution_tuple not in seen:
                    unique_indices.append(i)
                    seen.add(solution_tuple)
            
            temp_mn_pop = [temp_mn_pop[i] for i in unique_indices]
            temp_un_pop = [temp_un_pop[i] for i in unique_indices]
            temp_fitness = temp_fitness[unique_indices]

        # Step 4: 填充新个体
        num_to_add = self.N - len(temp_mn_pop)
        for _ in range(num_to_add):
            mn = random.sample(range(self.nt), self.nt)
            un = [random.randint(0, self.nu - 1) for _ in range(self.nt)]
            temp_mn_pop.append(mn)
            temp_un_pop.append(un)
            temp_fitness = np.append(temp_fitness, self._calculate_fitness(mn, un))

        # Step 5: 选择最优的N个个体
        sorted_indices = np.argsort(temp_fitness)
        self.mn_pop = [temp_mn_pop[i] for i in sorted_indices[:self.N]]
        self.un_pop = [temp_un_pop[i] for i in sorted_indices[:self.N]]
        self.fitness = temp_fitness[sorted_indices[:self.N]]

    def _approximate_average_division(self):
        """近似平均划分"""
        sub_pop_size = self.N // self.Num
        
        # 精英子种群
        elite_indices = list(range(sub_pop_size))
        
        # 质量子种群
        mass_pop_indices = [[] for _ in range(self.Num)]
        
        # 确保即使在无法整除的情况下也能处理所有个体
        num_segments = (self.N + self.Num - 1) // self.Num

        for i in range(num_segments):
            segment_indices = list(range(i * self.Num, min((i + 1) * self.Num, self.N)))
            random.shuffle(segment_indices)
            for j, idx in enumerate(segment_indices):
                mass_pop_indices[j].append(idx)
        
        return elite_indices, mass_pop_indices

    def run(self):
        """执行 MPPWPA 算法"""
        start_time = time.time()
        self._initialize_population()

        for it in range(self.max_iter):
            # 种群迁移 (通过融合、预处理和重分割实现)
            if random.random() < self.Pm:
                self._pretreatment(it)
                self._sort_population(self.fitness, self.mn_pop, self.un_pop)

            # 划分种群
            elite_indices, mass_pop_indices = self._approximate_average_division()

            # 提取精英种群数据
            elite_mn = [self.mn_pop[i] for i in elite_indices]
            elite_un = [self.un_pop[i] for i in elite_indices]
            elite_fitness = self.fitness[elite_indices]
            
            # 优化精英子种群
            elite_mn, elite_un, elite_fitness = self._wpa_operator(elite_mn, elite_un, elite_fitness)
            
            # 将优化后的精英个体放回主种群
            for i, idx in enumerate(elite_indices):
                self.mn_pop[idx], self.un_pop[idx], self.fitness[idx] = elite_mn[i], elite_un[i], elite_fitness[i]
                
            # 优化质量子种群 (串行模拟并行)
            for sub_pop_idx in range(self.Num):
                indices = mass_pop_indices[sub_pop_idx]
                mass_mn = [self.mn_pop[i] for i in indices]
                mass_un = [self.un_pop[i] for i in indices]
                mass_fitness = self.fitness[indices]

                mass_mn, mass_un, mass_fitness = self._wpa_operator(mass_mn, mass_un, mass_fitness)
                
                # 将优化后的质量子种群个体放回主种群
                for i, idx in enumerate(indices):
                    self.mn_pop[idx], self.un_pop[idx], self.fitness[idx] = mass_mn[i], mass_un[i], mass_fitness[i]
            
            # 更新全局最优解
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.gbest_fitness:
                self.gbest_fitness = self.fitness[current_best_idx]
                self.gbest_mn = self.mn_pop[current_best_idx][:]
                self.gbest_un = self.un_pop[current_best_idx][:]

            self.fit_record.append(self.gbest_fitness)
            if (it + 1) % 50 == 0:
                print(f"迭代 {it + 1}/{self.max_iter}, 当前最优适应度: {self.gbest_fitness}")

        end_time = time.time()
        print(f"\n算法运行结束，耗时 {end_time - start_time:.2f} 秒。")

        min_best_plan = deco.deco(self.gbest_mn, self.gbest_un)
        task_complete_time, average_time = fit.timecost(min_best_plan)

        return min_best_plan, task_complete_time, average_time, self.gbest_fitness, self.fit_record

# --- 主程序入口 ---
if __name__ == '__main__':
    print("开始运行论文复现的 MPPWPA 算法...")
    
    # 创建算法实例
    mppwpa_algo = MPPWPA_reproduced(
        population_size=160, 
        max_iterations=400, 
        num_sub_pops=4,       # 子种群数量，可以调整为 2, 8, 16 进行测试
        migration_prob=0.8    # 迁移概率
    )
    
    # 运行算法
    plan, total_time, avg_time, best_fit, fit_history = mppwpa_algo.run()

    # 打印结果
    print("\n--- 最终结果 ---")
    print(f"最优适应度: {best_fit}")
    print(f"总完成时间: {total_time}")
    print(f"平均飞行时间: {avg_time}")
    print("最优分配方案 (UdoT格式):")
    # 为了清晰显示，逐行打印
    for i, tasks in enumerate(plan):
        if tasks: # 只打印有任务的无人机
            print(f"  UAV {i}: {tasks}")
