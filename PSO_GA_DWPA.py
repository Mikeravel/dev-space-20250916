import time
import random
import numpy as np

# ----------------- 依赖项目内文件 -----------------
import initial as ini
import fit
import deco
import limit
# ----------------------------------------------------

class psogadwpa:
    """
    根据论文 'Task Assignment of UAV Swarm Based on Wolf Pack Algorithm' 复现的PSO-GA-DWPA算法。
    该版本已完全适配到您提供的项目结构和数据编码方案 (mn, un)。
    """
    def __init__(self, population_size=100, max_iterations=800):
        # --- 基础参数 ---
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.nt = ini.nt
        self.nu = ini.nu

        # --- 论文中定义的PSO-GA-DWPA特定参数 ---
        # 步长（即每次操作改变的列数）
        self.step_a = 2                   # 行走行为中的变异步长
        self.step_b = max(1, int(self.nt / 2)) # 呼唤行为中的复制步长，设为任务数的一半
        self.step_c = max(1, int(self.nt / 10))# 围攻行为的变异步长
        self.step_d = 2                   # 新狼生成时的变异步长
        # 比例因子
        self.alpha = 4                    # 探狼比例因子
        self.beta = 5                     # 更新比例因子
        # 其他
        self.T_max = 10                   # 行走行为最大次数
        self.c1 = 0.8                     # PSO学习因子1 (个体)
        self.c2 = 0.8                     # PSO学习因子2 (全局)

        # --- 存储变量，结构与 NewWPA.py 保持一致 ---
        self.mn = [[] for _ in range(self.population_size)]
        self.un = [[] for _ in range(self.population_size)]
        self.UdoT_list = [[] for _ in range(self.population_size)]
        self.fitness = np.zeros(self.population_size)
        
        # 个体最优解 (p_best)
        self.p_best_mn = [[] for _ in range(self.population_size)]
        self.p_best_un = [[] for _ in range(self.population_size)]
        self.p_best_fitness = np.zeros(self.population_size)

        # 全局最优解 (头狼)
        self.leader_mn = []
        self.leader_un = []
        self.leader_fitness = float('inf')

        # 记录迭代过程
        self.fit_record = [0.0] * self.max_iterations
        self.plan_record = [[] for _ in range(self.max_iterations)]


    def _calculate_fitness(self, mn, un):
        """统一的适应度计算函数，封装了deco, limit和fit的调用"""
        udot = deco.deco(mn, un)
        if not limit.limit_verify(udot):
            return float('inf'), udot
        return fit.fit_cal(udot), udot

    def _initialize_population(self):
        """初始化狼群种群"""
        for i in range(self.population_size):
            self.mn[i] = random.sample(range(0, self.nt), self.nt)
            self.un[i] = [random.randint(0, self.nu - 1) for _ in range(self.nt)]
            self.fitness[i], self.UdoT_list[i] = self._calculate_fitness(self.mn[i], self.un[i])
        
        # 初始化个体最优
        self.p_best_mn = [m.copy() for m in self.mn]
        self.p_best_un = [u.copy() for u in self.un]
        self.p_best_fitness = self.fitness.copy()

        self.sort_population()

    def sort_population(self):
        """根据适应度对狼群进行排序"""
        sorted_indices = np.argsort(self.fitness)
        self.fitness = self.fitness[sorted_indices]
        self.mn = [self.mn[i] for i in sorted_indices]
        self.un = [self.un[i] for i in sorted_indices]
        self.UdoT_list = [self.UdoT_list[i] for i in sorted_indices]
        
        # 更新头狼
        if self.fitness[0] < self.leader_fitness:
            self.leader_fitness = self.fitness[0]
            self.leader_mn = self.mn[0].copy()
            self.leader_un = self.un[0].copy()

    def _handle_mn_duplicates(self, temp_mn, overwritten_tasks, segment_tasks, start_idx, step):
        """处理mn列表在交叉/复制后产生的重复任务，确保其仍为合法排列"""
        # 找出不在新片段中的被覆盖任务
        tasks_to_reinsert = [task for task in overwritten_tasks if task not in segment_tasks]
        
        # 遍历整个mn列表（除了刚插入的段）
        for i in list(range(0, start_idx)) + list(range(start_idx + step, self.nt)):
            # 如果当前位置的任务已经被新片段占用了，并且还有待重插入的任务
            if temp_mn[i] in segment_tasks and tasks_to_reinsert:
                # 用一个待重插入的任务替换它
                temp_mn[i] = tasks_to_reinsert.pop(0)
        return temp_mn

    def _discrete_crossover(self, mn1, un1, mn2, un2):
        """离散交叉，用于行走行为，学习p_best或g_best"""
        child_mn, child_un = mn1.copy(), un1.copy()
        
        # 随机选择交叉片段
        start = random.randint(0, self.nt - 2)
        end = random.randint(start + 1, self.nt - 1)
        step = end - start

        # 记录被覆盖的任务和新插入的任务
        overwritten_tasks = child_mn[start:end]
        segment_tasks = mn2[start:end]

        # 执行交叉
        child_mn[start:end] = segment_tasks
        child_un[start:end] = un2[start:end]
        
        # 修复mn使其恢复为合法排列
        child_mn = self._handle_mn_duplicates(child_mn, overwritten_tasks, segment_tasks, start, step)

        return child_mn, child_un

    def _discrete_mutation(self, mn, un, step):
        """离散变异，用于行走、围攻和新狼生成"""
        mutated_mn, mutated_un = mn.copy(), un.copy()
        if step > 0:
            cols = random.sample(range(self.nt), step)
            # 变异un
            for col in cols:
                mutated_un[col] = random.randint(0, self.nu - 1)
            # 变异mn（通过交换）
            if len(cols) > 1:
                # 打乱选定列对应的mn值
                shuffled_mn_parts = [mutated_mn[i] for i in cols]
                random.shuffle(shuffled_mn_parts)
                for i, col_idx in enumerate(cols):
                    mutated_mn[col_idx] = shuffled_mn_parts[i]
        return mutated_mn, mutated_un

    def wandering(self):
        """行走行为 (PSO-inspired)"""
        snum = random.randint(int(self.population_size / (self.alpha + 1)), int(self.population_size / self.alpha))
        exploring_indices = range(1, snum + 1) # 头狼不参与

        for i in exploring_indices:
            leader_found = False
            for _ in range(self.T_max):
                temp_mn, temp_un = self.mn[i].copy(), self.un[i].copy()
                
                # 1. 学习个体最优
                if random.random() < self.c1:
                    temp_mn, temp_un = self._discrete_crossover(temp_mn, temp_un, self.p_best_mn[i], self.p_best_un[i])
                
                # 2. 学习全局最优 (头狼)
                if random.random() < self.c2:
                    temp_mn, temp_un = self._discrete_crossover(temp_mn, temp_un, self.leader_mn, self.leader_un)
                
                # 3. 个体变异
                new_mn, new_un = self._discrete_mutation(temp_mn, temp_un, self.step_a)
                new_fit, new_udot = self._calculate_fitness(new_mn, new_un)

                if new_fit < self.fitness[i]:
                    self.mn[i], self.un[i], self.fitness[i], self.UdoT_list[i] = new_mn, new_un, new_fit, new_udot
                    
                    if new_fit < self.p_best_fitness[i]:
                        self.p_best_mn[i], self.p_best_un[i], self.p_best_fitness[i] = new_mn.copy(), new_un.copy(), new_fit
                    
                    if new_fit < self.leader_fitness:
                        self.leader_fitness, self.leader_mn, self.leader_un = new_fit, new_mn.copy(), new_un.copy()
                        leader_found = True
                        break # 找到更优头狼，当前探狼停止本次行走
            if leader_found:
                break # 任何一个探狼找到更优头狼，则所有探狼结束行走

    def call_followers(self):
        """呼唤行为 (GA-inspired gene duplication)"""
        snum = random.randint(int(self.population_size / (self.alpha + 1)), int(self.population_size / self.alpha))
        fierce_indices = range(snum + 1, self.population_size)

        for i in fierce_indices:
            new_mn, new_un = self.mn[i].copy(), self.un[i].copy()
            
            # 随机选择复制片段
            start = random.randint(0, self.nt - self.step_b)
            end = start + self.step_b
            
            overwritten_tasks = new_mn[start:end]
            segment_tasks = self.leader_mn[start:end]

            # 执行复制
            new_mn[start:end] = segment_tasks
            new_un[start:end] = self.leader_un[start:end]
            
            # 修复mn
            new_mn = self._handle_mn_duplicates(new_mn, overwritten_tasks, segment_tasks, start, self.step_b)
            
            new_fit, new_udot = self._calculate_fitness(new_mn, new_un)

            if new_fit < self.fitness[i]:
                self.mn[i], self.un[i], self.fitness[i], self.UdoT_list[i] = new_mn, new_un, new_fit, new_udot
                if new_fit < self.leader_fitness:
                    self.leader_fitness, self.leader_mn, self.leader_un = new_fit, new_mn.copy(), new_un.copy()
                    # 在论文中，这里也会中断，为简化并与现有框架融合，我们完成全部猛狼的更新后再统一排序
    
    def siege(self):
        """围攻行为"""
        for i in range(1, self.population_size): # 头狼不参与
            new_mn, new_un = self._discrete_mutation(self.mn[i], self.un[i], self.step_c)
            new_fit, new_udot = self._calculate_fitness(new_mn, new_un)
            if new_fit < self.fitness[i]:
                self.mn[i], self.un[i], self.fitness[i], self.UdoT_list[i] = new_mn, new_un, new_fit, new_udot

    def live(self):
        """种群更新 (优胜劣汰)"""
        R = random.randint(int(self.population_size / (2 * self.beta)), int(self.population_size / self.beta))
        weakest_indices = range(self.population_size - R, self.population_size)
        
        for i in weakest_indices:
            # 基于头狼进行小变异生成新狼
            new_mn, new_un = self._discrete_mutation(self.leader_mn, self.leader_un, self.step_d)
            self.mn[i] = new_mn
            self.un[i] = new_un
            self.fitness[i], self.UdoT_list[i] = self._calculate_fitness(new_mn, new_un)

    def run(self):
        """主执行函数，接口与NewWPA.py保持一致"""
        start_time = time.time()
        
        # 1. 初始化
        self._initialize_population()
        
        iter_count = 0
        while iter_count < self.max_iterations:
            # 2. 行走行为
            self.wandering()
            
            # 3. 呼唤行为
            self.call_followers()
            
            # 4. 围攻行为
            self.siege()
            
            # 5. 种群更新
            self.live()

            # 6. 每轮迭代后排序，选出新头狼
            self.sort_population()
            
            # 记录当前迭代的最优结果
            self.fit_record[iter_count] = self.leader_fitness
            self.plan_record[iter_count] = deco.deco(self.leader_mn, self.leader_un) # 确保plan也是最优的

            if (iter_count + 1) % 50 == 0:
                print(f"PSO-GA-DWPA 迭代 {iter_count + 1}/{self.max_iterations}, 当前最优适应度: {self.leader_fitness}")
            
            iter_count += 1
            
        end_time = time.time()
        print(f"\nPSO-GA-DWPA 算法运行结束，耗时 {end_time - start_time:.2f} 秒。")

        # 整理并返回与NewWPA.py相同的变量
        min_best_fit = min(self.fit_record)
        min_index = self.fit_record.index(min_best_fit)
        min_best_plan = self.plan_record[min_index]
        task_complete_time, average_time = fit.timecost(min_best_plan)
        
        return min_best_plan, task_complete_time, average_time, min_best_fit, self.fit_record
