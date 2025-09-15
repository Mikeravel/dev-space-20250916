import numpy as np
import random

class PSO_GA_DWPA_UAV:
    def __init__(self, iterations, population_size, num_uavs, num_targets, uav_pos, target_pos,
                 step_a, step_b, step_c, step_d, T_max, alpha, beta, w1=0.5, w2=0.5, c1=0.8, c2=0.8):
        """
        根据论文 'Task Assignment of UAV Swarm Based on Wolf Pack Algorithm' 复现 PSO-GA-DWPA.
        
        参数:
        - iterations: 迭代次数
        - population_size: 狼群规模 (N)
        - num_uavs: 无人机数量 (Nv)
        - num_targets: 目标数量 (NT)
        - uav_pos: 无人机初始位置
        - target_pos: 目标位置
        - step_a, step_b, step_c, step_d: 各阶段的步长 (变异/交叉的列数)
        - T_max: 行走行为的最大次数
        - alpha, beta: 控制探索狼和淘汰狼数量的比例因子
        - w1, w2: 成本函数权重
        - c1, c2: PSO部分的学习因子 (交叉概率)
        """
        self.iterations = iterations
        self.N = population_size
        self.Nv = num_uavs
        self.NT = num_targets
        self.uav_pos = uav_pos
        self.target_pos = target_pos
        
        # 算法参数
        self.step_a = step_a
        self.step_b = step_b
        self.step_c = step_c
        self.step_d = step_d
        self.T_max = T_max
        self.alpha = alpha
        self.beta = beta
        self.w1 = w1
        self.w2 = w2
        self.c1 = c1
        self.c2 = c2

        # 初始化狼群
        self.population = self._initialize_population()
        self.costs = np.array([self.calculate_cost(wolf) for wolf in self.population])
        
        # 初始化头狼
        leader_idx = np.argmin(self.costs)
        self.leader_wolf = self.population[leader_idx].copy()
        self.leader_cost = self.costs[leader_idx]
        
        # 个体最优解 (用于行走行为)
        self.p_best = self.population.copy()
        self.p_best_costs = self.costs.copy()

    def _initialize_population(self):
        """生成初始种群 [cite: 216]"""
        population = []
        for _ in range(self.N):
            # 第一行：为每个目标随机分配一个无人机
            uav_assignments = np.random.randint(1, self.Nv + 1, size=self.NT)
            # 第二行：目标索引，保持不重复
            target_indices = np.arange(1, self.NT + 1)
            np.random.shuffle(target_indices)
            wolf = np.vstack([uav_assignments, target_indices])
            population.append(wolf)
        return np.array(population)

    def calculate_cost(self, wolf):
        """计算单匹狼（一个分配方案）的成本 [cite: 125]"""
        uav_routes = {i: [] for i in range(1, self.Nv + 1)}
        for j in range(self.NT):
            uav_id = wolf[0, j]
            target_id = wolf[1, j]
            uav_routes[uav_id].append(target_id)
            
        total_distance = 0
        max_distance = 0
        
        for uav_id, targets in uav_routes.items():
            if not targets:
                continue
            
            uav_path_dist = 0
            last_pos = self.uav_pos[uav_id - 1]
            
            # 按照wolf中出现的顺序确定攻击顺序
            sorted_targets = sorted(targets, key=lambda t: list(wolf[1,:]).index(t))

            for target_id in sorted_targets:
                target_idx = target_id - 1
                dist = np.linalg.norm(last_pos - self.target_pos[target_idx])
                uav_path_dist += dist
                last_pos = self.target_pos[target_idx]
            
            total_distance += uav_path_dist
            if uav_path_dist > max_distance:
                max_distance = uav_path_dist
        
        J1 = total_distance / self.Nv if self.Nv > 0 else 0
        J2 = max_distance
        
        return self.w1 * J1 + self.w2 * J2

    def _discrete_crossover(self, parent1, parent2):
        """离散交叉操作，用于PSO部分 [cite: 246]"""
        child = parent1.copy()
        a, b = sorted(random.sample(range(self.NT), 2))
        child[:, a:b+1] = parent2[:, a:b+1]
        return child

    def _discrete_mutation(self, wolf, step):
        """离散变异操作 [cite: 330]"""
        mutated_wolf = wolf.copy()
        if step > 0:
            columns_to_mutate = random.sample(range(self.NT), step)
            # 第一行随机分配UAV
            mutated_wolf[0, columns_to_mutate] = np.random.randint(1, self.Nv + 1, size=step)
            # 第二行随机交换目标
            if len(columns_to_mutate) > 1:
                permuted_targets = mutated_wolf[1, columns_to_mutate]
                np.random.shuffle(permuted_targets)
                mutated_wolf[1, columns_to_mutate] = permuted_targets
        return mutated_wolf

    def _gene_segment_duplication(self, wolf, leader, step):
        """基因片段复制，用于呼唤行为 [cite: 363]"""
        new_wolf = wolf.copy()
        if self.NT >= step > 0:
            start = random.randint(0, self.NT - step)
            new_wolf[:, start:start+step] = leader[:, start:start+step]
        return new_wolf

    def run(self):
        """执行PSO-GA-DWPA算法"""
        for iteration in range(self.iterations):
            # 1. 头狼选择 (Winner-take-all) [cite: 159]
            current_leader_idx = np.argmin(self.costs)
            if self.costs[current_leader_idx] < self.leader_cost:
                self.leader_cost = self.costs[current_leader_idx]
                self.leader_wolf = self.population[current_leader_idx].copy()
            
            # 排序狼群，确定探索狼、猛狼和弱狼
            sorted_indices = np.argsort(self.costs)
            
            # 2. 行走行为 (Walking Behavior - PSO Inspired) [cite: 227]
            num_exploring = random.randint(int(self.N / (self.alpha + 1)), int(self.N / self.alpha))
            exploring_indices = sorted_indices[1:num_exploring+1] # 头狼不参与

            for i in exploring_indices:
                for _ in range(self.T_max):
                    # a. 追踪个体极值 [cite: 234]
                    temp_wolf = self.population[i]
                    if random.random() < self.c1:
                        temp_wolf = self._discrete_crossover(temp_wolf, self.p_best[i])

                    # b. 追踪全局极值 [cite: 282]
                    if random.random() < self.c2:
                        temp_wolf = self._discrete_crossover(temp_wolf, self.leader_wolf)

                    # c. 个体变异 [cite: 325]
                    new_wolf = self._discrete_mutation(temp_wolf, self.step_a)
                    new_cost = self.calculate_cost(new_wolf)

                    if new_cost < self.costs[i]:
                        self.population[i] = new_wolf
                        self.costs[i] = new_cost
                        
                        # 更新个体最优
                        if new_cost < self.p_best_costs[i]:
                            self.p_best[i] = new_wolf.copy()
                            self.p_best_costs[i] = new_cost
                        
                        # 检查是否优于头狼
                        if new_cost < self.leader_cost:
                            self.leader_wolf = new_wolf.copy()
                            self.leader_cost = new_cost
                            break # 找到更好的头狼，停止行走

                if self.costs[i] < self.leader_cost: # 再次检查
                    break
            
            # 3. 呼唤行为 (Calling Behavior - GA Inspired) [cite: 357]
            num_fierce = self.N - num_exploring - 1
            fierce_indices = sorted_indices[num_exploring+1 : num_exploring+1+num_fierce]
            
            for i in fierce_indices:
                new_wolf = self._gene_segment_duplication(self.population[i], self.leader_wolf, self.step_b)
                new_cost = self.calculate_cost(new_wolf)
                if new_cost < self.costs[i]:
                    self.population[i] = new_wolf
                    self.costs[i] = new_cost
                    if new_cost < self.leader_cost:
                        self.leader_wolf = new_wolf.copy()
                        self.leader_cost = new_cost
                        break # 找到更好的头狼，停止呼唤

            # 4. 围攻行为 (Sieging Behavior) [cite: 403]
            for i in range(self.N):
                if np.array_equal(self.population[i], self.leader_wolf):
                    continue
                new_wolf = self._discrete_mutation(self.population[i], self.step_c)
                new_cost = self.calculate_cost(new_wolf)
                if new_cost < self.costs[i]:
                    self.population[i] = new_wolf
                    self.costs[i] = new_cost
            
            # 更新头狼
            current_leader_idx = np.argmin(self.costs)
            if self.costs[current_leader_idx] < self.leader_cost:
                self.leader_cost = self.costs[current_leader_idx]
                self.leader_wolf = self.population[current_leader_idx].copy()

            # 5. 种群更新 (Update Mechanism) [cite: 433]
            num_to_replace = random.randint(int(self.N / (2 * self.beta)), int(self.N / self.beta))
            weakest_indices = sorted_indices[-num_to_replace:]
            
            for i in weakest_indices:
                # 基于头狼进行小变异生成新狼
                new_wolf = self._discrete_mutation(self.leader_wolf, self.step_d)
                self.population[i] = new_wolf
                self.costs[i] = self.calculate_cost(new_wolf)
            
            print(f"Iteration {iteration+1}/{self.iterations}, Best Cost: {self.leader_cost}")

        return self.leader_wolf, self.leader_cost


# --- 使用示例 ---
if __name__ == '__main__':
    # 场景1：5 UAVs vs 8 目标 [cite: 475]
    NUM_UAVS = 5
    NUM_TARGETS = 8
    
    # 随机生成无人机和目标位置
    uav_positions = np.random.rand(NUM_UAVS, 2) * 100
    target_positions = np.random.rand(NUM_TARGETS, 2) * 100

    # 根据论文中的参数 Table 1 for Scenario 1 [cite: 469]
    params = {
        'iterations': 200,
        'population_size': 100,
        'num_uavs': NUM_UAVS,
        'num_targets': NUM_TARGETS,
        'uav_pos': uav_positions,
        'target_pos': target_positions,
        'step_a': 2,
        'step_b': 4,
        'step_c': 1,
        'step_d': 2,
        'T_max': 10,
        'alpha': 4,
        'beta': 5
    }

    # 创建并运行算法
    pso_ga_dwpa = PSO_GA_DWPA_UAV(**params)
    best_solution, best_cost = pso_ga_dwpa.run()

    print("\n--- 最终结果 ---")
    print(f"最优成本值: {best_cost}")
    print("最优分配方案 (第一行为UAV, 第二行为Target):")
    print(best_solution)