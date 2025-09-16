import multiprocessing
import numpy as np
import time

# from MPWPA import Mpwpa
from MPPWPA import mppwpa
from mycode import mdwpa
from LWPA import lwpa
from NewWPA import nlwpa
from PSO_GA_DWPA import psogadwpa
from RMPP import MPPWPA_reproduced
# --- 配置区 ---
# 定义每个算法需要运行的次数
RUNS_psogadwpa = 100
RUNS_MPPWPA_reproduced = 100
RUNS_MPPWPA = 100
RUNS_MDWPA = 100
RUNS_NLWPA = 100
RUNS_LWPA = 100



multiprocessing.cpu_count() 

NUM_CORES = 4

# --- 包装函数区 ---
# 为每个算法创建一个包装函数，用于被 pool.map 调用

# 参数 "_" 是一个占位符，因为 pool.map 需要传递一个参数，实际上用不到
def run_psogadwpa(_):
    """调用 psogadwpa 并返回 gbest_fitness"""
    plan, task_complete_time, average_time, gbest_fitness, best_fitness_mat = psogadwpa().run()
    return gbest_fitness

def run_MPPWPA_reproduced(_):
    """调用 MPPWPA_reproduced 并返回 gbest_fitness"""
    plan, task_complete_time, average_time, gbest_fitness, best_fitness_mat = MPPWPA_reproduced().run()
    return gbest_fitness

def run_mppwpa(_):
    """调用 mppwpa 并返回 gbest_fitness"""
    plan, task_complete_time, average_time, gbest_fitness, best_fitness_mat = mppwpa()
    return gbest_fitness

def run_mdwpa(_):
    """调用 mdwpa 并返回 gbest_fitness"""
    plan, task_complete_time, average_time, gbest_fitness, best_fitness_mat = mdwpa()
    return gbest_fitness

def run_nlwpa(_):
    """调用 nlwpa().run() 并返回 gbest_fitness"""
    plan, task_complete_time, average_time, gbest_fitness, best_fitness_mat = nlwpa().run()
    return gbest_fitness

def run_lwpa(_):
    """调用 lwpa().run() 并返回 gbest_fitness"""
    plan, task_complete_time, average_time, gbest_fitness, best_fitness_mat = lwpa().run()
    return gbest_fitness


# --- 主程序入口 ---
if __name__ == "__main__":

    
    print(f"开始并行计算，使用 {NUM_CORES} 个CPU核心...")
    start_time = time.time()

    # 创建一个进程池，管理所有的并行任务
    with multiprocessing.Pool(processes=NUM_CORES) as pool:

        print(f"\n正在并行执行 {RUNS_psogadwpa} 次 psogadwpa 算法...")
        numbers1 = pool.map(run_psogadwpa, range(RUNS_psogadwpa))
        print("psogadwpa 完成！")
        

        print(f"\n正在并行执行 {RUNS_MPPWPA_reproduced} 次 MPPWPA_reproduced 算法...")
        numbers2 = pool.map(run_MPPWPA_reproduced, range(RUNS_MPPWPA_reproduced))
        print("MPPWPA_reproduced 完成！")


        print(f"\n正在并行执行 {RUNS_MPPWPA} 次 mppwpa 算法...")
        numbers3 = pool.map(run_mppwpa, range(RUNS_MPPWPA))
        print("mppwpa 完成！")


        print(f"\n正在并行执行 {RUNS_MDWPA} 次 mdwpa 算法...")
        numbers4 = pool.map(run_mdwpa, range(RUNS_MDWPA))
        print("mdwpa 完成！")


        print(f"\n正在并行执行 {RUNS_NLWPA} 次 nlwpa 算法...")
        numbers5 = pool.map(run_nlwpa, range(RUNS_NLWPA))
        print("nlwpa 完成！")

        print(f"\n正在并行执行 {RUNS_LWPA} 次 lwpa 算法...")
        numbers6 = pool.map(run_lwpa, range(RUNS_LWPA))
        print("lwpa 完成！")

    end_time = time.time()
    print(f"\n所有计算任务完成，总耗时: {(end_time - start_time) / 60:.2f} 分钟")

    # --- 计算和打印结果 ---

    a = np.mean(numbers1)
    amin = np.min(numbers1)

    b = np.mean(numbers2)
    bmin = np.min(numbers2)

    c = np.mean(numbers3)
    cmin = np.min(numbers3)

    d = np.mean(numbers4)
    dmin = np.min(numbers4)

    e = np.mean(numbers5)
    emin = np.min(numbers5)

    f = np.mean(numbers6)
    fmin = np.min(numbers6)

    print("\n--- 平均适应度值 ---")
    print(f"平均数a (psogadwpa): {a} 最小值a: {amin}")

    print(f"平均数b (MPPWPA_reproduce): {b} 最小值b: {bmin}")

    print(f"平均数c (mppwpa): {c} 最小值c: {cmin}")

    print(f"平均数d (mdwpa): {d}  最小值d: {dmin}")

    print(f"平均数g (nlwpa): {e} 最小值e: {emin}")

    print(f"平均数g (lwpa): {f} 最小值f: {fmin}")