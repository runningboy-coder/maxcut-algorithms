import itertools
import numpy as np

def exhaustive(graph):
    n = len(graph)
    # max_weight = 0
    min_energy = 0
    best_partition = None

    # 生成所有可能的划分
    for partition in itertools.product([-1, 1], repeat=n):
        # weight = 0
        energy = 0
        for i in range(n):
            for j in range(i + 1, n):
                # 如果两个顶点属于不同的集合，增加边的权重 max_cut 问题
                # if partition[i] != partition[j]:
                #     weight += graph[i][j]

                # 伊辛问题
                energy += graph[i][j]*partition[i]*partition[j]

        # 更新最大权重和最佳划分
        # if weight > max_weight:
        #     max_weight = weight
        #     best_partition = partition


        if energy < min_energy:
            min_energy = energy
            best_partition = partition



    return min_energy, np.array(best_partition)


