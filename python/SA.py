import numpy as np

def SA(J):
    def energy(current_spins):
        # 计算当前自旋配置的能量
        #return 0.25 * np.sum(np.sum(J)) - 0.25 * np.dot(current_spins.T, np.dot(J, current_spins)) ## max-cut问题
        return 0.5*np.dot(current_spins.T, np.dot(J, current_spins)) # 波形优化

    def random_spins(n):
        # 生成随机的自旋配置
        return np.random.choice([-1, 1], size=n)

    def perturb(current_spins):
        # 随机扰动自旋配置
        perturbed_spins = current_spins.copy()
        perturbed_spins[np.random.randint(0, len(current_spins))] = -perturbed_spins[np.random.randint(0, len(current_spins))]
        return perturbed_spins

    n = J.shape[0]  # 节点数
    current_spins = random_spins(n)  # 初始自旋配置
    current_energy = energy(current_spins)
    best_spins = current_spins.copy()
    best_energy = current_energy
    T = 1.0  # 初始温度
    T_min = 1e-6  # 最小温度
    alpha = 0.99  # 冷却率

    while T > T_min:
        new_spins = perturb(current_spins)
        new_energy = energy(new_spins)
        delta_E = new_energy - current_energy

        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_spins = new_spins
            current_energy = new_energy

        if current_energy < best_energy:
            best_energy = current_energy
            best_spins = current_spins.copy()

        T *= alpha  # 降低温度

    return best_energy, best_spins



