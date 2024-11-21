import os
import sys
sys.path.append('..') 
import numpy as np
from SB import SB
from SA import SA
from SB_gpu import SB_gpu
from exhaustive import exhaustive
import time
from utils import ising_energy

import numpy as np

def read_ising_data():
    with open(r'H:\研究生工作\LPN\LPNToIsing\EXp1\IsingExample.txt', 'r') as file:
        lines = file.readlines()

    # 找到向量和矩阵的起始行
    vector_start = lines.index('IsingSample  --   Ising_Orig_Xsolv:\n') + 1
    matrix_start = lines.index('IsingSample  --   Ising_Link_Jmatrix:\n') + 1

    # 读取向量
    vector = []
    for line in lines[vector_start:matrix_start - 1]:
        values = line.split()
        for value in values:
            vector.append(int(value))

    # 读取矩阵
    matrix = []
    for line in lines[matrix_start:]:
        if line.strip():
            row = [int(x) for x in line.split()]
            matrix.append(row)

    return np.array(vector), np.array(matrix)

vector, matrix = read_ising_data()
print("Ising_Orig_Xsolv向量维度:", vector.shape)
print("Ising_Link_Jmatrix矩阵维度:", matrix.shape)

energy, spins = SB_gpu(matrix)
print(energy)