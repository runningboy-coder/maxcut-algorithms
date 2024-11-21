# 读取数据
# 解决最大割问
import numpy as np
from utils import getdata
from SB import SB

J, vertex = getdata("G05.csv")
print(J)
_, cutSize = SB(J, c1=230, beta=0.18)
print(_,cutSize)







