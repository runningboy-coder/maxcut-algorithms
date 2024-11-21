import torch
import numpy as np
from utils import ising_energy



def SB_gpu(J, a0=1, c1=0.5, stepNum=10, deltaT=0.01, algorithm="bSB", beta=0):
    """
    J: coupling cofficient matrix
    a: pump rate
    c1: constan paramerter
    stepNum: iteration time
    deltaT: time interval
    algorithm: aSB, dSB, bSB
    beat: heated term, when it greater than 0, procedure is heated, smaller than 0, it is cooling.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 将 J 转换为 PyTorch 张量并移动到 GPU
    J_tensor = torch.tensor(J, dtype=torch.float32).to(device)
    vertexNum = len(J)
    a = torch.zeros(1, device=device)  # 0
    position = torch.zeros(vertexNum, 1, device=device)
    momentum = torch.randn(vertexNum, 1, device=device)
    deltaT = torch.tensor(deltaT, device=device)
    couplingCofficient = c1 / torch.sqrt(torch.tensor(vertexNum, device=device))

    for t in range(stepNum):
        
        a = t * a0 / stepNum

        torch.add(momentum, position, alpha=(a - a0)*deltaT, out=momentum)

        ## position update x(t+1) = x(t) + a0*y(t)*deltat
        torch.add(position, momentum, alpha=a0*deltaT, out=position)

        ## momentum update y(t+1) = (1+gamma*deltaT)y(t) - (a0 -a)x(t)*deltaT + c0*f(t)*deltaT
        momentum = torch.addmm(momentum, J_tensor, torch.sign(position), alpha=couplingCofficient*deltaT)

        ## inelastic wall
        # print("momentum:",momentum)
        # print("position: ",position)
        momentum[torch.abs(position) > 1.0] = 0.0
        torch.clip(position, -1.0, 1.0, out=position)

        

    spinConfig = torch.sign(position)
    # 将 PyTorch 张量转换为 NumPy ndarray 以调用 cut 函数
    spinConfig_np = spinConfig.cpu().numpy()
    J_np = J_tensor.cpu().numpy()
    cutSize = ising_energy(spinConfig_np, J_np)

    return cutSize, spinConfig.cpu().numpy()

