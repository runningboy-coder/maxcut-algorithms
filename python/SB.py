import numpy as np
from utils import cut, ising_energy


def SB(J, a0=1, c1=0.5, stepNum=100, deltaT=0.01, algorithm="bSB", beta=0):
    """
    J: coupling cofficient matrix = -W
    a: pump rate
    c1: constan paramerter
    stepNum: iteration time
    deltaT: time interval
    algorithm: aSB, dSB, bSB
    beat: heated term, when it greater than 0, procedure is heated, smaller than 0, it is cooling.

    """
    vertexNum = len(J)
    a = 0
    position = np.zeros(vertexNum)
    #position = np.random.randn(vertexNum)
    momentum = np.random.randn(vertexNum)
    deltaT = 0.01
    couplingTerm = 0
    couplingCofficient  = c1 / np.sqrt(vertexNum)

    for t in range(stepNum):
        a = t*a0/stepNum

        for j in range(vertexNum):
            couplingTerm = 0
            for i in range(vertexNum):
                #调用ballistic SB
                if algorithm=="bSB" or algorithm=='aSB':
                    couplingTerm += J[j][i] * position[i]
                elif algorithm=="dSB":
                    couplingTerm += J[j][i] * np.sign(position[i])

                else:
                    print("Algorithm only support \'dSB\',\'bSB\',\'aSB\'.")
                    return 

    
            position[j] = a0*momentum[j]*deltaT + position[j]

            if abs(position[j]) > 1:
                position[j] = np.sign(position[j])
                momentum[j] = 0
                continue

            if algorithm=="dSB" or algorithm=="bSB":
                momentum[j] = momentum[j] -((a0 - a) * position[j] + couplingCofficient * couplingTerm)*deltaT + beta*momentum[j]
            elif algorithm=='aSB':
                momentum[j] = momentum[j] -((position[j]**2 + (a0 - a) )* position[j] + couplingCofficient * couplingTerm)*deltaT + beta*momentum[j]
    
    spinConfig  = np.sign(position)
    # cutSize = cut(spinConfig, J)
    energy = ising_energy(spinConfig, J)

    return  energy, spinConfig