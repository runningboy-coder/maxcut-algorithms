import numpy as np

def cut(spin, weight):
    cutSize = 0.25 * np.sum(np.sum(weight))
    for i in range(len(weight)):
        for j in range(len(weight)):
            cutSize -= 0.25 * weight[i][j]*spin[i]*spin[j]
    return cutSize

def ising_energy(spin, weight):
    return 0.5*np.dot(spin.T, np.dot(weight, spin))

def getdata(filename='G05.csv'):
    path = "H:/研究生工作/算法研究/算法复现/data_set/" + filename;
    edges = []
    with open(path,"r") as data_file:
        data = data_file.readline()
        while data:
            data = data.split(" ")
            edges.append([eval(data[0]), eval(data[1]), eval(data[2])])
            data = data_file.readline()

    edges = np.array(edges)
    # print(edges)
    vertexNum = edges[1:].max()
    graph = np.zeros((vertexNum,vertexNum))
    for i in range(1,len(edges)):
        graph[edges[i][0]-1][edges[i][1]-1] = edges[i][2]
        graph[edges[i][1]-1][edges[i][0]-1] = edges[i][2]
    
    return  graph, vertexNum
