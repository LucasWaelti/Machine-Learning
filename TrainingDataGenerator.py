import math
import numpy as np


def store(data = [[0,0]]):
    size = len(data)
    dim = len(data[0])
    string = ''
    file = open('data.txt','w+')

    for i in range(0,size):
        for j in range(0,dim):
            string += str(data[i][j])
            if j != dim-1:
                string += ' '
        string += '\n'
        file.write(string)
        string = ''
        
    file.close()
    return

def read(path = 'data.txt'):
    # f(x) = y
    file = open(path,'r')
    data = []

    while(True):
        line = file.readline()
        if line == '':
            break

        # Detect comments in text file
        if(line[0] == '#'):
            continue
        
        x,y = line.split(' ')
        x = float(x)
        y = float(y)
        data.append([x,y])

    file.close()
    return data

def read2D(path = 'data.txt'):
    # f(x,y) = z
    file = open(path,'r')
    data = []

    while(True):
        line = file.readline()
        if line == '':
            break

        # Detect comments in text file
        if(line[0] == '#'):
            continue
        
        x,y,z = line.split(' ')
        x = float(x)
        y = float(y)
        z = float(z)
        data.append([x,y,z])

    file.close()
    return data

def function(x):
    # Define here the function the NN has to reproduce
    return x**3

def generateData(inter = [0,1], n = 50):
    step = math.fabs(inter[1]-inter[0])
    step /= n
    x = inter[0]
    y = 0
    data = []
    datax = []
    datay = []
    for i in range(0,n):
        x += step
        y = function(x)
        data.append([x,y])
        datax.append([[x]]) # Needs to be a list of lists because of multiple inputs possible for 1 neuron
        datay.append([y])
    store(data)
    return datax,datay
        

if __name__ == "__main__":
    
    generateData()
    



