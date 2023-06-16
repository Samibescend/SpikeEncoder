import numpy as np

def wFun(M, N, q, d, B):
    
    #Decoding weights 
    Ni = round(N/q) #Ratio E to I neurons is q to 1
    NAll = [N, Ni]
    w = [[], []]
    for i in range(0,2):
        w_ran = np.random.randn(M, NAll[i]) #standard normal
        for j in range(len(w_ran)):
            w_ran[j] = w_ran[j] / M
        w[i] = w_ran  #divide with the number of features
    
    w[1] = w[1] * d #sets STD of decoding weights of I neurons to d*sigma_w^E (sigma_w^E=1)

    print(w)

    #Fast connectivity matrices 

    C = [[], [], [], []]

    weights1 = [[] , w[1], w[1], w[0]]
    weights2 = [[] , w[1], w[0], w[1]]

    for i in range(0, 4):
        w1 = weights1[i]
        w2 = weights2[i]
        proj = np.transpose(w1) @ w2

        C[i] = (proj * (np.sign(proj)))

    #slower connectivity matrices
    print(B)
    D = [[], []]
    DEe = np.transpose(w[0]) @ B @ w[0]
    DIe = np.transpose(w[1]) @ B @ w[0]

    signEe = (np.sign(DEe))
    signIe = (np.sign(DIe))

    D[0] = DEe * signEe
    D[1] = DIe * signIe

    return w, C, D