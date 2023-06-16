import numpy as np
def netFun(dt, sigmav, mu, tau_vec, s, w, C, D):

    M = len(s)
    T = len(s[0])
    N = len(w[0][0])
    print(N)
    Ni = len(w[1][0])

    lambdaE = 1/tau_vec[0]
    lambdaI = 1/tau_vec[1]
    alphaE = 1/tau_vec[2]
    alphaI = 1/tau_vec[3]

    deltaE = mu * (lambdaE - alphaE) #multiplies local current in E 
    deltaI = mu * (lambdaI - alphaI) #multipies local current in I 
    deltaIE = (lambdaI - lambdaE) #multiplies slow E-to-I synaptic curent

    CII = C[1]
    CIE = C[2]
    CEI = C[3]

    DEE = D[0]
    DIE = D[1]

    thresE =  np.diag(np.transpose(w[0]) @ w[0]) / (2 + (mu / 2)) #Firing threshold E neurons
    thresI =  np.diag(np.transpose(w[1]) @ w[1]) / (2 + (mu / 2)) #Firing treshold I neurons

    #to speed up the integration

    leakE = 1 - lambdaE *  dt #for leak current in E
    leakI =  1 - lambdaI * dt #for leak current in I

    noiseE = sigmav * np.random.randn(N, T) * np.sqrt(2 * dt)
    noiseI = sigmav * np.random.randn(Ni, T) * np.sqrt(2 * dt)

    ffe = np.transpose(w[0]) @ s * dt

    ##Integration

    Ve = np.zeros((N, T))  # membrane potential
    fe = np.zeros((N, T))  # spike train
    re = np.zeros((N, T))  # filtered spike train
    ze = np.zeros((N, T))  # filtered spike train
    xhatE = np.zeros((N, T)) #excitatory estimate

    Vi = np.zeros((Ni, T))  # membrane potential
    fi = np.zeros((Ni, T))  # spike train
    ri = np.zeros((Ni, T))  # filtered spike train
    xhati = np.zeros((Ni, T)) #excitatory estimate


    for i in range(N):

        Ve[i][0]  = np.random.randn(1) * 2 - 12  #initialisation with random membrane potentials 
    for i in range(Ni):
        Vi[i][0]  = np.random.randn(1) * 2 - 12

    for t in range(500):
        #E neurons
        for i in range(N): 
            Ve[i][t + 1] = leakE * Ve[i][t] + ffe[i][t] - CEI[i] @ np.transpose(fi)[t] + DEE[i] @ np.transpose(ze)[t] + noiseE[i][t] - deltaE * re[i][t] - mu * fe[i][t] 

        for i in range(Ni): 
            Vi[i][t + 1] = leakI * Vi[i][t]  + CIE[i] @ np.transpose(fe)[t] - CII[i] @ np.transpose(fi)[t]   + noiseI[i][t] + deltaIE * DIE[i] @ np.transpose(ze)[t] - mu * fi[i][t] - deltaI * ri[i][t]

        activation = []
        for i in range(N):
            activation.append(Ve[i][t + 1] > (thresE  + noiseE[i][t]))
        """if sum(activation) > 0:
            tmp = np.transpose(fe) 
            tmp[t] =  activation
            fe = np.transpose(tmp)
        
        activation = []
        for i in range(Ni):
            activation.append(Ve[i][t + 1] > (thresE  + noiseE[i][t]))
        if sum(activation) > 0:
            tmp = np.transpose(fe) 
            tmp[t] =  activation
            fe = np.transpose(tmp)"""