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

    noiseE = sigmav * np.random.randn(T, N) * np.sqrt(2 * dt)
    noiseI = sigmav * np.random.randn(T, Ni) * np.sqrt(2 * dt)

    ffe = np.transpose(w[0]) @ s * dt

    ##Integration

    Ve = np.zeros((T, N))  # membrane potential
    fe = np.zeros((T, N))  # spike train
    re = np.zeros((T, N))  # filtered spike train
    ze = np.zeros((T, N))  # filtered spike train
    xhatE = np.zeros((T, M, N)) #excitatory estimate

    Vi = np.zeros((T, Ni))  # membrane potential
    fi = np.zeros((T, Ni))  # spike train
    ri = np.zeros((T, Ni))  # filtered spike train
    xhatI = np.zeros((T, M, Ni)) #excitatory estimate


    Ve[0]  = np.random.randn(N) * 2 - 12  #initialisation with random membrane potentials 
    Vi[0]  = np.random.randn(Ni) * 2 - 12


    for t in range(T - 1):

        #E neurons
        Ve[t + 1] = leakE * Ve[t] + np.transpose(ffe)[t] - CEI @ fi[t]  + DEE @ (ze)[t] + noiseE[t] - deltaE * re[t] - mu * fe[t]
        Vi[t + 1] = leakI * Vi[t]  + CIE @ (fe)[t] - CII @ (fi)[t]   + noiseI[t] + deltaIE * DIE @ (ze)[t] - mu * fi[t] - deltaI * ri[t]

        activation = []

        for i in range(N):
            if Ve[t + 1][i] > (thresE[i]  + noiseE[t][i]):
                activation.append(1)
            else:
                activation.append(0)
        fe[t + 1] = activation
        activation = []     
        for i in range(Ni):
            if Vi[t + 1][i] > (thresI[i]  + noiseI[t][i]):
                activation.append(1)
            else:
                activation.append(0)
        fi[t + 1] = activation    
        ze[t + 1] = (1 - lambdaE * dt) * ze[t] + fe[t]
        re[t + 1] = (1 - alphaE * dt) * re[t] + fe[t]
        ri[t + 1] = (1 - alphaI * dt) * ri[t] + fi[t]
        for i in range(M):
            xhatE[t + 1][i] = ((1 - lambdaE * dt) * xhatE[t][i]) + (w[0][i] * fe[t + 1])
            xhatI[t + 1][i] = (1 - lambdaI * dt) * xhatI[t][i] + (w[1][i] * fi[t + 1])

    return fe, fi, xhatE, xhatI, re, ri