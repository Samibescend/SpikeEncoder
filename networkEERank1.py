import matplotlib.pyplot as plt
import math
import numpy as np
from stimFun import stimFun
from wFun import wFun
from netFun import netFun
import seaborn

savefig=0

stim_type=1
legend_on=0
# parameters

M=3;                                   # number of encoded features        
N=400;                                 # number of E neurons   

nsec=1;                                # duration of the trial in seconds 

tau_e=10;                              # time constant of the excitatory estimate  
tau_i=5;                               # time constant inhibitory estimate 

tau_re=20;                             # time constant of the low-pass filtered spike train in E neurons
tau_ri=10;                             # time constant of the low-pass filtered spike train in I neurons 
   
a_mu=1.0;                              # sets the strength of the regularizer     
a_sigma=1.5;                           # sets the noise intensity 
mu=a_mu*math.log(N);                        # constant of the quadratic regularizer
sigmav=a_sigma/math.log(N)                 # noise intensity spike generation

dt=0.02;                               # time step in ms     
q=4;                                   # ratio of E to I neurons
d=3;                                   # ratio of standard deviations for I to E decoding weights 

#set decoding weights w and the input s(t), compute connectivity weights C and D and the signal x(t)

b1= np.array([[0.5, 0, 0.3]])
aB=0.035
B= ((b1.T) @ b1) * aB
print(B)

T = int((nsec * 1000) / dt)


s = stimFun(M, T, stim_type)


w, C, D = wFun(M, N, q, d, B)
print(np.array(C)[0])

tau_vec = [1, tau_e, tau_i, tau_re, tau_ri]

if tau_e < tau_re:
    print("adaptation in E neurons")
elif tau_e > tau_re:
    print("facilitation in E neurons")
else:
    print("no local currents in E neurons")


if tau_i < tau_ri:
    print("adaptation in I neurons")
elif tau_i > tau_ri:
    print("facilitation in I neurons")
else:
    print("no local currents in I neurons")

print(np.array_equal(B, np.transpose(B)))

""""
e = np.linalg.eig(B)
print(e)
tol = len(e) * np.finfo(float).eps #DOUTE
print(tol)


isSemiDef = np.all(e > -tol)
print(isSemiDef + " is B positive semi-definite")
"""

#Simulate network activity

fe, fi, xhatE, xhatI, re, ri  = netFun(dt, sigmav, mu, tau_vec, s, w, C, D)

print(re[0])
print("over")


with open(r"./xhat_e.txt", "w") as doc:
    for n in np.transpose(xhatE)[0][0]:
        doc.write(str(n) + "\n")

#Plot input, spikes and pop. Firing rate


""""
fig, axes = plt.subplots(1, 1, figsize=(12, 7))
axes.matshow(np.transpose(fe), cmap = "binary")
plt.tight_layout()
plt.savefig("spikesHistory.jpg")"""

print("spike history fe")
palette = {0 : "white", 1 : "black"}
test = seaborn.heatmap(np.transpose(fe), cmap="rocket_r", cbar=False)
plt.savefig("spikesHistory.jpg")

print("spikeHistory fi")
test = seaborn.heatmap(np.transpose(fi), cmap="rocket_r", cbar=False)
plt.savefig("spikesHistoryFi.jpg")

print("spikeHistory re")
test = seaborn.heatmap(np.transpose(re), cmap="rocket_r", cbar=False)
plt.savefig("spikesHistoryRe.jpg")

print("spikeHistory ri")
test = seaborn.heatmap(np.transpose(ri), cmap="rocket_r", cbar=False)
plt.savefig("spikesHistoryRi.jpg")


print("spikeHistory xhatE")
test = seaborn.heatmap(np.transpose(xhatE), cmap="rocket_r", cbar=False)
plt.savefig("spikesHistoryxhatE.jpg")

print("spikeHistory xhatI")

test = seaborn.heatmap(np.transpose(xhatI), cmap="rocket_r", cbar=False)
plt.savefig("spikesHistoryxhatI.jpg") 
