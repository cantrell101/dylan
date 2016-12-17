import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def BlackScholes(S, K, r, v, q, T):
    N = norm.cdf
    d1 = (np.log(S/K) + (r - q - 0.5 * v * v) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    callPrice = S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)

    return callPrice


def GeometricAsianCall(S, K, r, v, q, T, N):
    # S = spot
    # K = Strike
    # r = rate
    # v = volatility
    # q = dividend
    # T = expiry
    # N = steps`

    dt = T / N
    nu = r - q - 0.5 * v * v
    a = N * (N-1) + (2.0 * N + 1.0) / 6.0
    V = np.exp(-r * T) * S * np.exp(((N+1) * nu / 2.0 + v * v * a / (2.0 * N * N)) * dt)
    vavg = v * np.sqrt(a) / (pow(N, 1.5))
    
    price = BlackScholes(V, K, r, vavg, 0, T)

    return price


def CallPayoff(S, K):
    return np.maximum(S - K, 0.0)


## main 
b = -1.0
S = 100
K = 100.0
r = 0.06
v = 0.20
T = 1.00
q = 0.03
N = 10
M = 1000
dt = T/N

Gstar = GeometricAsianCall(S, K, r, v, q, T, N)
spath = np.zeros((M,N))
spath[:,0] = S
A = np.zeros(M)
G = np.zeros(M)

nudt = (r - q - 0.5 * v * v) * dt
sigsdt = v * np.sqrt(dt)
disc = np.exp(-r * T)
z = np.random.normal(size=(M,N))

for i in range(M):
    for j in range(1, N):
        spath[i,j] = spath[i,j-1] + np.exp(nudt + sigsdt * z[i,j])

    amean = spath[i].mean()
    gmean = pow(spath[i].prod(), 1/N)
    A[i] = CallPayoff(amean, K)
    G[i] = CallPayoff(gmean, K)

callPrice = disc * A.mean() + b * (Gstar - G.mean())
fmt = "The Fixed Strike Arithmetic Asian Call Price is: {0:0.3f}"
print(fmt.format(callPrice))

def graph(CallPayoff, S):
    S = []
    
    while S <= K:
        CallPayoff = 0
    else:
        CallPayoff = CallPayoff
        while True:
            fig = plt.figure()
            axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
            axes.set_xlabel('Payoff')
            axes.set_ylabel('Spot Price')
            x = CallPayoff
            y = S.int([0,125])
            axes.plot(x, y)
            plt.grid(True)
            plt.show()
