import numpy as np
import scipy.special as sp

def stable_logsumexp(x):
    a = np.max(x)
    return a+np.log(np.sum(np.exp(x-a)))

def log_comb(n, k):
    return (sp.gammaln(n + 1) -sp.gammaln(k + 1) - sp.gammaln(n - k + 1))

########################################################################################################################
################################## 1st Upper bound Theorem1 ############################################################
########################################################################################################################
def RDP_comp_integer_order(eps0, n, q, lambd):
    if lambd == 1:
        D = 0
    else:
        ex_eps0 = np.exp(eps0)
        m = (q*n-1)/(2*ex_eps0)+1
        c = 2*((ex_eps0-1/ex_eps0)**2)/(m)
        a = []
        a.append(0)
        term = log_comb(lambd,2)+2*np.log(q)+np.log(c/2)
        a.append(term)
        for j in range(3,lambd+1,1):
            term = log_comb(lambd, j)+np.log(j)+sp.gammaln(j/2)+(j/2)*np.log(c)+j*np.log(q)
            a.append(term)
        a.append(-(q*n-1)/(8*ex_eps0)+np.log(((1-q+q*np.exp(eps0))**lambd)-1-q*np.exp(eps0)+q))
        D = stable_logsumexp(a)/(lambd-1)
    return D

def RDP_comp(eps0, n, q, lambd):
    D = np.zeros_like(lambd,dtype=float)
    for i in range(0,len(lambd),1):
        c = int(np.ceil(lambd[i]))
        f = int(np.floor(lambd[i]))
        if c==f:
            D[i] = RDP_comp_integer_order(eps0, n, q, f)
        else:
            a = c-lambd[i]
            D[i] = (f-1)*a*RDP_comp_integer_order(eps0, n, q, f)
            D[i] += (c-1)*(1-a)*RDP_comp_integer_order(eps0, n, q, c)
            D[i] /= (lambd[i]-1)
    return D

########################################################################################################################
############################################ Optimize from RDP to DP ###################################################
########################################################################################################################
def optimize_RDP_To_DP(delta,acc,eps0,n,q,T,rdp_fun):
    lmax = np.array(10**3)
    lmin = np.array(1)
    err = lmax
    while (err>acc):
        l = []
        l.append((lmax+lmin)/2)
        l.append((lmax+lmin)/2+0.01)
        D = T*rdp_fun(eps0,n,q,l)+np.log(1-1/np.array(l))-np.log(delta*np.array(l))/(np.array(l)-1)
        err = lmax-lmin
        if D[0]>D[1]:
            lmin = l[0]
            eps = D[1]
        else:
            lmax = l[0]
            eps = D[0]
    return eps
    