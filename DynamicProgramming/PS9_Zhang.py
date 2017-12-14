# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:27:34 2017

@author: yafei
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:30:30 2017

@author: yafei
"""
import scipy.optimize as opt
import numpy as np
import numba
    
def main(theta):
    ### import packages
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.stats import norm
    import scipy.integrate as integrate
    import scipy.optimize as opt
    import time
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import numba
    
    
    
    ### Define functions
    # Define a function to get aggregate value
    @numba.jit
    def Agg(matA, matB):
        aggregate = (np.multiply(matA, matB)).sum()
        return aggregate
    
    # Define a function of VF_loop
    @numba.jit
    def VFI_loop(V, e, betafirm, sizez, sizek, Vmat, pi):
        V_prime = np.dot(pi, V)
        for i in range(sizez):  # loop over z
            for j in range(sizek):  # loop over k
                for k in range(sizek): # loop over k'
                    Vmat[i, j, k] = e[i, j, k] + betafirm * V_prime[i, k]
        return Vmat
    
    # Define a function of SD_loop
    @numba.jit
    def SD_loop(PF, pi, Gamma, sizez, sizek):
        HGamma = np.zeros((sizez, sizek))
        for i in range(sizez):  # z
            for j in range(sizek):  # k
                for m in range(sizez):  # z'
                    HGamma[m, PF[i, j]] = \
                        HGamma[m, PF[i, j]] + pi[i, m] * Gamma[i, j]
        return HGamma
    
    ### set our parameters
    rho = 0.7605
    mu = 0.0
    sigma_eps = 0.213
    alpha_k = 0.297
    alpha_l = 0.650
    delta = 0.154
    psi = 1.08
    r = 0.04
    h = 6.616
    betafirm = (1 / (1 + r))
    
    w = 0.7
    
    Nf = 1000
    T = 50
    
    alpha_k, psi, rho, sigma_eps = theta
    ### Get grid points of z (productivity shocks)
    
    # Call rouwen function to get z grids
    
    # draw our shocks
    num_draws = 100000 # number of shocks to draw
    eps = np.random.normal(0.0, sigma_eps, size=(num_draws))
    
    # Compute z
    z = np.empty(num_draws)
    z[0] = 0.0 + eps[0]
    for i in range(1, num_draws):
        z[i] = rho * z[i - 1] + (1 - rho) * mu + eps[i]
    # Remember here z is actually log(z)
    sigma_z = sigma_eps / ((1 - rho ** 2) ** (1 / 2))
    
    # Compute cut-off values
    N = 9  # number of grid points
    z_cutoffs = (sigma_z * norm.ppf(np.arange(N + 1) / N)) + mu
    
    # compute grid points for z
    z_grid = np.exp((((N * sigma_z * (norm.pdf((z_cutoffs[:-1] - mu) / sigma_z)
                                  - norm.pdf((z_cutoffs[1:] - mu) / sigma_z)))
                  + mu)))
    # Now z_grid is z rather than log(z)
    
    
    
    ### Simulate transition matrix
    # define function that we will integrate over
    def integrand(x, sigma_z, sigma_eps, rho, mu, z_j, z_jp1):
        val = (np.exp((-1 * ((x - mu) ** 2)) / (2 * (sigma_z ** 2)))
                * (norm.cdf((z_jp1 - (mu * (1 - rho)) - (rho * x)) / sigma_eps)
                   - norm.cdf((z_j - (mu * (1 - rho)) - (rho * x)) / sigma_eps)))
    
        return val
    
    # compute transition probabilities
    pi = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            results = integrate.quad(integrand, z_cutoffs[i], z_cutoffs[i + 1],
                                     args = (sigma_z, sigma_eps, rho, mu,
                                             z_cutoffs[j], z_cutoffs[j + 1]))
            pi[i,j] = (N / np.sqrt(2 * np.pi * sigma_z ** 2)) * results[0]
    
    
    ### Value function iteration
    z = 1
    ### Get grid points for k (capital stock)
    dens = 5
    # put in bounds here for the capital stock space
    kstar = ((((1 / betafirm - 1 + delta) * ((w / alpha_l) ** (alpha_l / (1 - alpha_l)))) /
             (alpha_k * (z ** (1 / (1 - alpha_l))))) **
             ((1 - alpha_l) / (alpha_k + alpha_l - 1)))
    kbar = 2*kstar
    lb_k = 0.001
    ub_k = kbar
    krat = np.log(lb_k / ub_k)
    numb = np.ceil(krat / np.log(1 - delta))
    K = np.zeros(int(numb * dens))
    # we'll create in a way where we pin down the upper bound - since
    # the distance will be small near the lower bound, we'll miss that by little
    for j in range(int(numb * dens)):
        K[j] = ub_k * (1 - delta) ** (j / dens)
    kgrid = K[::-1]
    sizek = kgrid.shape[0]
    
    
    ### Value function iteration (solve firm problem)
    # operating profits, op
    sizez = z_grid.shape[0]
    op = np.zeros((sizez, sizek))
    for i in range(sizez):
        for j in range(sizek):
            op[i,j] = ((1 - alpha_l) * ((alpha_l / w) ** (alpha_l / (1 - alpha_l))) *
          ((kgrid[j] ** alpha_k) ** (1 / (1 - alpha_l))) * (z_grid[i] ** (1/(1 - alpha_l))))
    
    # firm cash flow, e
    e = np.zeros((sizez, sizek, sizek))
    for i in range(sizez):
        for j in range(sizek):
            for k in range(sizek):
                e[i, j, k] = (op[i,j] - kgrid[k] + ((1 - delta) * kgrid[j]) -
                           ((psi / 2) * ((kgrid[k] - ((1 - delta) * kgrid[j])) ** 2)
                            / kgrid[j]))
    
    # Value funtion iteration
    VFtol = 1e-6
    VFdist = 7.0
    VFmaxiter = 3000
    V = np.zeros((sizez, sizek))  # initial guess at value function
    Vmat = np.zeros((sizez, sizek, sizek))  # initialize Vmat matrix
    Vstore = np.zeros((sizez, sizek, VFmaxiter))  # initialize Vstore array
    VFiter = 1
    
    start_time = time.clock()
    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = V
        Vmat = VFI_loop(V, e, betafirm, sizez, sizek, Vmat, pi)
        Vstore[:, :, VFiter] = V.reshape(sizez, sizek,)  # store value function at each
        # iteration for graphing later
        V = Vmat.max(axis=2)  # apply max operator to Vmat (to get V(k))
        PF = np.argmax(Vmat, axis=2)  # find the index of the optimal k'
        Vstore[:,:, i] = V  # store V at each iteration of VFI
        VFdist = (np.absolute(V - TV)).max()  # check distance between value
        # function for this iteration and value function from past iteration
        VFiter += 1
    
    VFI_time = time.clock() - start_time
    if VFiter < VFmaxiter:
        print('Value function converged after this many iterations:', VFiter)
    else:
        print('Value function did not converge')
    print('VFI took ', VFI_time, ' seconds to solve')
    
    VF = V  # solution to the functional equation
    PF = PF    
    
    # Optimal capital stock k'
    optK = kgrid[PF]
    
    # optimal investment I
    optI = optK - (1 - delta) * kgrid
    
    
    ### Simulation
    
    
    # Simulate z grids
    def sim_markov(z_grid, pi, T):
        
        np.random.seed(seed = 42)
        u = np.random.uniform(size=T)
        # Do simulations
        z_discrete = np.empty((T), dtype = np.int)  # this will be a vector of values
        # we land on in the discretized grid for z
        N_zgrid = z_grid.shape[0]
        oldind = int(np.ceil((N_zgrid - 1) / 2))  # set initial value to median of grid
        z_discrete[0] = oldind
        for i in range(1, T):
            sum_p = 0
            ind = 0
            while sum_p < u[i]:
                sum_p = sum_p + pi[ind, oldind]
    #             print('inds =  ', ind, oldind)
                ind += 1
            if ind > 0:
                ind -= 1
            z_discrete[i] = ind
            oldind = ind
    
        return z_discrete
    
    @numba.jit
    def zdraws():
        z_shocks = np.zeros((Nf, T))
        for i in range(N):
            z_shocks[i] = sim_markov(z_grid, pi, T)
            
        return z_shocks.astype(dtype=np.int)
    
    zshocks = zdraws()
    
     
    # simulate K, V, I, and Profits
    @numba.jit
    def k_sim():
        k_sim_loc = np.zeros((Nf, T), dtype = np.int)
        for i in range(Nf):
            for j in range(T-1):
                k_sim_loc[i, j+1] = PF[zshocks[i, j]][k_sim_loc[i, j]]
        return k_sim_loc
    
    k_sim_loc = k_sim()
        
    k_sim = kgrid[k_sim_loc]
    
    
    @numba.jit()
    def I_sim():
        I_sim = np.zeros((Nf, T))
        for i in range(Nf):
            for j in range(T-1):
                I_sim[i, j + 1] = k_sim[i, j+1] - k_sim[i, j] * (1-delta)
        return I_sim
    
    @numba.jit()
    def V_sim():
        V_sim = np.zeros((Nf, T))
        for i in range(Nf):
            for j in range(T):
                V_sim[i, j] = VF[zshocks[i, j]][k_sim_loc[i, j]]
    
        return V_sim
    
    
    @numba.jit()
    def pi_sim():
        pi_sim = np.zeros((Nf, T))
        for i in range(Nf):
            for j in range(T):
                pi_sim[i, j] = z_grid[zshocks[i, j]] * kgrid[k_sim_loc[i, j]] ** alpha_k
    
        return pi_sim
    
    I_sim = I_sim()
    
    V_sim = V_sim()
    
    pi_sim = pi_sim()

    return k_sim, I_sim, V_sim, pi_sim

k_sim, I_sim, V_sim, pi_sim = main(theta)

### Calculate moments from simulated data
def moments(k_sim, I_sim, V_sim, prof_sim):

    IoverK = (I_sim/k_sim).reshape((Nf*50, 1))
    avgQ = (V_sim/k_sim).reshape((Nf*50, 1))
    PIoverK = (prof_sim/k_sim).reshape((Nf*50, 1))

    y = np.matrix(IoverK)
    X = np.matrix(np.concatenate((np.ones((Nf*50, 1)), avgQ, PIoverK), axis = 1))

    a0, a1, a2 = np.linalg.inv(X.T * X) * X.T * y  # reg coeffs

    ik = IoverK.reshape((1, Nf*50))  # I over K reshaped for corrcoef()
    sc_ik = np.corrcoef(ik[0][1:], ik[0][:50000-1])[0,1]  # sc_
    sd_piK = np.std(PIoverK)
    qbar = V_sim.sum() / k_sim.sum()

    U_sim = np.array((a1[0, 0], a2[0, 0], sc_ik, sd_piK, qbar))

    return U_sim


### Calculate distance between data and simulation
def dist(theta):
    
    k_sim, I_sim, V_sim, pi_sim = main(theta)   
    U_sim = moments(k_sim, I_sim, V_sim, pi_sim)
    
    U_data = np.array((.03, .24, .4, .25, 3.0))

    dist = np.matrix((U_data - U_sim)) * np.eye(5) * np.matrix((U_data - U_sim)).T

    return dist[0, 0]

theta_0 = np.array([0.699, 0.1647, 0.111, 0.857])

results = opt.minimize(dist, theta_0, method = 'nelder-mead', options = {'maxiter': 100})