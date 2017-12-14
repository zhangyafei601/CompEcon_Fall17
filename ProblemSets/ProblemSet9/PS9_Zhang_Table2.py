# importing necessary packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time
from matplotlib import cm
import numba

import numpy as np
import scipy.stats as st
from scipy.stats import norm
import scipy.integrate as integrate
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
#from scipy.optimize import anneal
from numpy import inf
import ar1_approx as ar1

# Setting the initial values of all parameters from PS8

rho = 0.7605
mu = 0.0
sigma_eps = 0.213
alpha_k = 0.297 # Here, alpha_k parameter refers to alpha in the paper.
#alpha_l = 0.650 # Here, alpha_l parameter is redundant since production function depends only on physical capital.
delta = 0.154
psi = 1.08 # Here, the adjustment cost parameter psi refers to gamma in the paper.
r = 0.04
#h = 6.616 # Here, h is redundant since we are solving only partial equilibrium (i.e. firms' problem).
betafirm = (1 / (1 + r))
num = 9
#w = 0.7 # Here, initial wage rate is redundant since we are solving only partial equilibrium (i.e. firms' problem).
sizez = 9 # Here, z refers to the productivity A in the paper.
num_draws = 100 # same with T, number of periods.
sigma_z = sigma_eps / ((1 - rho ** 2) ** (1 / 2)) #Here, sigma_z refers to the sigma parameter in the paper.
theta = np.array((alpha_k, rho, psi, sigma_z))

##For the firm problem
# Grid for k
dens = 5
kbar = 12 #kstar * 500
lb_k = 0.001
ub_k = kbar
krat = np.log(lb_k / ub_k)
numb = np.ceil(krat / np.log(1 - delta))
K = np.empty(int(numb * dens))
for j in range(int(numb * dens)):
    K[j] = ub_k * (1 - delta) ** (j / dens)
kvec = K[::-1]
sizek = kvec.shape[0]

#z_fun function approximates the process of productivity z, which is A in the paper.
def z_fun(theta):
    # Setting grid for z and the transition matrix
    # Rouwenhorst (1995) method is used to approximate the continuous distribution of shocks with a Markov process.

    alpha_k, rho, psi, sigma_z = theta # Defining all the model parameters as theta.

    num_sigma = 3
    step = (num_sigma * sigma_z) / (sizez / 2)
    Pi, z = ar1.rouwen(rho, mu, step, sizez)
    Pi = np.transpose(Pi)
    z = np.exp(z)  # As AR(1) process is for the natural logarithm of productivity.
    return z, Pi

z, Pi = z_fun(theta)

@numba.jit # Using numba to accelerate the process.
# Defining the same value function iteration likewise in PS8 as a function called VFI_loop.
def VFI_loop(V, e, betafirm, sizez, sizek, Vmat, pi):
    V_prime = np.dot(pi, V)
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            for k in range(sizek): # loop over k'
                Vmat[i, j, k] = e[i, j, k] + betafirm * V_prime[i, k]
    return Vmat

# Defining operating profits, op and earnings, e and doing the value function iteration
def solve_firm(theta):

    alpha_k, rho, psi, sigma_z = theta

    sizez = z.shape[0]
    op = np.zeros((sizez, sizek))
    for i in range(sizez):
        for j in range(sizek):
            op[i,j] = z[i] * (kvec[j]**alpha_k)

    e = np.zeros((sizez, sizek, sizek))
    for i in range(sizez):
        for j in range(sizek):
            for k in range(sizek):
                e[i, j, k] = (op[i,j] - kvec[k] + ((1 - delta) * kvec[j]) -
                            ((psi / 2) * ((kvec[k] - ((1 - delta) * kvec[j])) ** 2)
                            / kvec[j]))


    # Value funtion iteration
    VFtol = 1e-6
    VFdist = 7.0
    VFmaxiter = 3000
    V = np.zeros((sizez, sizek))  # initial guess for value function
    Vmat = np.zeros((sizez, sizek, sizek))  # initialize Vmat matrix
    Vstore = np.zeros((sizez, sizek, VFmaxiter))  # initialize Vstore array
    VFiter = 1

    start_time = time.clock()
    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = V
        Vmat = VFI_loop(V, e, betafirm, sizez, sizek, Vmat, Pi)
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

    VF = V  # Finding solution to the value function iteration


    # Finding the optimal decision rule to simulate data as the next step

    # Obtaining optimal values(functions)
    # Optimal capital stock k' as the policy function, PF
    optK = kvec[PF]

    # Finding the optimal investment I
    optI = optK - (1 - delta) * kvec

    return VF, PF, optK, optI

VF, PF, optK, optI= solve_firm(theta)

# Simulating the Markov process as a function called sim_markov

def sim_markov(z, Pi, num_draws):
    # draw some random numbers on [0, 1]
    u = np.random.uniform(size=num_draws)

    # Doing simulations
    z_discrete = np.empty(num_draws)  # this will be a vector of values
    # discretized grid for z
    N = z.shape[0]
    oldind = 0
    z_discrete[0] = oldind
    for i in range(1, num_draws):
        sum_p = 0
        ind = 0
        while sum_p < u[i]:
            sum_p = sum_p + Pi[ind, oldind]
            ind += 1
        if ind > 0:
            ind -= 1
        z_discrete[i] = ind
        oldind = ind
        z_discrete = z_discrete.astype(dtype = np.int)

    return z_discrete

# Calling the sim_markov function to obtain the simulated values
z_discrete = sim_markov(z, np.transpose(Pi), num_draws)

n = 1000 # Defining the number of firms
T = 100 # Defining the number of time periods
z_new = np.zeros((n,T), dtype = np.int)
for i in range(n):
    z_new[i] = sim_markov(z, np.transpose(Pi), num_draws)  #correct z

# Defining the loop_k function to gather the simulated data values of physical capital in the next period
def loop_k():
    next_k = np.zeros((n,T), dtype = np.int)
    for i in range(n):
        for j in range(T-1):
            next_k[i, j+1] = PF[z_new[i,j]][next_k[i,j]]
    return next_k

next_k=loop_k()
next_k

# Defining the loop_I function to gather the simulated data values of investment
def loop_I():
    next_optI = np.zeros((n,T))
    for i in range(n):
        for j in range(T-1):
            next_optI[i,j+1] = kvec[next_k[i,j+1]] - (1 - delta) * kvec[next_k[i,j]]
    return next_optI

next_optI = loop_I()
next_optI

# Defining the profit function to gather the simulated data values of profit
def profit():
    profit = np.zeros((n, T))
    for i in range(n):
        for j in range(T):
            profit[i,j] = z_new[i,j] * (kvec[next_k[i,j]]**alpha_k)
    return profit

profit1 = profit()
profit1

@numba.jit
# Defining the loop_VFIs function to gather the simulated data values of value function levels
def loop_VFIs():
    next_Vs = np.zeros((n,T), dtype = np.int)
    for i in range(n):
        for j in range(T):
            next_Vs[i,j] = VF[z_new[i,j], next_k[i,j]]
    return next_Vs

next_Vs = loop_VFIs()
next_Vs

# Defining the sim_firm function to gather the simulated data values of all variables listed above for the firms.
def sim_firm(theta):
    alpha_k, rho, psi, sigma_z = theta
    z, Pi = z_fun(theta)
    z_new = np.zeros((n,T), dtype = np.int)
    for i in range(n):
        z_new[i] = sim_markov(z, np.transpose(Pi), num_draws)  #correct z

    next_k = loop_k()
    next_optI = loop_I()
    profit_1 = profit()
    next_Vs = loop_VFIs()

    return next_Vs, profit_1, next_optI, next_k

next_Vs, profit_1, next_optI, next_k = sim_firm(theta)

# Defining the moments function to get the moments of simulated data
def moments(theta):
    alpha_k, rho, psi, sigma_z = theta

    '''This function defines and obtains the moments of simulated data'''

    # Defining the moments
    inv_nextk = next_optI/kvec[next_k]
    invest_nextk = inv_nextk.reshape((1, n*100))  # I over K reshaped for corrcoef()
    sc_invest_nextk = np.corrcoef(invest_nextk[0][1:], invest_nextk[0][:100000-1])[0,1]
    profit_nextk = profit_1/kvec[next_k]
    sd_profit_nextk = profit_nextk.reshape((1, n*100))
    sd_profit_nextk = np.std(sd_profit_nextk)
    q_bar = next_Vs.sum()/kvec[next_k].sum()
    Y = invest_nextk
    Y = Y.reshape(100000, 1)
    q = next_Vs/kvec[next_k]
    q = q.reshape(100000, 1)
    prof_k = profit_1/kvec[next_k]
    prof_k = prof_k.reshape(100000, 1)
    cons = np.ones(100000)
    cons = cons.reshape(100000, 1)

    # Running the regressions
    x = np.hstack((cons, q, prof_k))
    trans = x.transpose()
    xx = np.dot(trans, x)
    inv = np.linalg.inv(xx)
    xy = np.dot(trans, Y)
    reg_coef = np.dot(inv, xy)
    mew_s = np.array([max(reg_coef[1]), max(reg_coef[2]), sc_invest_nextk, sd_profit_nextk, q_bar])  #Mu_s

    return mew_s

# Finding the values for the moments of simulated data (mus_s) by employing the moment values of actual data (mu_d) from the
# second row of panel b in Table 2.

guess=(1,1,1,1)
mu_d=np.array([0.045,0.24,0.040,0.250,2.96])
mu_s=moments(theta)
W=np.eye(len(mu_d))

# Calculating the distance between mu_s and mu_d by defining the dist function and
# using the identity matrix as the weighting matrix such that W=np.eye(len(mu_d))
def dist(A,B,C):
    d = np.linalg.multi_dot([np.transpose(A - B),np.linalg.inv(C),(A - B)])
    return d

# Defining the objective function for simulated method of moments in order to minimize it by using theta as its argument.

def Qfunc2(theta):
    alpha_k, rho, psi, sigma_z = theta
    a1 = 0.045  #these values are from table 2, row 4
    a2=0.24
    sc=0.04
    std=0.25
    qbar=2.96
    mu_s=moments(theta)
    mu_d=np.array([a1,a2,sc,std,qbar])
    W=np.eye(len(mu_d))
    dis=dist(mu_s,mu_d,W)
    return dis

dist(mu_s, mu_d, W)

# Setting bounds for alpha_k, rho, psi, sigma_z
bnds2=((0,1),(0,0.2),(0,0.2),(0,0.9))

# Finding the global minimum of a multivariate function
theta_hat=differential_evolution(Qfunc2,bnds2)

theta_hat.x #coefficients at table 2

epsilon = 0.1
#new theta with epsilon
theta_new = theta + epsilon

# Finding the global minimum of a multivariate function
theta_new_hat=differential_evolution(Qfunc2,bnds2)

theta_new_hat.x #Finding the var-cov matrix

numerator = theta_new_hat.x - theta_hat.x
numerical_derivative = np.mat([numerator/epsilon])
standard_errors = np.dot(numerical_derivative.T,numerical_derivative)
np.diag(standard_errors) # Finding the standard errors.

# Printing results
print(theta_hat.x)
print(np.diag(standard_errors))