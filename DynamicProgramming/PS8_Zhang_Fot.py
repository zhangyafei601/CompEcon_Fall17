# import packages
#import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.optimize as opt

# to print plots inline
#get_ipython().run_line_magic('matplotlib', 'inline')

# specify parameters
rho = 0.7605
mu = 0.0
sigma_eps = 0.213
num_draws = 100000
alpha_k = 0.297
alpha_l = 0.650
delta = 0.154
psi = 1.08
r = 0.04
z = 1
betafirm = (1 / (1 + r))
dens = 5

params = (rho, mu, sigma_eps, num_draws, alpha_k, alpha_l, delta, psi, r, z, betafirm, dens)

# Define distance function
def dis(w, *params):
    
    # Import packages
    #import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import scipy.optimize as opt
    import numpy as np
    from scipy.stats import norm    
    import scipy.integrate as integrate
    import numpy as np
    #import matplotlib.pyplot as plt
    import time
    
    rho, mu, sigma_eps, num_draws, alpha_k, alpha_l, delta, psi, r, z, betafirm, dens = params
    # set our parameter
    # draw our shocks
    # number of shocks to draw
    eps = np.random.normal(0.0, sigma_eps, size=(num_draws))

    # Compute z
    z = np.empty(num_draws)
    z[0] = 0.0 + eps[0]
    for i in range(1, num_draws):
        z[i] = rho * z[i - 1] + (1 - rho) * mu + eps[i]
    # Remember here z is acually log(z)

    # plot distribution of z
    # sns.distplot(z, hist=False)
    sns.kdeplot(np.array(z), bw=0.5)

    # theory says:
    sigma_z = sigma_eps / ((1 - rho ** 2) ** (1 / 2))
    print('Theoretical sigma_z = ', sigma_z)

    

    # Compute cut-off values
    N = 9  # number of grid points (will have one more cut-off point than this)
    z_cutoffs = (sigma_z * norm.ppf(np.arange(N + 1) / N)) + mu
    print('Cut-off values = ', z_cutoffs)

    # compute grid points for z
    # Now z_grid is z rather than z_log
    z_grid = np.exp((((N * sigma_z * (norm.pdf((z_cutoffs[:-1] - mu) / sigma_z)
                                  - norm.pdf((z_cutoffs[1:] - mu) / sigma_z)))
                  + mu)))
    print('Grid points = ', z_grid)

    # Transition matrix
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

    print('Transition matrix = ', pi)
    print('pi sums = ', pi.sum(axis=0), pi.sum(axis=1))

    # Calculate the expectation of z'
    exp_z_prime = np.dot(pi, z_grid)
    exp_z_prime

    # Using SS capital to get the grids of capital
    # to print plots inline
    #get_ipython().run_line_magic('matplotlib', 'inline')


    # put in bounds here for the capital stock space
    kstar = ((((1 / betafirm - 1 + delta) * ((w / alpha_l) **
                                             (alpha_l / (1 - alpha_l)))) /
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

    k_linear = np.linspace(lb_k, ub_k, num=sizek)
    plt.scatter(k_linear, kgrid)


    # # Value function iteration to get policy function of K prime
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

   # np.shape(e)

    # Value funtion iteration
    VFtol = 1e-6
    VFdist = 7.0
    VFmaxiter = 3000
    V = np.zeros((sizez, sizek))  # initial guess at value function
    Vmat = np.zeros((sizez, sizek, sizek))  # initialize Vmat matrix
    Vstore = np.zeros((sizez, sizek, VFmaxiter))  # initialize Vstore array
    VFiter = 1


    #np.shape(Vstore)
    
    start_time = time.clock()
    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = V
        for i in range(sizez):  # loop over z
            for j in range(sizek):  # loop over k
                for k in range(sizek): # loop over k'
                    Vmat[i, j, k] = e[i, j, k] + betafirm * np.dot(pi, V[i, k])
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

    #PF
    
    # Pull out k'


    optK = kgrid[PF]

    #np.shape(optK)

    optK

    ## plot the SD

    # import packages
    #import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D
    #from matplotlib import cm

    # to print plots inline
    #get_ipython().run_line_magic('matplotlib', 'inline')

    # Stationary distribution in 3D

    #np.shape(VF)

    # Calculate the investment function

    optI = optK - (1 - delta) * kgrid

    #np.shape(optI)

    # Pull out Cash Flow function ( function e)

    #np.shape(e)

    # Calculate Labor Demand

    optLD = np.zeros((sizez, sizek))
    for i in range(sizez):
        for j in range(sizek):
            optLD[i,j] = (((alpha_l / w) ** (1 / (1 - alpha_l))) *
          ((kgrid[j] ** alpha_k) ** (1 / (1 - alpha_l))) * (z_grid[i] ** (1/(1 - alpha_l))))



    # Find Stationary Distribution


    '''
    ------------------------------------------------------------------------
    Compute the stationary distribution of firms over (k, z)
    ------------------------------------------------------------------------
    SDtol     = tolerance required for convergence of SD
    SDdist    = distance between last two distributions
    SDiter    = current iteration
    SDmaxiter = maximium iterations allowed to find stationary distribution
    Gamma     = stationary distribution
    HGamma    = operated on stationary distribution
    ------------------------------------------------------------------------
    '''
    Gamma = np.ones((sizez, sizek)) * (1 / (sizek * sizez))
    SDtol = 1e-12
    SDdist = 7
    SDiter = 0
    SDmaxiter = 1000
    while SDdist > SDtol and SDmaxiter > SDiter:
        HGamma = np.zeros((sizez, sizek))
        for i in range(sizez):  # z
            for j in range(sizek):  # k
                for m in range(sizez):  # z'
                    HGamma[m, PF[i, j]] =                     HGamma[m, PF[i, j]] + pi[i, m] * Gamma[i, j]
        SDdist = (np.absolute(HGamma - Gamma)).max()
        Gamma = HGamma
        SDiter += 1

    if SDiter < SDmaxiter:
        print('Stationary distribution converged after this many iterations: ',
              SDiter)
    else:
        print('Stationary distribution did not converge')


    #np.shape(Gamma)


    # Aggregate Labor Demand


    optALD = (np.multiply(optLD, Gamma)).sum()

    #optALD

    # Aggregate Investment

    optAI = (np.multiply(optI, Gamma)).sum()

    #optAI
    # Aggregate Adjustment Cost
    optADJC = psi/2 * np.multiply((optI)**2, 1/kgrid)

    optAADJC = (np.multiply(optADJC, Gamma)).sum()

    #optAADJC

    # Aggregate Output
    optY = np.multiply(np.multiply((optLD) ** alpha_l, kgrid ** alpha_k),np.transpose([z_grid]))
    #np.shape(optY)
    optAY = (np.multiply(optI, Gamma)).sum()
    #optAY

    # Calculate Aggregate Consumption
    optCON = optAY - optAI - optAADJC
    #optCON

    # Calculate Aggregate Labor Supply(using the FOC of household)
    # Set parameter value
    h = 6.616
    optALS = w/(h * optCON)
    #optALS

    return optALS - optALD

# Call the minimizer
# Minimize with Nelder-Mead
w_min = opt.minimize(dis, x0=0.7, method='Nelder-Mead', args = params,
                     tol=1e-15, options={'maxiter': 5000})
print('The minimum of distance found numerically is ', w_min['w'])
