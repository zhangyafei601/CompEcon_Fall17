# Import needed packages
import pandas as pd
import numpy as np
import scipy.optimize as opt
import time
from scipy.optimize import differential_evolution
from geopy.distance import vincenty as vc

# Read in the data
ps4_data = pd.read_excel('radio_merger_data.xlsx')

# Scale variables
ps4_data['pop_ths_log'] = np.log(ps4_data['population_target'] / 1000)
ps4_data['price_ths_log'] = np.log(ps4_data['price'] / 1000)
ps4_data['num_stations_log'] = np.log(1 + ps4_data['num_stations_buyer'])
ps4_data['hhi_log'] = np.log(ps4_data['hhi_target'])

# Define a function to calculate the distance of observed matches
def distance_calc1 (row):
    start = (row['buyer_lat'], row['buyer_long'])
    stop = (row['target_lat'], row['target_long'])
    return np.log(vc(start, stop).miles)

# Calculate the variables for observed matches
ps4_data['var1'] = ps4_data['num_stations_log'] * ps4_data['pop_ths_log']
ps4_data['var2'] = ps4_data['corp_owner_buyer'] * ps4_data['pop_ths_log']
ps4_data['var3'] = ps4_data.apply (lambda row: distance_calc1 (row),axis = 1)

# Create dataframes for different years
ps4_data_2007 = ps4_data[(ps4_data['year'] == 2007)].copy()
ps4_data_2007['index'] = ps4_data_2007['buyer_id'] - 1 # This creates my own index to fix the indexing/location problem
ps4_data_2007 = ps4_data_2007.set_index('index')
ps4_data_2008 = ps4_data[(ps4_data['year'] == 2008)].copy()
ps4_data_2008['index'] = ps4_data_2008['buyer_id'] - 1
ps4_data_2008 = ps4_data_2008.set_index('index')

# Define a function to calculate the distance of counterfactual matches
def distance_calc (data,row1,row2):
    start = (data.iloc[row1, 3], data.iloc[row1, 4])
    stop = (data.iloc[row2, 5], data.iloc[row2, 6])
    return np.log(vc(start, stop).miles)

# Define the function to calculate all the variables used in the payoff function
def payoff(data):
    # Define some arrays to store the output numbers
    np_temp1 = np.zeros(10, dtype=np.int).reshape(1,10)
    np_temp2 = np.zeros(5, dtype=np.int).reshape(1,5)
    np_temp3 = np.zeros(5, dtype=np.int).reshape(1,5)
    for b in data['buyer_id']:
        for t in data['target_id']:
            if b < t:
                ob1 = data['var1'][b - 1]
                ob2 = data['var2'][b - 1]
                ob3 = data['var3'][b - 1]
                ob4 = data['var1'][t - 1]
                ob5 = data['var2'][t - 1]
                ob6 = data['var3'][t - 1]
                # This returns the six variables on the left hand side of the inequalities (observed matches)

                ob7 = data['hhi_log'][b - 1]
                ob8 = data['price_ths_log'][b - 1]
                ob9 = data['hhi_log'][t - 1]
                ob10 = data['price_ths_log'][t - 1]
                # This returns two additional variables in model2 (the transfered model)
                np_temp1 = np.vstack([np_temp1, [ob1, ob2, ob3, ob4, ob5, ob6, ob7, ob8, ob9, ob10]])
                # This stacks the observations of the above variables

                cf1 = data['num_stations_log'][b - 1] * data['pop_ths_log'][t - 1]
                cf2 = data['corp_owner_buyer'][b - 1] * data['pop_ths_log'][t - 1]
                cf3 = distance_calc(data, b-1, t-1)
                # This returns the three variables of the first part of the right hand side (counterfatual matches)

                cf7 = data['hhi_log'][t - 1]
                cf8 = data['price_ths_log'][t - 1]
                # This returns two additional variables in model2
                np_temp2 = np.vstack([np_temp2, [cf1, cf2, cf3, cf7, cf8]])
                # This stacks the observations of the above variables

            if b > t:
                cf4 = data['num_stations_log'][b - 1] * data['pop_ths_log'][t - 1]
                cf5 = data['corp_owner_buyer'][b - 1] * data['pop_ths_log'][t - 1]
                cf6 = distance_calc(data, b-1, t-1)
                # This returns the other three variables of the second part of the right hand side (counterfactual matches)

                cf9 = data['hhi_log'][t - 1]
                cf10 = data['price_ths_log'][t - 1]
                # This returns two additional variables in model2
                np_temp3 = np.vstack([np_temp3, [cf4, cf5, cf6, cf9, cf10]])
                # This stacks the observations of the above variables


    # Drop the first row of the array (the first row are all zeros)
    np_temp1 = np.delete(np_temp1, 0, 0)
    np_temp2 = np.delete(np_temp2, 0, 0)
    np_temp3 = np.delete(np_temp3, 0, 0)
    # Combine all the variables (stored in arrays) to one dataframe
    ps4_mse = pd.DataFrame({'ob1':np_temp1[:,0], 'ob2':np_temp1[:,1], 'ob3':np_temp1[:,2], 'ob4':np_temp1[:,3], 'ob5':np_temp1[:,4],
                            'ob6':np_temp1[:,5], 'ob7':np_temp1[:,6], 'ob8':np_temp1[:,7], 'ob9':np_temp1[:,8], 'ob10':np_temp1[:,9],
                            'cf1':np_temp2[:,0], 'cf2':np_temp2[:,1], 'cf3':np_temp2[:,2], 'cf7':np_temp2[:,3], 'cf8':np_temp2[:,4],
                            'cf4':np_temp3[:,0], 'cf5':np_temp3[:,1], 'cf6':np_temp3[:,2], 'cf9':np_temp3[:,3], 'cf10':np_temp3[:,4]})

    return ps4_mse

# Append dataframes of two years together
ps4_mse_2007 = payoff(ps4_data_2007) # This calls the function above twice to get dataframes of two years
ps4_mse_2008 = payoff(ps4_data_2008)
together = [ps4_mse_2007, ps4_mse_2008]
ps4_mse_both = pd.concat(together, ignore_index=True) # I re-index the data

# Write indicator function of model1
def mse(coefs):
    alpha, beta = coefs
    for i in ps4_mse_both.index:
        indicator = (ps4_mse_both['ob1'] + alpha * ps4_mse_both['ob2'] + beta * ps4_mse_both['ob3'] +
                  ps4_mse_both['ob4'] + alpha * ps4_mse_both['ob5'] + beta * ps4_mse_both['ob6'] >=
                  ps4_mse_both['cf1'] + alpha * ps4_mse_both['cf2'] + beta * ps4_mse_both['cf3'] +
                  ps4_mse_both['cf4'] + alpha * ps4_mse_both['cf5'] + beta * ps4_mse_both['cf6'])
        total = -1 * sum(indicator)
        return total

# Write indicator function of model2 (the transfered model)
def mse_tansf(coefs):
    sigma, alpha, gamma, beta = coefs
    for i in ps4_mse_both.index:
        indicator = ((sigma * ps4_mse_both['ob1'] + alpha * ps4_mse_both['ob2'] + beta * ps4_mse_both['ob3'] +
                      gamma * ps4_mse_both['ob7'] - ps4_mse_both['ob8'] >=
                      sigma * ps4_mse_both['cf1'] + alpha * ps4_mse_both['cf2'] + beta * ps4_mse_both['cf3'] +
                      gamma * ps4_mse_both['cf7'] - ps4_mse_both['cf8']) &
                     (sigma * ps4_mse_both['ob4'] + alpha * ps4_mse_both['ob5'] + beta * ps4_mse_both['ob6'] +
                      gamma * ps4_mse_both['ob9'] - ps4_mse_both['ob10'] >=
                      sigma * ps4_mse_both['cf4'] + alpha * ps4_mse_both['cf5'] + beta * ps4_mse_both['cf6'] +
                      gamma * ps4_mse_both['cf9'] - ps4_mse_both['cf10']))
        total = -1 * sum(indicator)
        return total

# Call the minimizer for model1
params_initial = [1, 1]
mse_results = opt.minimize(mse, params_initial, method='Nelder-Mead', tol = 1e-12, options={'maxiter': 5000})

# Call the minimizer for model2
params_initial_transf = [1,1,1,1]
mse_results_transf = opt.minimize(mse_tansf, params_initial_transf, method='Nelder-Mead', tol = 1e-12, options={'maxiter': 5000})

# Display the results
coefs = (['alpha', 'beta'])
for i in range(2):
    print('Estimated ', coefs[i], "in model(1) = ", mse_results['x'][i])

print()

coefs_transf = (['sigma', 'alpha', 'gamma', 'beta'])
for i in range(4):
    print('Estimated ', coefs_transf[i], "in model(2) = ", mse_results_transf['x'][i])
