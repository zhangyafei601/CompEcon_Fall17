# import packages we'll use here
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

def distance_calc1 (row):
    start = (row['buyer_lat'], row['buyer_long'])
    stop = (row['target_lat'], row['target_long'])
    return np.log(vc(start, stop).miles)

# Calculate the variables for observed matches
ps4_data['var1'] = ps4_data['num_stations_buyer'] * ps4_data['pop_ths_log']
ps4_data['var2'] = ps4_data['corp_owner_buyer'] * ps4_data['pop_ths_log']
ps4_data['var3'] = ps4_data.apply (lambda row: distance_calc1 (row),axis = 1)

# Create dataframes for different years
ps4_data_2007 = ps4_data[(ps4_data['year'] == 2007)].copy()
ps4_data_2007['index'] = ps4_data_2007['buyer_id'] - 1 # This creates my own index to fix the indexing/location problem
ps4_data_2007 = ps4_data_2007.set_index('index')
ps4_data_2008 = ps4_data[(ps4_data['year'] == 2008)].copy()
ps4_data_2008['index'] = ps4_data_2008['buyer_id'] - 1
ps4_data_2008 = ps4_data_2008.set_index('index')

# Define a function to calculate the distance
def distance_calc (data,row1,row2):
    start = (data.iloc[row1, 3], data.iloc[row1, 4])
    stop = (data.iloc[row2, 5], data.iloc[row2, 6])
    return np.log(vc(start, stop).miles)

def payoff(data):
    # Define some arrays to store the output numbers
    np_temp1 = np.zeros(6, dtype=np.int).reshape(1,6)
    np_temp2 = np.zeros(3, dtype=np.int).reshape(1,3)
    np_temp3 = np.zeros(3, dtype=np.int).reshape(1,3)
    for b in data['buyer_id']:
        for t in data['target_id']:
            if b < t:
                ob1 = data['var1'][b - 1]
                ob2 = data['var2'][b - 1]
                ob3 = data['var3'][b - 1]
                ob4 = data['var1'][t - 1]
                ob5 = data['var2'][t - 1]
                ob6 = data['var3'][t - 1]
                np_temp1 = np.vstack([np_temp1, [ob1, ob2, ob3, ob4, ob5, ob6]])
                # This returns the six variables on the left hand side of the inequalities (observed matches)

                cf1 = data['num_stations_buyer'][b - 1] * data['pop_ths_log'][t - 1]
                cf2 = data['corp_owner_buyer'][b - 1] * data['pop_ths_log'][t - 1]
                cf3 = distance_calc(data, b-1, t-1)
                np_temp2 = np.vstack([np_temp2, [cf1, cf2, cf3]])
                # This returns the three variables of the first part of the right hand side (counterfatual matches)

            if b > t:
                cf4 = data['num_stations_buyer'][b - 1] * data['pop_ths_log'][t - 1]
                cf5 = data['corp_owner_buyer'][b - 1] * data['pop_ths_log'][t - 1]
                cf6 = distance_calc(data, b-1, t-1)
                np_temp3 = np.vstack([np_temp3, [cf4, cf5, cf6]])
                # This returns the other three variables of the second part of the right hand side (counterfactual matches)

    # Drop the first row of the array
    np_temp1 = np.delete(np_temp1, 0, 0)
    np_temp2 = np.delete(np_temp2, 0, 0)
    np_temp3 = np.delete(np_temp3, 0, 0)
    # Combine all the variables (stored in arrays) to one dataframe
    ps4_mse = pd.DataFrame({'ob1':np_temp1[:,0], 'ob2':np_temp1[:,1], 'ob3':np_temp1[:,2],
                            'ob4':np_temp1[:,3], 'ob5':np_temp1[:,4], 'ob6':np_temp1[:,5],
                            'cf1':np_temp2[:,0], 'cf2':np_temp2[:,1], 'cf3':np_temp2[:,2],
                            'cf4':np_temp3[:,0], 'cf5':np_temp3[:,1], 'cf6':np_temp3[:,2]})

    return ps4_mse

# Write indicator function
def mse(coefs):
    std_coef, alpha, beta = coefs
    #total = 0
    for i in ps4_mse_both.index:
        indicator = (std_coef * ps4_mse_both['ob1'] + alpha * ps4_mse_both['ob2'] + beta * ps4_mse_both['ob3'] +
                  std_coef * ps4_mse_both['ob4'] + alpha * ps4_mse_both['ob5'] + beta * ps4_mse_both['ob6'] >=
                  std_coef * ps4_mse_both['cf1'] + alpha * ps4_mse_both['cf2'] + beta * ps4_mse_both['cf3'] +
                  std_coef * ps4_mse_both['cf4'] + alpha * ps4_mse_both['cf5'] + beta * ps4_mse_both['cf6'])
        total = -1 * sum(indicator)
        return total

# Append the two dataframes together
ps4_mse_2007 = payoff(ps4_data_2007)
ps4_mse_2008 = payoff(ps4_data_2008)
together = [ps4_mse_2007, ps4_mse_2008]
ps4_mse_both = pd.concat(together, ignore_index=True)

params_initial = [1,10,10]
mse_results = opt.minimize(mse, params_initial, method='Nelder-Mead', tol = 1e-12, options={'maxiter': 5000})

mse_results
