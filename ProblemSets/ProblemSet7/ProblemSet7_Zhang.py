# Import web scraping tools and other packages
import requests
import json
import ast # this package is to convert string to the object in the string
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

#####################################################
############# Pull Out Data Through API #############
#####################################################
# Get personal income data from BEA
url = "https://www.bea.gov/api/data?\
&UserID=237D3162-F404-43D4-B5AA-A64AC96D31C0&method=GetData\
&datasetname=RegionalIncome&TableName=SA1&GeoFIPS=STATE&LineCode=3\
&Year=All&ResultFormat=JSON&"

response = requests.get(url)
ps7_data = response.text
ps7_data2 = ast.literal_eval(ps7_data) # This just drops the string symbol and only keeps the dictionaries

# The above dic has multiple layers, get the data dic
ps7_data3 = ps7_data2['BEAAPI']['Results']['Data']

# Convert it to dataframe
ps7_data4 = pd.DataFrame(ps7_data3)


##################################################
############# Clean Up The Dataframe #############
##################################################
# Drop unnecessary variables
ps7_data5 = ps7_data4.drop(['CL_UNIT', 'Code', 'GeoFips', 'NoteRef', 'UNIT_MULT'], axis=1)

# Rename variables
ps7_data5.columns = ['PersIncome_per', 'Area', 'Year']
# The unit of PersIncome_per is dollars

# Reorder the variables
ps7_data5 = ps7_data5[['Area', 'Year', 'PersIncome_per']]

# Convert the format of variables
ps7_data5.Year = pd.to_numeric(ps7_data5.Year, errors='coerce')
ps7_data5.PersIncome_per = pd.to_numeric(ps7_data5.PersIncome_per, errors='coerce')

# Remove the * in state names
ps7_data5['Area'] = ps7_data5['Area'].str.replace('*', '')


#################################################################
############# National Personal Income across Years #############
#################################################################
# Get national aggregated data
ps7_nation = ps7_data5[(ps7_data5.Area == 'United States')]

# Plot Fiure 1
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.set(xlabel='Year') # plot title, axis labels
ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)
ax.plot(ps7_nation.Year, ps7_nation['PersIncome_per'], 'b')
ax.set_ylabel('Personal Income per Capita\n (Dollars)', color='b')
ax.tick_params('y', colors='b')

fig.savefig('figure1.png', transparent=False, dpi=80, bbox_inches="tight")


#################################################################
############# Regional Personal Income across Years #############
#################################################################
# Get Regional Aggregate data
ps7_region = ps7_data5[(ps7_data5.Area == 'Great Lakes')| (ps7_data5.Area == 'Plains')|(ps7_data5.Area == 'Rocky Mountain')|
                       (ps7_data5.Area == 'New England')|(ps7_data5.Area == 'Mideast')| (ps7_data5.Area == 'Southeast')|
                       (ps7_data5.Area == 'Southwest')|(ps7_data5.Area == 'Far West')]

# Use pivot to plot Figure2
ps7_region.pivot(index = 'Year', columns = 'Area', values='PersIncome_per').plot(kind="line")

# Set labels and title
plt.xlabel('Year')
plt.ylabel('Personal Income per Capita\n (Dollars)', color = 'b')
plt.tick_params('y', colors='b')
fig2 = plt.gcf()

fig2.savefig('figure2.png', transparent=False, dpi=80, bbox_inches="tight")


###################################################################
############# California Personal Income across Years #############
###################################################################
# Get California data
ps7_ca = ps7_data5[(ps7_data5.Area == 'California')]

# Plot Figure3
plt.style.use('ggplot')
fig, ax = plt.subplots()
plt.plot(ps7_ca['Year'], ps7_ca['PersIncome_per'], axes=ax)
ax.set_xlim([1929, 2016])
ax.set_xlabel(xlabel='Year')
ax.set_ylabel('California\n Personal Income per Capita\n (Dollars)', color='b')
ax.tick_params('y', colors='b')
ax.axvline(x=1970, color='k', linestyle='--')
ax.annotate('1970\n Intra-state\n Banking \n Deregulation', xy=(1950,40000), xytext=(1950, 45000))
ax.axvline(x=1987, color='b', linestyle='--')
ax.annotate('1987\n Inter-state\n Banking \n Deregulation', xy=(1987,40000), xytext=(1987, 45000))

# save figure
fig.savefig('figure3.png', transparent=False, dpi=80, bbox_inches="tight")
