import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style

###############################################
############# READ AND CLEAN DATA #############
###############################################

# Read in data from Excel workbook
ps5_data = pd.read_excel('all_prvtplace.xlsx',sheetname="Sheet1", header=0, skiprows=7, usecols=(0, 12, 13, 15))

# Rename some variables
ps5_data = ps5_data.rename(columns={'All Transactions Announced Date':'date_announce',
                                   'State of Incorporation [Target/Issuer]': 'issuer_state',
                                   'Aggregated Amount Raised ($USDmm, Historical rate)': 'raised_amt',
                                   'Round Number': 'round_num'})

# Convert object to numeric
ps5_data['raised_amt'] = pd.to_numeric(ps5_data['raised_amt'], errors='coerce')
ps5_data['Round'] = pd.to_numeric(ps5_data['round_num'], errors='coerce')

# Create year variable from date
ps5_data['year'] = ps5_data['date_announce'].dt.year

# Sample selection
ps5_data = ps5_data[(ps5_data.raised_amt >0)]


################################################
############# PLOT FIGURES 1 AND 2 #############
################################################

# Compute year level data
year_ps5 = pd.DataFrame({'raised_amt' : ps5_data.groupby('year').apply(lambda x: np.sum(x['raised_amt'])),
                        'count': ps5_data.groupby('year').apply(lambda x: np.ma.count(x['raised_amt']))})

# Plot figure 1
plt.style.use('ggplot')
fig1, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set(xlabel='Year') # plot title, axis labels
ax1.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)
ax1.axvline(x = 1996, color='k', linestyle='--')
ax1.annotate('This line\n denotes\n year 1996\n when NSMIA\n is effective', xy=(1996,80000),
             arrowprops=dict(facecolor='black', shrink=0.05), xytext=(1990, 70000))

ax1.grid(False)
ax1.xaxis.grid(True)
ax1.plot(year_ps5.index, year_ps5['raised_amt'], 'b')
ax1.set_ylabel('Million Dollars', color='b')
ax1.tick_params('y', colors='b')

ax2.grid(False)
ax2.plot(year_ps5.index, year_ps5['count'], 'r')
ax2.set_ylabel('Number', color='r')
ax2.tick_params('y', colors='r')

fig1.savefig('figure1.png', transparent=False, dpi=80, bbox_inches="tight")


# Plot figure 2
ps5_data['Round'] = ps5_data['Round'].astype(int)
rd_ps5 = ps5_data[(ps5_data['Round'] <= 3)].copy() # Select first three rounds to compare across years

year_rd_ps5 = pd.DataFrame({'raised_amt' : rd_ps5.groupby(['year', 'Round']).apply(lambda x: np.sum(x['raised_amt'])),
                        'count': rd_ps5.groupby(['year', 'Round']).apply(lambda x: np.ma.count(x['raised_amt']))})
                        # Create year-round_num level data

year_rd_ps5['year'] = year_rd_ps5.index.get_level_values('year')
year_rd_ps5['Round'] = year_rd_ps5.index.get_level_values('Round')
year_rd_ps5 = year_rd_ps5.reset_index(drop = True) # Convert index to variables

# Use pivot to plot 3 bars together for each year
year_rd_ps5.pivot(index = 'year', columns = 'Round', values='count').plot(kind="bar")

# Set labels and title
plt.xlabel('Year')
plt.ylabel('Number')
fig2 = plt.gcf()

fig2.savefig('figure2.png', transparent=False, dpi=80, bbox_inches="tight")


###############################################
################ PLOT FIGURE 3 ################
###############################################
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

# Drop observations with missing state
ps5_data = ps5_data[(ps5_data.issuer_state != '-')]

# Create state level data
state_ps5 = pd.DataFrame({'raised_amt' : ps5_data.groupby('issuer_state').apply(lambda x: np.sum(x['raised_amt'])),
                        'count': ps5_data.groupby('issuer_state').apply(lambda x: np.ma.count(x['raised_amt']))})

state_ps5['issuer_state'] = state_ps5.index.get_level_values('issuer_state')
state_ps5 = state_ps5.reset_index(drop = True)
state_ps5['count'] = np.log(1 + state_ps5['count'])
# Delaware has the largest number of counts which is many times larger than the second one,
# so if plotting by original value, the color on the map is not differentiable,
# convertting to log value makes it more comparable to each states

# Lambert Conformal map of U.S. states
m = Basemap(llcrnrlon=-121,llcrnrlat=20,urcrnrlon=-62,urcrnrlat=51,
    projection='lcc',lat_1=32,lat_2=45,lon_0=-95)

# draw state boundaries.
# data from U.S Census Bureau
# https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=States+%28and+equivalent%29
shp_info = m.readshapefile('data/tl_2010_us_state00',name='states',drawbounds=True)
# choose a color for each state based on population density.
colors={}
statenames=[]
cmap = plt.cm.Reds # use 'Reds' colormap
vmin = state_ps5['count'].min() * 0.95
vmax = state_ps5['count'].max() * 1.05 # set range.
for shapedict in m.states_info:
    statename = shapedict['NAME00']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia','Puerto Rico']:
        count = float(state_ps5[state_ps5['issuer_state'] == statename]['count'].values)
        # calling colormap with value between 0 and 1 returns
        # rgba value.
        colors[statename] = cmap(((count - vmin) / (vmax - vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.
ax = plt.gca() # get current axes instance
fig = plt.gcf()
for nshape,seg in enumerate(m.states):
    # skip DC and Puerto Rico.
    if statenames[nshape] not in ['Puerto Rico', 'District of Columbia']:
    # Offset Alaska and Hawaii to the lower-left corner.
        if statenames[nshape] == 'Alaska':
        # Alaska is too big. Scale it down to 35% first, then transate it.
            seg = list(map(lambda x_y: (0.35*x_y[0] + 1100000, 0.35*x_y[1]-1300000), seg))
        if statenames[nshape] == 'Hawaii':
            seg = list(map(lambda x_y: (x_y[0] + 5200000, x_y[1] - 1400000), seg))
        color = rgb2hex(colors[statenames[nshape]])
        poly = Polygon(seg,facecolor=color,edgecolor=color)
        ax.add_patch(poly)

# construct custom colorbar
data_min = state_ps5['count'].min()
data_max = state_ps5['count'].max()
norm = Normalize(vmin=data_min, vmax=data_max)
cax = fig.add_axes([0.17, 0.01, 0.7, 0.05])
cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
cb.ax.set_xlabel('Log (Number of Private Placement)')
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.legend_.remove() # remove the legend

fig.savefig('figure3.png', transparent=False, dpi=80, bbox_inches="tight")
