# -*- coding: utf-8 -*-.
"""
Created on Wed Apr 15 11:21:26 2020

@author: leon
"""


#%% 0.1 Packages
# general data handling modules
import numpy as np
import scipy.stats as ss
import math
from statsmodels.stats.diagnostic import lilliefors

# visualization
from matplotlib import pyplot as plt
import seaborn as sns

# figure export
import tikzplotlib

# other
import importlib

# inhouse modules
import auxiliary_functions as aux  # repeatedly used functions are outsourced to this module
import data_preprocessing as dp    # data preprocessing module
importlib.reload(dp)  # automatic reload (sometimes, jupyter lab needs a kernel restart instead, if python file changed)
importlib.reload(aux)

#%% 0.2 Variables and constants
# data preparation
onehot_ = False       # one-hot encode data y/n -> supposedly worse performance with onehot encoding compared to ordinal encoding
standardize_ = False   # standardize input and target t/f
drop_nan_ = False      # drop all rows that contain nan values after data cleaning t/f, not necessary for XGBoosted trees
drop_outlier_ = 0  # drop outliers in strength values

# visualization
colormap = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB"]  # https://personal.sron.nl/~pault/
colormap_desat = [sns.desaturate(c, 0.85) for c in colormap]  # desaturated colormap for plots with wide bars

#%% 1.0 Import and prepare data
file = 'data points_v1.12.xlsx'
data_temp = dp.data_cleaning(filename=file)  # method agnostic initial data cleaning

_, _, X_f, y_f = dp.data_prep_strength_pred(data_temp, freshwater=True, onehot=onehot_,
                                            drop_nan=drop_nan_, drop_outlier=drop_outlier_)

_, _, X_s, y_s = dp.data_prep_strength_pred(data_temp, freshwater=False, onehot=onehot_,
                                            drop_nan=drop_nan_, drop_outlier=drop_outlier_)

data = dp.data_prep_exploratory(data_temp)   # further data cleaning, keep only columns relevant for analysis

#%% 1.1 General exploratory analysis
num_features = data.shape[1]
num_samples = data.shape[0]

print(f'Number of input feautures: {num_features}')

# --- histogram of all inputs
fig, axes = plt.subplots(4, 3, figsize=(7, 7))
alpha_ = 0.8
kwargs = dict(alpha=alpha_)
kwargs_salt = dict(color=colormap_desat[3], label='saltwater', alpha=alpha_)
kwargs_fresh = dict(color=colormap_desat[0], label='freshwater', alpha=alpha_)


# - type test
sns.countplot(data=data, x='type_test', hue='type_water', **kwargs, ax=axes[0, 0], palette=[colormap[0], colormap[3]],
              order=['uniaxial compression', 'biaxial compression', 'triaxial compression', 'uniaxial tension'])
axes[0, 0].set(title='type test', ylabel='frequency')
axes[0, 0].set_xticklabels(['uc', 'bc', 'tc', 'ut'])
axes[0, 0].set(xlabel=None)

# - type ice
sns.countplot(data=data, x='type_ice', hue='type_water', **kwargs, ax=axes[0, 1], palette=[colormap[0], colormap[3]],
              order=data['type_ice'].value_counts().index)
axes[0, 1].set(title='type ice')
axes[0, 1].legend_.remove()
axes[0, 1].set(xlabel=None)
axes[0, 1].set(ylabel=None)

# - columnar loading
sns.countplot(data=data, x='columnar_loading', hue='type_water', **kwargs, ax=axes[0, 2], palette=[colormap[0], colormap[3]],
              order=['along', 'across', '45'])
axes[0, 2].set(title='columnar loading')
axes[0, 2].legend_.remove()
axes[0, 2].set_xticklabels(['along', 'across', '45°'])
axes[0, 2].set(xlabel=None)

# - strain rate
axes[1, 0].hist(data.loc[data.type_water == 's', 'strain_rate'], **kwargs_salt)
axes[1, 0].hist(data.loc[data.type_water == 'f', 'strain_rate'], **kwargs_fresh)
axes[1, 0].set(title='log_10(strain rate) [-]', ylabel='frequency')


# - temperature
axes[1, 1].hist(data.loc[data.type_water == 's', 'temperature'], **kwargs_salt)
axes[1, 1].hist(data.loc[data.type_water == 'f', 'temperature'], **kwargs_fresh)
axes[1, 1].set(title='temperature [°C]')

# - grain size
axes[1, 2].hist(data.loc[data.type_water == 'f', 'grain_size'], **kwargs_fresh)
axes[1, 2].set(title='grain size [mm]')

# - triaxiality
axes[2, 0].hist(data.loc[data.type_water == 's', 'triaxiality'], **kwargs_salt)
axes[2, 0].hist(data.loc[data.type_water == 'f', 'triaxiality'], **kwargs_fresh)
axes[2, 0].set(title='triaxiality [-]', ylabel='frequency')

# - porosity
axes[2, 1].hist(data.loc[data.type_water == 's', 'porosity'], **kwargs_salt)
axes[2, 1].set(title='porosity saltwater [%]')

# - porosity
axes[2, 2].hist(data.loc[data.type_water == 'f', 'porosity'], **kwargs_fresh)
axes[2, 2].set(title='porosity freshwater [%]')

# - salinity
axes[3, 0].hist(data.loc[data.type_water == 's', 'salinity'], **kwargs_salt)
axes[3, 0].set(title='salinity saltwater [‰]', ylabel='frequency')

# - volume
axes[3, 1].hist(data.loc[data.type_water == 's', 'volume'], **kwargs_salt)
axes[3, 1].hist(data.loc[data.type_water == 'f', 'volume'], **kwargs_fresh)
axes[3, 1].set(title='volume [mm^3]')

# - largest dimension
axes[3, 2].hist(data.loc[data.type_water == 's', 'largest_dim'], **kwargs_salt)
axes[3, 2].hist(data.loc[data.type_water == 'f', 'largest_dim'], **kwargs_fresh)
axes[3, 2].set(title='largest dimension [mm]')


# axes[3, 2].axis('off')

fig.tight_layout()
tikzplotlib.save("./tikzplots/hists_inputs.tex")
plt.show()

# --- histograms of outputs
# bar chart of type behavior
g = plt.figure()
kwargs_temp = dict(width=0.6)
g = sns.countplot(data=data, x='type_behavior', hue='type_water',
                  palette=[colormap[0], colormap[3]], saturation=0.75)
g.set(title='type behavior', ylabel='frequency')
g.set(xlabel=None)
g.legend(labels=['freshwater', 'saltwater'])
tikzplotlib.save("./tikzplots/hist_behavior_type.tex")
plt.show()

#%% 1.2 Correlation analysis
# This only works if the feature 'largest_dim' is not excluded during data preparation

# Pairplot for freshwater ice, excluding categorical data
# g = plt.figure()
# g = sns.pairplot(X_f, hue="type_ice", palette=colormap,
#                  x_vars=['strain_rate', 'temperature', 'grain_size', 'porosity', 'triaxiality', 'volume', 'largest_dim'],
#                  y_vars=['strain_rate', 'temperature', 'grain_size', 'porosity', 'triaxiality', 'volume', 'largest_dim'])
# plt.tight_layout()

# # Pairplot for saltwater ice, excluding categorical data
# g = plt.figure()
# g = sns.pairplot(X_s, hue="type_ice", palette=colormap,
#                  x_vars=['strain_rate', 'temperature', 'salinity', 'porosity', 'triaxiality', 'volume', 'largest_dim'],
#                  y_vars=['strain_rate', 'temperature', 'salinity', 'porosity', 'triaxiality', 'volume', 'largest_dim'])
# plt.tight_layout()
# y_vars=['strain_rate', 'temperature', 'salinity', 'porosity', 'triaxiality', 'volume', 'largest_dim'])


