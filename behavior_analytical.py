"""
Copyright (c) 2020 -
Leon Kellner, Merten Stender, Hamburg University of Technology, Germany
https://www2.tuhh.de/skf/
https://cgi.tu-harburg.de/~dynwww/cgi-bin/home/

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
"""


#%% 0.1 Packages
# general data handling modules
import pandas as pd
import numpy as np

# machine learning
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

# other
import importlib
import pickle

# inhouse modules
import auxiliary_functions as aux  # repeatedly used functions are outsourced to this module
import data_preprocessing as dp    # data preprocessing module
importlib.reload(dp)  # automatic reload (sometimes needed)
importlib.reload(aux)

# %% 0.2 Variables and constants
# control variables - data cleaning
freshwater_ = False    # only use freshwater data t/f
drop_nan_ = False      # drop all rows that contain nan values after data cleaning t/f, not necessary for XGBoosted trees
exp_cat_ = False
onehot_ = False
fixed_params_ = False  # use fixed parameter values for Schulson Model t/f

# some general variables for postprocessing
dpi_ = 1000
fig_size_ = aux.cm2inch(12, 10)
fig_path_name_template = "raw_figures/empirical_sw_"
excel_path_name_template = 'excel_outputs/empirical_sw_'
if freshwater_:
    fig_path_name_template = "raw_figures/empirical_fw_"
    excel_path_name_template = 'excel_outputs/empirical_fw_'

file = 'data points_v1.12.xlsx'

# %% 1.0 Preprocessing
# method specific data cleaning & split into encoded and non-encoded data
# Display datasets are not encoded or scaled
data = dp.data_cleaning(filename=file)  # method agnostic initial data cleaning

# for the next step to work in this script, the principal stresses (sig1 - sig3) have to be kept in "data_preprocessing.py"
X, y, X_disp, y_disp = dp.data_prep_behavior_pred(data, freshwater=freshwater_, onehot=onehot_,
                                                  drop_nan=drop_nan_, exp_cat=exp_cat_)

# drop rows in data where grain size is NaN
y.index = X.index  # set index of target to index of predictor data

if freshwater_:
    y = y[X[X['grain_size'].notnull()].index]       # remove grain_size = NaN rows in target
    X.dropna(subset=['grain_size'], inplace=True)   # remove grain_size = NaN rows in predictor data

# check results for only uniaxial data if desired
# y.drop(X[(X.type_test != 2)].index, inplace=True)
# X.drop(X[(X.type_test != 2)].index, inplace=True)   # drop rows that are not uniaxial

# %% 1.1 Functions


def friction_coefficient(v_s, T):
    """Estimates coefficient of friction based on sliding velocity and
    temperature with polynomial fit.

    Source: Linear fit to datapoints from different sources
    (dois: 10.1080/01418610008212103, 10.3189/172756503781830647, 10.1029/2012JB009219)

    Parameters
    ----------
    v_s : Sliding velocity in m/s
    T : Temperature in °C


    Returns
    -------
    coefficient of friction : Dimensionless coefficient of friction
    """
    v_s = np.log10(v_s)   # logarithmize sliding velocity for fit

    # coefficients for polynomial fit
    # f(v_s, T) = p00 + p10*v_s + p01*T + p20*vs^2
    poly_coeffs = [-0.4814, -0.3413, -0.00672, -0.03005]
    mu = np.dot(poly_coeffs, [1., v_s, T, v_s*v_s])  # compute friction coefficient with fitted polynomial

    # limit friction coefficient to feasible values
    if mu == 1:
        return 0.95  # take value slightly lower than 1, else denominator in equation can be 0 for R = 0
    return mu


def creep_parameter(temperature=-10, B_ref=5e-7, type_ice=1):
    """Compute creep parameter B in dependence of temperature

    Sources: Schulson and Duval, 2009, pp. 325ff; Sanderson, 1988, pp. 79ff.

    Parameters
    ----------
    temperature : Temperature in °C
    B_ref : Reference value for B to calculate B_0 in MPa^-3 s^-1
    type_ice : Granular/1 or columnar/0. This might change for other types of encoding!

    Returns
    -------
    B : Creep parameter in MPa^-3 s^-1
    """
    Sanderson = False      # Use values from Sanderson or Schulson and Duval
    R_gas = 8.31446261    # J K^-1 mol^-1, universal gas constant

    if Sanderson:  # calculate B_0 after Sanderson
        if type_ice == 1:  # granular ice
            Q = 78e3 if temperature < -8 else 120  # J mol^-1, apparent activation energy
            B_0 = 4.1e8 if temperature < -8 else 7.8e16  # MPa^-3 s^-1
        elif type_ice == 0:  # columnar ice
            Q = 65e3
            B_0 = 3.5e6
        else:
            print('Type of ice not implemented! Taking granular ice values.')
            Q = 78e3 if temperature < -8 else 120  # J mol^-1, apparent activation energy
            B_0 = 4.1e8 if temperature < -8 else 7.8e16  # MPa^-3 s^-1
    else:  # simpler calculation of B_0 after Schulson, Duval, 2009
        # Take the same Q values for all ice types
        Q = 78e3
        B_0 = B_ref/(np.exp(-1*Q/(R_gas*(-10+273.15))))  # B_0 for temperature dependent calculation of B with B_ref at -10°C

    temperature = temperature + 273.15
    return B_0*np.exp(-1*Q/(R_gas*temperature))


def is_ductile_behavior(strain_rate, sigma_1, sigma_3, temperature=-10, grain_size=np.nan,
                        fixed_params=False, freshwater=True, type_ice=1):
    """Compute transition strain rate with analytical model.

    Source: Schulson and Duval, 2009, pp. 325ff.

    Parameters
    ----------
    strain_rate : Logarithmized global strain rate
    grain_size : Grain size in mm.
    sigma_1, sigma_3: Greatest (sigma_1) and least (sigma_3) compressive stresses in MPa.
    temperature : Temperature in degrees Celsius.
    fixed_params : Boolean. Whether or not to use a fixed set of estimated parameters.
    freshwater : Boolean. Whether or not to calculate for freshwater ice.
    type_ice : Int. 1 = granular, 0 = columnar.

    Returns
    -------
    ductile : True/False, depending on transition strain rate being smaller or bigger than the global strain rate.
    """
    strain_rate = 10**strain_rate           # de-logarithmized strain rate [1/s]

    if not freshwater: grain_size = 6       # if grain size is not known (always the case for saltwater ice), estimate with 6 [mm]
    grain_size = grain_size/1000.0          # convert grain size from mm to m

    try:
        R = sigma_3/sigma_1                 # ratio between the least (s3) and the greatest (s1) principal (compressive) stresses [-]
    except ZeroDivisionError:
        R = 0                               # if sig_1 == 0 -> R = 0 (uniaxial tests)

    K_Ic = 0.1  # resistance to crack propagation [MPa m^0.5]

    B = 4.3e-7 if freshwater else 5.1e-6  # fixed values for B, [MPa^-3 s^-1]
    mu = 0.5
    if not fixed_params:
        B = creep_parameter(temperature, B, type_ice)
        v_sliding = 2*strain_rate*grain_size    # Sliding velocity calculated after Schulson and Duval, 2009, p. 261 [m/s]
        mu = friction_coefficient(v_sliding, temperature)

    try:
        strain_rate_transition = (25*B*K_Ic**3) / (grain_size**(3./2.) * ((1-R) - mu*(1+R)))
    except ZeroDivisionError:
        print('Zero division in equation!')

    if strain_rate > strain_rate_transition:
        return 0  # return brittle behavior (ductile = 1)
    return 1      # return ductile behavior


#%% 2.0 Processing

# --- Predictions by Schulson model
y_Schulson = pd.Series([], dtype='float64')
y_Schulson.reindex(index=X.index)  # Set index to index from dataset
for idx, row in X.iterrows():
    sig_3 = max(row['sig_1'], row['sig_2'], row['sig_3'])  # least compressive stress (compr. stresses are negative)
    sig_1 = min(row['sig_1'], row['sig_2'], row['sig_3'])  # greatest compressive stress
    if freshwater_:
        y_Schulson[idx] = is_ductile_behavior(row['strain_rate'], sig_1, sig_3, row['temperature'],
                                              grain_size=row['grain_size'], fixed_params=fixed_params_,
                                              freshwater=freshwater_, type_ice=row['type_ice'])
    else:
        y_Schulson[idx] = is_ductile_behavior(row['strain_rate'], sig_1, sig_3, row['temperature'],
                                              grain_size=None, fixed_params=fixed_params_,
                                              freshwater=freshwater_, type_ice=row['type_ice'])

# --- predictions by ML model
X.drop(columns=['sig_1', 'sig_2', 'sig_3'], inplace=True)  # drop sig columns since they are not meant as inputs for ML model
model_path_name_template = 'models/clf_xgb_sw_pickle.dat'
if freshwater_:
    model_path_name_template = 'models/clf_xgb_fw_pickle.dat'
model = pickle.load(open(model_path_name_template, "rb"))  # load ML model
y_ML = model.predict(xgboost.DMatrix(X))  # predict with ML model
y_ML = (y_ML > 0.5).astype(int)  # convert probabilities to 0/1 classes

#%% 3.0 Postprocessing
mcc_ML = matthews_corrcoef(y_ML, y)  # calculate matthews correlation coefficient
acc_ML = accuracy_score(y_ML, y)  # calculate accuracy
mcc_Schulson = matthews_corrcoef(y_Schulson, y)  # calculate matthews correlation coefficient
acc_Schulson = accuracy_score(y_Schulson, y)  # calculate accuracy




















