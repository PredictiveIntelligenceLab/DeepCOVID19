#!/usr/bin/python3
"""
Created on Wed Apr 29 04:15:34 2020

@author: mohamedazizbhouri
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 16,
                     'lines.linewidth': 2,
                     'axes.labelsize': 20,
                     'axes.titlesize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'legend.fontsize': 20,
                     'axes.linewidth': 2,
                     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
                     "text.usetex": True,                # use LaTeX to write all text
                     })

dpiv = 100

is_norm_CC = 1 # 1 to normalize cumulative cases number by county population ; 0 to keep the absolute number of cumulative cases
is_norm_m = 0 # 1 to normalize Google mobility parameters by the variable Norm_m ; 0 otherwise
Norm_m = 100

is_norm_m_un = 0 # 1 to normalize Unacast social behavior parameters by the variable Norm_m_un ; 0 otherwise
Norm_m_un = 1

is_Max_Norm_m_un = 1 # 1 to normalize each Unacast social behavior parameter by its maximum value over the counties and over time ; 0 otherwise
is_Max_Norm_m_un_per_county = 0 # 1 to normalize each county's Unacast social behavior parameter by its maximum value over time ; 0 otherwise

is_Gauss_Norm_m = 0 # 1 to perform Gaussian standarization of each Google mobility parameter based on its mean and std deviation over the counties and over time ; 0 otherwise
is_Max_Norm_m = 1 # 1 to normalize each Google mobility parameter by its maximum value over the counties and over time ; 0 otherwise
is_Max_norm_m_all = 0 # 1 to normalize Google mobility parameters by their maximum value over the counties and over time ; 0 otherwise
is_Max_Norm_m_per_county = 0 # 1 to normalize each county's Google mobility parameter by its maximum value over time ; 0 otherwise

is_LSTM = 1 # 1 if the NN considered is LSTM ; 0 otherwise

max_diff = 0.002 # upper bound to be imposed on the maximum value of the daily increase of cumulative cases (normalized by the number of population) for the retained counties
tau_shift = 21 # the time lag in days to consider between the mobility and social behavior parameters and the number of cumulative cases
Nt_min = 81 # the minimum number of days of the disease spread that must be satisfied by each county that is retained
NCC_min = 0.002 # the minimum number of cumulative cases (normalized by the number of population) on the last day of the data considered that must be satisfied by each county that is retained

# Load Google mobility data
dm1 = pd.read_csv('raw/google_mobility_perso_1.csv') # US (0) | United states (1) | State Name (2) | County Name (3) | Date (4) | m_i's (5-10)
dm2 = pd.read_csv('raw/google_mobility_perso_2.csv') # US (0) | United states (1) | State Name (2) | County Name (3) | Date (4) | m_i's (5-10)
dm3 = pd.read_csv('raw/google_mobility_perso_3.csv') # US (0) | United states (1) | State Name (2) | County Name (3) | Date (4) | m_i's (5-10)
dm4 = pd.read_csv('raw/google_mobility_perso_4.csv') # US (0) | United states (1) | State Name (2) | County Name (3) | Date (4) | m_i's (5-10)
dm = [dm1, dm2, dm3, dm4]
dmf1 = pd.read_csv('raw/google_mobility_perso_fips_1.csv') # State Name (0) | County Name (1) | empty (2) | FIPS (3)
dmf2 = pd.read_csv('raw/google_mobility_perso_fips_2.csv') # State Name (0) | County Name (1) | empty (2) | FIPS (3)
dmf3 = pd.read_csv('raw/google_mobility_perso_fips_3.csv') # State Name (0) | County Name (1) | empty (2) | FIPS (3)
dmf4 = pd.read_csv('raw/google_mobility_perso_fips_4.csv') # State Name (0) | County Name (1) | empty (2) | FIPS (3)
dmf = [dmf1, dmf2, dmf3, dmf4]
# Load Unacast social behavior data
dm_un = pd.read_csv('raw/unacast_mobility_perso.csv')

Nd_mob = len(dm1.axes[0]) # total number of days of available Google mobility data (same for every county)
Nc=[(len(dm1.axes[1])-1)//9, (len(dm2.axes[1])-1)//9, (len(dm3.axes[1])-1)//9, (len(dm4.axes[1])-1)//9] # list of number of counties in Google mobility files

Nd_un = 109 # total number of days of available Unacast social behavior data (same for every county)
Nc_un = 3054 # total number of counties considered in the available Unacast social behavior data

# Load USAFacts cumulative cases data
di = pd.read_csv('raw/usafacts_infections_perso.csv')

Nd_CC =184 # total number of days of available cumulative cases data

first_day_I_to_mob = 24 # Day difference beteween first day of available Google mobility data and cumulative cases data (1st day of cumulative cases data is the earlier one)
last_day_mob_to_I = 4 # Day difference beteween last day of available Google mobility data and cumulative cases data (Last day of Google mobility data is the earlier one)

first_day_I_to_un = 33 # Day difference beteween first day of available Unacast social behavior data and cumulative cases data (1st day of cumulative cases data is the earlier one)
last_day_un_to_I = 42 # Day difference beteween last day of available Unacast social behavior data and cumulative cases data (Last day of Unacast social behavior data is the earlier one)

# Load a csv file containing a list of US counties with their county FIPS codes, populations, and corresponding states.
dc = pd.read_csv('raw/county_popcenters_perso.csv')

# Verify that there is no empty entry in Google mobility data files
for i in range(len(dm1.axes[0])):
    for j in range(len(dm1.axes[1])):
        if j!=3 and pd.isnull(dm1.iloc[i,j]) == True:
            print(1,i,j)
            print('here')
for i in range(len(dm2.axes[0])):
    for j in range(len(dm2.axes[1])):
        if j!=3 and pd.isnull(dm2.iloc[i,j]) == True:
            print(2,i,j)
            print('here')
for i in range(len(dm3.axes[0])):
    for j in range(len(dm3.axes[1])):
        if j!=3 and pd.isnull(dm3.iloc[i,j]) == True:
            print(3,i,j)
            print('here')
for i in range(len(dm4.axes[0])):
    for j in range(len(dm4.axes[1])):
        if j!=3 and pd.isnull(dm4.iloc[i,j]) == True:
            print(4,i,j)
            print('here')

CC = [ [], [], [], [] ]
Nt = [ [], [], [], [] ]
Np = [ [], [], [], [] ]
I0 = [ [], [], [], [] ]
t0 = [ [], [], [], [] ]
county_name = [ [], [], [], [] ]
state_name = [ [], [], [], [] ]
FIPS = [ [], [], [], [] ]
Ind_c = [ [], [], [], [] ]

I0_min = 10 # The minimum number of infections defining the start of the disease spread in each county

for g in range(4):
    for i in range(Nc[g]):
        state = dm[g].iloc[0,1+9*i]
        if state != 'District of Columbia':
            county = dm[g].iloc[0,2+9*i]
            FIPS_loc = dmf[g].iloc[i,3]
            b = False
            j = -1
            if (county=='St. Louis County'and state=='Missouri')or(county=='St. Louis'and state=='Missouri')or(county=='Baltimore'and state=='Maryland')or(county=='Baltimore County'and state=='Maryland'):
                while not(b):
                    j += 1
                    if j>len(dc.axes[0])-1:
                        print(county,state,FIPS_loc)
                    if FIPS_loc == dc.iloc[j,7]:
                        Npl = dc.iloc[j,4]
                        b = True
            else:
                county = county.replace(' County', '')
                county = county.replace(' Parish', '')
                while not(b):
                    j += 1
                    if j>len(dc.axes[0])-1:
                        print(county,state,FIPS_loc)
                    if (county == dc.iloc[j,2]) and (state == dc.iloc[j,3]):
                        FIPS_loc = dc.iloc[j,7]
                        Npl = dc.iloc[j,4]
                        b = True
                
            b = False
            j = -1
            while not(b):
                j += 1
                if FIPS_loc == di.iloc[j,0]:
                    b2 = False
                    k = 0
                    while not(b2):
                        k +=1
                        if di.iloc[j,k] >= I0_min:
                            b2 = True
                    b = True
            kf = max(k,tau_shift+first_day_I_to_mob+1,tau_shift+first_day_I_to_un+1)
            if is_LSTM == 1:
                CCloc = di.iloc[j,kf:min(Nd_CC+1-last_day_mob_to_I+1,Nd_CC+1-last_day_un_to_I+1)]
            else:
                CCloc = di.iloc[j,kf:min(Nd_CC+1-last_day_mob_to_I+tau_shift,Nd_CC+1-last_day_un_to_I+tau_shift)]
            if (CCloc.shape[0] >= Nt_min):
                CC[g].append(CCloc)
                Nt[g].append(CCloc.shape[0])
                I0[g].append(di.iloc[j,kf])
                t0[g].append(kf)
                Np[g].append(Npl)
                county_name[g].append(county)
                state_name[g].append(state)
                Ind_c[g].append(i)
                FIPS[g].append(FIPS_loc)
        else:
            county = 'District of Columbia'
            FIPS_loc = dmf[g].iloc[i,3]
            b = False
            j = -1
            while not(b):
                j += 1
                if state == dc.iloc[j,3]:
                    Npl = dc.iloc[j,4]
                    b = True
            b = False
            j = -1
            while not(b):
                j += 1
                if FIPS_loc == di.iloc[j,0]:
                    b2 = False
                    k = 0
                    while not(b2):
                        k +=1
                        if di.iloc[j,k] >= I0_min:
                            b2 = True
                    b = True
            kf = max(k,tau_shift+first_day_I_to_mob+1,tau_shift+first_day_I_to_un+1)
            if is_LSTM == 1:
                CCloc = di.iloc[j,kf:min(Nd_CC+1-last_day_mob_to_I+1,Nd_CC+1-last_day_un_to_I+1)]
            else:
                CCloc = di.iloc[j,kf:min(Nd_CC+1-last_day_mob_to_I+tau_shift,Nd_CC+1-last_day_un_to_I+tau_shift)]
            if (CCloc.shape[0] >= Nt_min): # and (CCloc[-1]/Npl >= NCC_min):
                CC[g].append(CCloc)
                Nt[g].append(CCloc.shape[0])
                I0[g].append(di.iloc[j,kf])
                t0[g].append(kf)
                Np[g].append(Npl)
                county_name[g].append(county)
                state_name[g].append(state)
                Ind_c[g].append(i)
                FIPS[g].append(FIPS_loc)

CCn = []
Ntn = []
Npn = []
I0n = [ [], [], [], [] ]
t0n = [ [], [], [], [] ]
county_namen = []
state_namen = []
Ind_cn = [ [], [], [], [] ]
FIPSn = []

Ntf = min(Nd_mob,Nd_un)
for g in range(4):
    Ntf = min(Ntf,min(Nt[g]))
    
print('Number of days: ',Ntf)

for g in range(4):
    for i in range(len(Nt[g])):
        if CC[g][i][Ntf-1]/Np[g][i] >= NCC_min and np.max( np.array(CC[g][i][1:Ntf])-np.array(CC[g][i][:Ntf-1]) )/Np[g][i] < max_diff :
            Ntn.append(Nt[g][i])
            I0n[g].append(CC[g][i][0])
            t0n[g].append(t0[g][i])
            Npn.append(Np[g][i])
            county_namen.append(county_name[g][i])
            state_namen.append(state_name[g][i])
            FIPSn.append(int(FIPS[g][i]))
            Ind_cn[g].append(Ind_c[g][i])
            if is_norm_CC ==0:
                CCn.append(CC[g][i][:Ntf])
            else:
                CCn.append(CC[g][i][:Ntf]/Np[g][i])

CC = CCn
I0 = I0n
Np = Npn
county_name = county_namen
state_name = state_namen
FIPS = FIPSn

t0f = []

Ncf = len(CCn)
print('Nombre of counties:',Ncf)

m = []
indf = -1
for g in range(4):
    for i in range(len(Ind_cn[g])):
        indf = indf + 1
        if is_LSTM == 0:
            indm = Ntn[indf]-Ntf + (last_day_un_to_I-last_day_mob_to_I)*(last_day_un_to_I>last_day_mob_to_I)
            if is_norm_m ==0:
                mloc = dm[g].iloc[Nd_mob-indm-Ntf:Nd_mob-indm,4+Ind_cn[g][i]*9:10+Ind_cn[g][i]*9]
            else:
                mloc = dm[g].iloc[Nd_mob-indm-Ntf:Nd_mob-indm,4+Ind_cn[g][i]*9:10+Ind_cn[g][i]*9] / Norm_m
        else:
            indm = Ntn[indf]-Ntf+tau_shift-1 + (last_day_un_to_I-last_day_mob_to_I)*(last_day_un_to_I>last_day_mob_to_I)
            if is_norm_m ==0:
                mloc = dm[g].iloc[Nd_mob-indm-Ntf:Nd_mob-indm+tau_shift-1,4+Ind_cn[g][i]*9:10+Ind_cn[g][i]*9]
            else:
                mloc = dm[g].iloc[Nd_mob-indm-Ntf:Nd_mob-indm+tau_shift-1,4+Ind_cn[g][i]*9:10+Ind_cn[g][i]*9] / Norm_m
        m.append( np.array(mloc) )
        t0f.append ( t0n[g][i] )

m_un = []
ii_m = [12,15,18] # indices of columns in the csv file downloaded as the variable dm_un and corresponding to Unacast social behavior parameters

for i in range(Ncf):
    for j in range(Nc_un):
        if FIPS[i] == dm_un.iloc[j*Nd_un,3]:
            if is_LSTM == 0:
                indm = Ntn[i]-Ntf + (last_day_mob_to_I-last_day_un_to_I)*(last_day_un_to_I<last_day_mob_to_I)
                mloc = dm_un.iloc[(j+1)*Nd_un-indm-Ntf:(j+1)*Nd_un-indm,ii_m]
            else:
                indm = Ntn[i]-Ntf+tau_shift-1 + (last_day_mob_to_I-last_day_un_to_I)*(last_day_un_to_I<last_day_mob_to_I)
                mloc = dm_un.iloc[(j+1)*Nd_un-indm-Ntf:(j+1)*Nd_un-indm+tau_shift-1,ii_m]
            m_un.append( np.array(mloc) )
    
    
CCf = np.array(CCn)
m = np.array(m)      
m_un = np.array(m_un) 

if is_norm_m_un == 1:
    m_un = m_un/Norm_m_un
if is_Max_Norm_m_un == 1:
    m_un_max = np.max(np.abs(m_un),(0,1))
    m_un = m_un/m_un_max
if is_Max_Norm_m_un_per_county == 1:
    m_un_max = np.max(np.abs(m_un),1)
    m_un = m_un/m_un_max[:,None,:]

if is_Gauss_Norm_m == 1:
    Xmean = np.mean(m,(0,1))
    Xstd = np.std(m,(0,1))
    m = (m - Xmean)/Xstd
if is_Max_Norm_m == 1:
    Xmax = np.max(np.abs(m),(0,1))
    m = m/Xmax
if is_Max_Norm_m_per_county == 1:
    Xmax = np.max(np.abs(m),1) # Nc x Nm
    m = m/Xmax[:,None,:]
if is_Max_norm_m_all == 1:
    Xmax = np.max(np.abs(m))
    m = m/Xmax

m = np.concatenate((m,m_un), axis=2)

np.save('processed/X_train', m)
np.save('processed/Y_train', CCf)

with open("processed/state_name.txt", "wb") as fp:
    pickle.dump(state_name, fp)
    
with open("processed/county_name.txt", "wb") as fp:
    pickle.dump(county_name, fp)

