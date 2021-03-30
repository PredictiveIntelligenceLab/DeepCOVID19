SEIR-mobility Code and Data Processing Guide
@author: mohamedazizbhouri

Step 1: Download Google mobility data as "Global_Mobility_Report.csv" file from https://www.google.com/covid19/mobility/ (version of July 23rd considered in this project). Such a file can be found in the folder "raw".

Step 2: For Google mobility data, remove the counties containing missing mobility data for more than 3 consecutive days or 20 days in total over the considered time period. 
	Use linear interpolation for missing data for 3 or less consecutive days.
	Apply such data processing to generate mobility data in csv files under the names "google_mobility_perso_i.csv", where i is an index (from 1 to 4 in this project), such that columns of data for the different counties are concatenated horizontally which allows adding further data of future days if needed. The generation of multiple files instead of one is imposed by the limit number of columns that can be considered in csv file's sheets. Such files can be found in the folder "raw".

Step 3: Create the csv files containing the corresponding county FIPS codes of the counties considered in the files "google_mobility_perso_i.csv". Such files for the conducted project are named "google_mobility_perso_fips_i.csv", where i is an index (from 1 to 4 in this project), and can be found in the folder "raw".

Step 4: Extract "COVID_2020-06-15_sds-v3-full-county.csv" file from "unacast.zip" file  and sort its rows such that data for each county is grouped within consecutive rows and sorted in time (days) to obtain the file "unacast_mobility_perso.csv". Such a file can be found in the folder "raw".

Step 5: Download USAFacts county level COVID-19 known cases as "covid_confirmed_usafacts.csv" file from https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/ (version of July 23rd considered in this project can be found as the file "usafacts_infections_perso.csv" in the folder "raw").

Step 6: Conisder a csv file containing a list of US counties with their county FIPS codes, populations, and corresponding states. Such a file can be found under the name "county_popcenters_perso.csv" in the folder "raw".

Step 7: Run code "load_perso_raw_google_unacast.py" to generate:
	"X_train.npy" file containing the mobility and social behavior data as a numpy tensor of size N_c x (N_t+\tau-1) x N_m ;
	"Y_train.npy" containing the cumulative cases data as a numpy tensor of size N_c x N_t ; 
	"county_name.txt" a text file containing an ordered list of the counties considered for the processed data (based on the order used for X_train.npy and Y_train.npy) ;
	and "state_name.txt" a text file containing an ordered list of the corresponding states of the counties considered for the processed data (based on the order used for X_train.npy and Y_train.npy). "county_name.txt" and "state_name.txt" will be used in the plots generation. These four files are saved in folder "processed".

Step 8: Run code "SEIR_mobility_beta_E0_LSTM_multi_step_trapeze.py" to:
	train the models; 
	save the trained models parameters as "weights_i.npy" files, where i is the index of the model, in the folder "weights"; 
	save a plot of the models losses convergence as a function of the iteration number for the trained models as "loss.png" file in folder "plots";
	and compute the average relative prediction and extrapolation errors.

Step 8 bis: If you want the computational framework to also learn \gamma and \delta parameters of the SEIR model, run code "SEIR_mobility_all_param_E0_LSTM_multi_step_trapeze.py" instead to:
	train the models; 
	save the trained models parameters as "all_param_weights_i.npy" files, where i is the index of the model, in the folder "all_param_weights"; 
	save a plot of the models losses convergence as a function of the iteration number for the trained models as "all_param_loss.png" file; 
	save plots of the learned gamma and delta parameters as functions of the iteration number for the trained models as "gamma_traj.png" and "delta_traj.png" files respectively in folder "all_param_plots"; 
	and compute the average relative prediction and extrapolation errors.

Step 9: Following Step 8, run code "SEIR_mobility_beta_E0_LSTM_multi_step_trapeze_create_plots.py" to:
	save plots of baseline and predicted trajectories of cumulative cases percentage and basic reproduction number as "C_R0_county_name_state_name.png" files, where county_name and state_name are the appropriate county and state names respectively, for selected counties, the plots are save in folder "plots";
	save plots of altered mobility, the corresponding predictions for beta(t) parameter, basic reproduction number and cumulative cases percentage as "altered_mobility.png", "altered_mobility_beta.png", "altered_mobility_R0.png", "altered_mobility_C.png" files respectively for a selected county and a chosen mobility to alter, the plots are save in folder "plots";
	save a heat map of the global sensitivity analysis conducted for the dependency of the parameter beta(t) on the mobility parameters as "sens_avg.png" file which is saved in folder "plots".

Step 9 bis: Following Step 8 bis, run code "SEIR_mobility_all_param_E0_LSTM_multi_step_trapeze_create_plots.py" to:
	save plots of baseline and predicted trajectories of cumulative cases percentage and basic reproduction number as "C_R0_county_name_state_name.png" files, where county_name and state_name are the appropriate county and state names respectively, for selected counties, the plots are save in folder "all_param_plots";
	save a heat map of the global sensitivity analysis conducted for the dependency of the parameter beta(t) on the mobility parameters as "sens_avg.png" file which is saved in folder "all_param_plots".




 