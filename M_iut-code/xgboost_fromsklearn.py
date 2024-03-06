
import numpy as np
from dscribe.descriptors import SOAP
from ase.io.vasp import read_vasp
import pandas as pd
import matplotlib.pyplot as plt
###################################

file_shirazi=open('../list2-copy','r')


file_read_shirazi=file_shirazi.read()

list_files_shirazi=file_read_shirazi.split('\n')[0:-1]

file=open('../list-copy','r')


file_read=file.read()

list_files=file_read.split('\n')[0:-1]
for i in range(len(list_files)):
      list_files[i]=list_files[i][2::]
a=[]
for i in range(len(list_files)):
      a.append(list_files[i].split("-"))

import pandas as pd
#################################################

###########################################################################################
# apply the maximum absolute scaling in Pandas using the .abs() and .max() methods
def maximum_absolute_scaling(df):
    # copy the dataframe
    df_scaled = df.copy()
    # apply maximum absolute scaling
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()
    return df_scaled
    
# call the maximum_absolute_scaling function


# apply the min-max scaling in Pandas using the .min() and .max() methods
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        
    return df_norm
    
from sklearn.preprocessing import StandardScaler
##############################################################################
########################################################
from sklearn.model_selection import train_test_split
########################################################################
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l1, l2

def model_create(train_shape_1,first_neurons,num_hidden_layers,num_neurons,activation,
                 regularization,dropout_rate,l2_reg = 0.001):
      model = keras.Sequential()
      model.add(layers.Dense(first_neurons, activation=activation,input_shape=[train_shape_1]))
      model.add(layers.Dropout(0.3))
      model.add(layers.BatchNormalization())
      for i in range(num_hidden_layers):
            model.add(layers.Dense(num_neurons, activation=activation))
            if regularization=='Yes':
                  model.add(layers.Dense(num_neurons, activation=activation,
                                         kernel_regularizer=l2(l2_reg)))
            else :
                  model.add(layers.Dense(num_neurons, activation=activation))
            model.add(keras.layers.Dropout(0.3))
            model.add(keras.layers.BatchNormalization())
      model.add(layers.Dense(1))
      return model
###########################################################
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint

lr_schedule = ReduceLROnPlateau(
    factor=0.1,
    patience=20,
    verbose=1,
    min_lr=0.00001)
#lr_schedule = LearningRateScheduler(lr_schedule)

from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.0001, 
    patience=100, 
    restore_best_weights=True,
)
import random


print("---------------------------- iteration : ", i, "----------------------------------")
first_neurons=203
l2_reg=0.001
shift_val=10#13200.412
num_hidden_layers=8
num_neurons=649
activation='tanh'
regularization='Yes'
learning_rate=0.001
dropout_rate=0.6

print("features is creating ...")
feature=pd.DataFrame({"all_electron_energy" : [],
                "ewald_cont":[],
                "hartree_cont":[],
                "internal_energy":[],
                "Fermi_energy":[],
                "one_center_paw":[],
                "one_electron":[],
                "paw_hartree_AE":[],
                "paw_hartree_PS":[],
                "paw_xc_AE":[],
                "paw_xc_PS":[],
                "total_e_H_PAW":[],
                "total_energy":[],
                "total_force":[],
                "totla_E_XC_PAW":[],
                "TS":[],
                "volume":[],
                "XC_cor":[]})
for i in range(len(list_files)):
      src="/home/mostafa/BAND_STRUCTURES/Si-semiconductors/QE_files/Si"+a[i][0]+"/s"+a[i][1]
      g_v=np.loadtxt(src+"/band/GAP")+shift_val
      n_atom=np.loadtxt(src+"/n_atom.txt")
      all_electron_energy=np.loadtxt(src+"/all_electron_energy.txt")/n_atom
      ewald_cont=np.loadtxt(src+"/ewald_cont.txt")/n_atom
      fermi_energy=np.loadtxt(src+"/Fermi_energy.txt")/n_atom
      hartree_cont=np.loadtxt(src+"/hartree_cont.txt")/n_atom
      internal_energy=np.loadtxt(src+"/internal_energy.txt")/n_atom
      one_center_paw=np.loadtxt(src+"/one_center_paw.txt")/n_atom
      one_electron=np.loadtxt(src+"/one_electron.txt")/n_atom
      paw_hartree_AE=np.loadtxt(src+"/paw_hartree_AE.txt")/n_atom
      paw_hartree_PS=np.loadtxt(src+"/paw_hartree_PS.txt")/n_atom
      paw_xc_AE=np.loadtxt(src+"/paw_xc_AE.txt")/n_atom
      paw_xc_PS=np.loadtxt(src+"/paw_xc_PS.txt")/n_atom
      total_e_H_PAW=np.loadtxt(src+"/total_e_H_PAW.txt")/n_atom
      total_energy=np.loadtxt(src+"/total_energy.txt")/n_atom
      total_force=np.loadtxt(src+"/total_force.txt")/n_atom
      volume=np.loadtxt(src+"/volume.txt")/n_atom
      totla_E_XC_PAW=np.loadtxt(src+"/totla_E_XC_PAW.txt")/n_atom
      #TS=np.loadtxt(src+"/TS.txt")/n_atom
      TS=np.loadtxt(src+"/min_gap_scf")
      XC_cor=np.loadtxt(src+"/XC_cor.txt")/n_atom
      feature=feature.append({"all_electron_energy" : all_electron_energy,
                      "ewald_cont":ewald_cont,
                      "hartree_cont":hartree_cont,
                      "internal_energy":internal_energy,
                      "Fermi_energy":fermi_energy,
                      "one_center_paw":one_center_paw,
                      "one_electron":one_electron,
                      "paw_hartree_AE":paw_hartree_AE,
                      "paw_hartree_PS":paw_hartree_PS,
                      "paw_xc_AE":paw_xc_AE,
                      "paw_xc_PS":paw_xc_PS,
                      "total_e_H_PAW":total_e_H_PAW,
                      "total_energy":total_energy,
                      "total_force":total_force,
                      "totla_E_XC_PAW":totla_E_XC_PAW,
                      "TS":TS,
                              "volume":volume,
                      "XC_cor":XC_cor,
                            "GAP":g_v},ignore_index = True)
for i in range(len(list_files_shirazi)):
      src="/home/mostafa/BAND_STRUCTURES/Si-semiconductors/shirazi_relaxed/shirazi_relaxed/Si"+list_files_shirazi[i]
      g_v=np.loadtxt(src+"/band/GAP")+shift_val
      n_atom=np.loadtxt(src+"/n_atom.txt")
      all_electron_energy=np.loadtxt(src+"/all_electron_energy.txt")/n_atom
      ewald_cont=np.loadtxt(src+"/ewald_cont.txt")/n_atom
      fermi_energy=np.loadtxt(src+"/Fermi_energy.txt")/n_atom
      hartree_cont=np.loadtxt(src+"/hartree_cont.txt")/n_atom
      internal_energy=np.loadtxt(src+"/internal_energy.txt")/n_atom
      one_center_paw=np.loadtxt(src+"/one_center_paw.txt")/n_atom
      one_electron=np.loadtxt(src+"/one_electron.txt")/n_atom
      paw_hartree_AE=np.loadtxt(src+"/paw_hartree_AE.txt")/n_atom
      paw_hartree_PS=np.loadtxt(src+"/paw_hartree_PS.txt")/n_atom
      paw_xc_AE=np.loadtxt(src+"/paw_xc_AE.txt")/n_atom
      paw_xc_PS=np.loadtxt(src+"/paw_xc_PS.txt")/n_atom
      total_e_H_PAW=np.loadtxt(src+"/total_e_H_PAW.txt")/n_atom
      total_energy=np.loadtxt(src+"/total_energy.txt")/n_atom
      total_force=np.loadtxt(src+"/total_force.txt")/n_atom
      volume=np.loadtxt(src+"/volume.txt")/n_atom
      totla_E_XC_PAW=np.loadtxt(src+"/totla_E_XC_PAW.txt")/n_atom
      #TS=np.loadtxt(src+"/TS.txt")/n_atom
      TS=np.loadtxt(src+"/min_gap_scf")
      XC_cor=np.loadtxt(src+"/XC_cor.txt")/n_atom
      feature=feature.append({"all_electron_energy" : all_electron_energy,
                      "ewald_cont":ewald_cont,
                      "hartree_cont":hartree_cont,
                      "internal_energy":internal_energy,
                      "Fermi_energy":fermi_energy,
                      "one_center_paw":one_center_paw,
                      "one_electron":one_electron,
                      "paw_hartree_AE":paw_hartree_AE,
                      "paw_hartree_PS":paw_hartree_PS,
                      "paw_xc_AE":paw_xc_AE,
                      "paw_xc_PS":paw_xc_PS,
                      "total_e_H_PAW":total_e_H_PAW,
                      "total_energy":total_energy,
                      "total_force":total_force,
                      "totla_E_XC_PAW":totla_E_XC_PAW,
                      "TS":TS,
                              "volume":volume,
                      "XC_cor":XC_cor,
                            "GAP":g_v},ignore_index = True)
#feature = maximum_absolute_scaling(feature)
#feature = min_max_scaling(feature)
#std_scaler = StandardScaler()
#feature = pd.DataFrame(std_scaler.fit_transform(feature), columns=feature.columns)
print("features are created")
gaps=feature['GAP']
feature=feature.drop('GAP',axis=1)
gaps=np.array(gaps)

gaps=np.array(gaps)
max_gap1=np.max(gaps)
gaps=gaps/max_gap1


min_gap2=np.min(gaps)
max_gap2=np.max(gaps)
gaps=(gaps-min_gap2)/(max_gap2-min_gap2)
scaler2 = StandardScaler()
scaler2.fit(gaps.reshape(-1,1))
gaps=scaler2.transform(gaps.reshape(-1,1))
gaps=gaps.flatten()




feature = maximum_absolute_scaling(feature)
feature = min_max_scaling(feature)
std_scaler = StandardScaler()
feature = pd.DataFrame(std_scaler.fit_transform(feature), columns=feature.columns)
X_train, X_test, y_train, y_test = train_test_split(feature, gaps, test_size=0.05)#, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.01)#, random_state=42)

##y_train=y_train.to_numpy(dtype="float64")
##y_test= y_test.to_numpy(dtype="float64")
##y_val=y_val.to_numpy(dtype="float64")

X_train_scaled = (X_train)
x_val_scaled = (X_val)
x_test_scaled = (X_test)
X_train_scaled = X_train_scaled.to_numpy('float64')#.reshape((X_train.shape[0], X_train.shape[1], 1))
x_test_scaled = x_test_scaled.to_numpy('float64')#.reshape((X_test.shape[0], X_test.shape[1], 1))
x_val_scaled = x_val_scaled.to_numpy('float64')#.reshape((X_val.shape[0], X_val.shape[1], 1))


############################################
file_shirazi_lost=open('../lost_shrazi','r')


file_read_shirazi_lost=file_shirazi_lost.read()

list_files_shirazi_lost=file_read_shirazi_lost.split('\n')[0:-1]

file2=open('../lost_yagh','r')


file_read2=file2.read()

list_files2=file_read2.split('\n')[0:-1]
for i in range(len(list_files2)):
      list_files2[i]=list_files2[i][2::]
a2=[]
for i in range(len(list_files2)):
      a2.append(list_files2[i].split("-"))
feature2=pd.DataFrame({"all_electron_energy" : [],
                "ewald_cont":[],
                "hartree_cont":[],
                "internal_energy":[],
                "Fermi_energy":[],
                "one_center_paw":[],
                "one_electron":[],
                "paw_hartree_AE":[],
                "paw_hartree_PS":[],
                "paw_xc_AE":[],
                "paw_xc_PS":[],
                "total_e_H_PAW":[],
                "total_energy":[],
                "total_force":[],
                "totla_E_XC_PAW":[],
                #"TS":[],
                      "scf_gap":[],
                "volume":[],
                "XC_cor":[]})
for i in range(len(list_files2)):
      src="/home/mostafa/BAND_STRUCTURES/Si-semiconductors/QE_files/Si"+a2[i][0]+"/s"+a2[i][1]
      g_v=np.loadtxt(src+"/band/GAP")+shift_val
      n_atom=np.loadtxt(src+"/n_atom.txt")
      all_electron_energy=np.loadtxt(src+"/all_electron_energy.txt")/n_atom
      ewald_cont=np.loadtxt(src+"/ewald_cont.txt")/n_atom
      fermi_energy=np.loadtxt(src+"/Fermi_energy.txt")/n_atom
      hartree_cont=np.loadtxt(src+"/hartree_cont.txt")/n_atom
      internal_energy=np.loadtxt(src+"/internal_energy.txt")/n_atom
      one_center_paw=np.loadtxt(src+"/one_center_paw.txt")/n_atom
      one_electron=np.loadtxt(src+"/one_electron.txt")/n_atom
      paw_hartree_AE=np.loadtxt(src+"/paw_hartree_AE.txt")/n_atom
      paw_hartree_PS=np.loadtxt(src+"/paw_hartree_PS.txt")/n_atom
      paw_xc_AE=np.loadtxt(src+"/paw_xc_AE.txt")/n_atom
      paw_xc_PS=np.loadtxt(src+"/paw_xc_PS.txt")/n_atom
      total_e_H_PAW=np.loadtxt(src+"/total_e_H_PAW.txt")/n_atom
      total_energy=np.loadtxt(src+"/total_energy.txt")/n_atom
      total_force=np.loadtxt(src+"/total_force.txt")/n_atom
      volume=np.loadtxt(src+"/volume.txt")/n_atom
      totla_E_XC_PAW=np.loadtxt(src+"/totla_E_XC_PAW.txt")/n_atom
      #TS=np.loadtxt(src+"/TS.txt")/n_atom
      x=np.loadtxt(src+"/min_gap_scf")
      XC_cor=np.loadtxt(src+"/XC_cor.txt")/n_atom
      feature2=feature2.append({"all_electron_energy" : all_electron_energy,
                      "ewald_cont":ewald_cont,
                      "hartree_cont":hartree_cont,
                      "internal_energy":internal_energy,
                      "Fermi_energy":fermi_energy,
                      "one_center_paw":one_center_paw,
                      "one_electron":one_electron,
                      "paw_hartree_AE":paw_hartree_AE,
                      "paw_hartree_PS":paw_hartree_PS,
                      "paw_xc_AE":paw_xc_AE,
                      "paw_xc_PS":paw_xc_PS,
                      "total_e_H_PAW":total_e_H_PAW,
                      "total_energy":total_energy,
                      "total_force":total_force,
                      "totla_E_XC_PAW":totla_E_XC_PAW,
                      #"TS":TS,
                              "scf_gap":x,
                              "volume":volume,
                      "XC_cor":XC_cor,
                            "GAP":g_v},ignore_index = True)
for i in range(len(list_files_shirazi_lost)):
      src="/home/mostafa/BAND_STRUCTURES/Si-semiconductors/shirazi_relaxed/shirazi_relaxed/Si"+list_files_shirazi_lost[i]
      g_v=np.loadtxt(src+"/band/GAP")+shift_val
      n_atom=np.loadtxt(src+"/n_atom.txt")
      all_electron_energy=np.loadtxt(src+"/all_electron_energy.txt")/n_atom
      ewald_cont=np.loadtxt(src+"/ewald_cont.txt")/n_atom
      fermi_energy=np.loadtxt(src+"/Fermi_energy.txt")/n_atom
      hartree_cont=np.loadtxt(src+"/hartree_cont.txt")/n_atom
      internal_energy=np.loadtxt(src+"/internal_energy.txt")/n_atom
      one_center_paw=np.loadtxt(src+"/one_center_paw.txt")/n_atom
      one_electron=np.loadtxt(src+"/one_electron.txt")/n_atom
      paw_hartree_AE=np.loadtxt(src+"/paw_hartree_AE.txt")/n_atom
      paw_hartree_PS=np.loadtxt(src+"/paw_hartree_PS.txt")/n_atom
      paw_xc_AE=np.loadtxt(src+"/paw_xc_AE.txt")/n_atom
      paw_xc_PS=np.loadtxt(src+"/paw_xc_PS.txt")/n_atom
      total_e_H_PAW=np.loadtxt(src+"/total_e_H_PAW.txt")/n_atom
      total_energy=np.loadtxt(src+"/total_energy.txt")/n_atom
      total_force=np.loadtxt(src+"/total_force.txt")/n_atom
      volume=np.loadtxt(src+"/volume.txt")/n_atom
      totla_E_XC_PAW=np.loadtxt(src+"/totla_E_XC_PAW.txt")/n_atom
      #TS=np.loadtxt(src+"/TS.txt")/n_atom
      x=np.loadtxt(src+"/min_gap_scf")
      XC_cor=np.loadtxt(src+"/XC_cor.txt")/n_atom
      feature2=feature2.append({"all_electron_energy" : all_electron_energy,
                      "ewald_cont":ewald_cont,
                      "hartree_cont":hartree_cont,
                      "internal_energy":internal_energy,
                      "Fermi_energy":fermi_energy,
                      "one_center_paw":one_center_paw,
                      "one_electron":one_electron,
                      "paw_hartree_AE":paw_hartree_AE,
                      "paw_hartree_PS":paw_hartree_PS,
                      "paw_xc_AE":paw_xc_AE,
                      "paw_xc_PS":paw_xc_PS,
                      "total_e_H_PAW":total_e_H_PAW,
                      "total_energy":total_energy,
                      "total_force":total_force,
                      "totla_E_XC_PAW":totla_E_XC_PAW,
                      #"TS":TS,
                              "scf_gap":x,
                              "volume":volume,
                      "XC_cor":XC_cor,
                            "GAP":g_v},ignore_index = True)
#######################################
#####################################
#########################################
#############################################
###############################################
gaps2=feature2['GAP']
feature2=feature2.drop('GAP',axis=1)

gaps2=np.array(gaps2)



gaps2=gaps2/max_gap1




gaps2=(gaps2-min_gap2)/(max_gap2-min_gap2)


gaps2=scaler2.transform(gaps2.reshape(-1,1))
gaps2=gaps2.flatten()




feature2 = maximum_absolute_scaling(feature2)
feature2 = min_max_scaling(feature2)

feature2 = pd.DataFrame(std_scaler.fit_transform(feature2), columns=feature2.columns)


a = feature2.to_numpy('float64')#.reshape((X_train.shape[0], X_train.shape[1], 1))

x_test_scaled=np.concatenate((x_test_scaled,a))
y_test=np.concatenate((y_test,gaps2))
##############################################
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn import  ensemble
params = {
    "n_estimators": 600,
    "max_depth": 5,
    "min_samples_split": 10,
    "learning_rate": 0.01,
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train_scaled, y_train)
mse = mean_squared_error(y_test, reg.predict(x_test_scaled))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
# Make predictions on the testing data
y_pred = reg.predict(x_test_scaled)
pred_train = reg.predict(X_train_scaled)
#pred_spc = reg.predict(feature2)
# Evaluate the performance of the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
prediction=y_pred

import matplotlib.pyplot as plt

y_test=scaler2.inverse_transform(y_test.reshape(-1,1)).flatten()
y_test=(y_test*(max_gap2-min_gap2))+min_gap2
y_test=y_test*max_gap1
y_test=y_test-shift_val
y_train=scaler2.inverse_transform(y_train.reshape(-1,1)).flatten()
y_train=(y_train*(max_gap2-min_gap2))+min_gap2
y_train=y_train*max_gap1
y_train=y_train-shift_val
pred_train=scaler2.inverse_transform(pred_train.reshape(-1,1)).flatten()
pred_train=(pred_train*(max_gap2-min_gap2))+min_gap2
pred_train=pred_train*max_gap1
pred_train=pred_train-shift_val
#pred_spc=scaler2.inverse_transform(pred_spc.reshape(-1,1)).flatten()
#pred_spc=(pred_spc*(max_gap2-min_gap2))+min_gap2
#pred_spc=pred_spc*max_gap1
#pred_spc=pred_spc-shift_val
prediction=scaler2.inverse_transform(prediction.reshape(-1,1)).flatten()
prediction=(prediction*(max_gap2-min_gap2))+min_gap2
prediction=prediction*max_gap1
prediction=prediction-shift_val
plt.style.use("seaborn-darkgrid")
plt.plot(y_test,prediction,'o',color='black',markersize=15)
plt.plot(y_train,pred_train,'o',color='blue',markersize=10,alpha=0.3)
plt.plot(y_train,y_train,color='red')
plt.xlabel('real band gap (ev)')
plt.ylabel('predicted band gap (ev)')
plt.show()

print("R2 is on test data is:",r2_score(y_test,prediction))
print("RMSE is on test data:",np.sqrt(mean_squared_error(y_test,prediction)))
print("R2 is on train data is:",r2_score(y_train,pred_train))
print("RMSE is on test data:",np.sqrt(mean_squared_error(y_train,pred_train)))
##import shap
##from sklearn.datasets import load_boston
##
##
### Compute SHAP values for each feature
###explainer = shap.KernelExplainer(rf_reg.predict, X_train_scaled)
##explainer = shap.Explainer(reg.predict, X_train_scaled)
###shap_values = explainer.shap_values(x_test_scaled)
##shap_values = explainer(x_test_scaled)
### Plot feature importance
##shap.summary_plot(shap_values, x_test_scaled,feature_names=['all_electron_energy','ewald_cont',
##                                                            'hartree_cont','internal_energy',
##                                                            'Fermi_energy','one_center_paw',
##                                                            'one_electron','paw_hartree_AE',
##                                                            'paw_hartree_PS','paw_xc_AE',
##                                                            'paw_xc_PS','total_e_H_PAW',
##                                                            'total_energy','total_force',
##                                                            'totla_E_XC_PAW','TS',
##                                                            'volume','XC_cor'])
