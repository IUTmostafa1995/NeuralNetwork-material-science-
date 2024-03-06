import numpy as np
from dscribe.descriptors import SOAP
from ase.io.vasp import read_vasp
import pandas as pd
import matplotlib.pyplot as plt
###################################

file_shirazi=open('../list2','r')


file_read_shirazi=file_shirazi.read()

list_files_shirazi=file_read_shirazi.split('\n')[0:-1]

file=open('../list','r')


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
                "scf_gap":[],
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
                      "scf_gap":TS,
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
                      "scf_gap":TS,
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
X_train, X_test, y_train, y_test = train_test_split(feature, gaps, test_size=0.2)#, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)#, random_state=42)

##y_train=y_train.to_numpy(dtype="float64")
 ##y_test= y_test.to_numpy(dtype="float64")
##y_val=y_val.to_numpy(dtype="float64")

X_train_scaled = (X_train)
x_val_scaled = (X_val)
x_test_scaled = (X_test)
X_train_scaled = X_train_scaled.to_numpy('float64')#.reshape((X_train.shape[0], X_train.shape[1], 1))
x_test_scaled = x_test_scaled.to_numpy('float64')#.reshape((X_test.shape[0], X_test.shape[1], 1))
x_val_scaled = x_val_scaled.to_numpy('float64')#.reshape((X_val.shape[0], X_val.shape[1], 1))

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model = model_create(train_shape_1=X_train_scaled.shape[1],
                     first_neurons=first_neurons,
                     num_hidden_layers=num_hidden_layers,
                     num_neurons=num_neurons,
                     activation=activation,
                     regularization=regularization,
                     dropout_rate=dropout_rate,
                     l2_reg = l2_reg)
model.compile(loss='mae', optimizer=optimizer)
print("... model is compiled.")
print("fitting ...")
history = model.fit(X_train_scaled, y_train,
              validation_data=(x_val_scaled, y_val),
              batch_size=512,epochs=1500,
              callbacks=[early_stopping,lr_schedule],
              verbose=0)
print('validation ...')
prediction=model.predict(x_test_scaled,verbose=0)
pred_train=model.predict(X_train_scaled,verbose=0)
y_test=y_test

import matplotlib.pyplot as plt

y_test=scaler2.inverse_transform(y_test.reshape(-1,1)).flatten()
y_test=(y_test*(max_gap2-min_gap2))+min_gap2
y_test=y_test*max_gap1
y_test=y_test-shift_val
prediction=scaler2.inverse_transform(prediction.reshape(-1,1)).flatten()
prediction=(prediction*(max_gap2-min_gap2))+min_gap2
prediction=prediction*max_gap1
prediction=prediction-shift_val
y_train=scaler2.inverse_transform(y_train.reshape(-1,1)).flatten()
y_train=(y_train*(max_gap2-min_gap2))+min_gap2
y_train=y_train*max_gap1
y_train=y_train-shift_val
pred_train=scaler2.inverse_transform(pred_train.reshape(-1,1)).flatten()
pred_train=(pred_train*(max_gap2-min_gap2))+min_gap2
pred_train=pred_train*max_gap1
pred_train=pred_train-shift_val
plt.plot(y_test,prediction,'o',color='black')
plt.plot(y_train,pred_train,'o',color='red',alpha=0.5)
plt.plot(y_train,y_train,'blue')
plt.xlabel('real band gap (ev)')
plt.ylabel('predicted band gap (ev)')

print("R2 on test data is:",r2_score(y_test,prediction))
print("RMSE on test data:",np.sqrt(mean_squared_error(y_test,prediction)))
print("R2 on train data is:",r2_score(y_train,pred_train))
print("RMSE on train data:",np.sqrt(mean_squared_error(y_train,pred_train)))


plt.show()

import shap
##from sklearn.datasets import load_boston
####
####
##### Compute SHAP values for each feature
##explainer = shap.Explainer(model, X_train_scaled)
#explainer = shap.KernelExplainer(model.predict, X_train_scaled[0:100])
#shap_values = explainer.shap_values(x_test_scaled)
explainer = shap.DeepExplainer(model = model, data = X_train_scaled[0:100])
#explainer = shap.KernelExplainer(model.predict, X_train_scaled)
#shap_values = explainer.shap_values(x_test_scaled)
##
### Plot feature importance
#shap_values = explainer(X_train_scaled[0:100])
shap_values = explainer.shap_values(x_test_scaled)
shap_values = shap_values[0].reshape(-1, 18)
x_test_scaled = x_test_scaled.reshape(-1, 18)
###shap_values = explainer(x_test_scaled)
shap.summary_plot(shap_values, x_test_scaled,feature_names=['all_electron_energy','ewald_cont',
                                                            'hartree_cont','internal_energy',
                                                            'Fermi_energy','one_center_paw',
                                                            'one_electron','paw_hartree_AE',
                                                            'paw_hartree_PS','paw_xc_AE',
                                                            'paw_xc_PS','total_e_H_PAW',
                                                            'total_energy','total_force',
                                                            'totla_E_XC_PAW','scf_gap',
                                                            'volume','XC_cor'])

