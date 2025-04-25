#!/usr/bin/env python
# coding: utf-8

# # Bioreactor Optimization
# ### CHE 883
# ### Sam Schulte and Lauren Murray

# ---
# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fancyimpute import IterativeImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import load_model


# ---
# ### Importing Data

# In[2]:


data1200 = pd.read_excel("cleaned_bioreactor_data.xlsx", engine="openpyxl", sheet_name="1200")
data1200.head()


# In[3]:


data300 = pd.read_excel("cleaned_bioreactor_data.xlsx", engine="openpyxl", sheet_name="300")
data300.head()


# In[4]:


data600 = pd.read_excel("cleaned_bioreactor_data.xlsx", engine="openpyxl", sheet_name="600")
data600.head()


# In[5]:


data900 = pd.read_excel("cleaned_bioreactor_data.xlsx", engine="openpyxl", sheet_name="900")
data900.head()


# In[6]:


# check for non numeric columns 
data1200.dtypes


# ---
# ### REGEM Imputation

# In[7]:


#apply REGEM, filling in missing values, using iterative imputer
imputer=IterativeImputer(max_iter=10, random_state=42)
data300_imputed=pd.DataFrame(imputer.fit_transform(data300),columns=data300.columns)
print("Data after REGEM imputation")
print(data300_imputed.head())


# In[8]:


#apply REGEM, filling in missing values, using iterative imputer
imputer=IterativeImputer(max_iter=10, random_state=42)
data600_imputed=pd.DataFrame(imputer.fit_transform(data600),columns=data600.columns)
print("Data after REGEM imputation")
print(data600_imputed.head())


# In[9]:


#apply REGEM, filling in missing values, using iterative imputer
imputer=IterativeImputer(max_iter=10, random_state=42)
data900_imputed=pd.DataFrame(imputer.fit_transform(data900),columns=data900.columns)
print("Data after REGEM imputation")
print(data900_imputed.head())


# In[10]:


#apply REGEM, filling in missing values, using iterative imputer
imputer=IterativeImputer(max_iter=10, random_state=42)
data1200_imputed=pd.DataFrame(imputer.fit_transform(data1200),columns=data1200.columns)
print("Data after REGEM imputation")
print(data1200_imputed.head())


# ---
# ## Polynomial Fitting

# In[11]:


#adding 3rd order polynomial regression to CPC, nitrate, and biomass to replicate plot in paper
#using imputed data (above) and then generating smooth curves to plot

# Biomass
x300_bio = data300_imputed[["Time (Days)"]] # adds input x to regression
y300_bio = data300_imputed["Biomass (mg/mL)"]  #y target number one, biomass
poly300_bio = PolynomialFeatures(degree=2) #makes 3rd order polynomial 
x300_bio_poly = poly300_bio.fit_transform(x300_bio) #Actually creates the polynomial features from time values
model300_bio = LinearRegression().fit(x300_bio_poly, y300_bio) #trains the regression model to learn a curve that fits your biomass data over time

x300_vals=data300_imputed["Time (Days)"] #grabs full age column again to have range for smoothed values
x300_dense = np.linspace(x300_vals.min(), x300_vals.max(), 200).reshape(-1, 1) #makes 200 evenly spaced time points between first and last day and gives smooth line when predicting values between them

x300_dense_bio = poly300_bio.transform(x300_dense) #Turns smooth x-values into polynomial features
y300_dense_bio = model300_bio.predict(x300_dense_bio)#Uses the model to predict biomass values at all the smooth time points.

# Nitrate
nitrate_data300_imputed = data300_imputed.dropna(subset=["Nitrate (mg/mL)"])
x300_nit = nitrate_data300_imputed[["Time (Days)"]]
y300_nit = nitrate_data300_imputed["Nitrate (mg/mL)"]
poly300_nit = PolynomialFeatures(degree=2)
x300_nit_poly = poly300_nit.fit_transform(x300_nit)
model300_nit = LinearRegression().fit(x300_nit_poly, y300_nit)

x300_dense_nit = poly300_nit.transform(x300_dense)
y300_dense_nit = model300_nit.predict(x300_dense_nit)

# C-PC smoothing
cpc_data300_imputed = data300_imputed.dropna(subset=["C-PC (mg/mL)"])
x300_cpc = cpc_data300_imputed[["Time (Days)"]]
y300_cpc = cpc_data300_imputed["C-PC (mg/mL)"]
poly300_cpc = PolynomialFeatures(degree=2)
x300_cpc_poly = poly300_cpc.fit_transform(x300_cpc)
model300_cpc = LinearRegression().fit(x300_cpc_poly, y300_cpc)

x300_dense_cpc = poly300_cpc.transform(x300_dense)
y300_dense_cpc = model300_cpc.predict(x300_dense_cpc)


# In[12]:


# print coefficients for comparison to paper
print(model300_bio.intercept_)
print(model300_bio.coef_)


# In[13]:


fig, ax1 = plt.subplots(figsize=(6, 6))
x_vals = data300["Time (Days)"] #grabs raw age values from original data to plot raw data


#plotting raw data for biomass and nitrate on left y-axis
ax1.scatter(x_vals, data300["Biomass (mg/mL)"], color='blue', label='Biomass (raw)',s=20)
ax1.scatter(x_vals, data300["Nitrate (mg/mL)"], color='red', label='Nitrate (raw)',s=20)

#draws a smooth biomass curve using predicted values (y_dense_bio/y_dense_nitrate) over a smooth time range (x_dense) for nitrate and biomass
ax1.plot(x300_dense, y300_dense_bio, color='blue', label='Biomass (fit)')
ax1.plot(x300_dense, y300_dense_nit, color='red', label='Nitrate (fit)')

#limits and labels for primary y-axis and x-axis (limits based on paper to make look the same)
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Biomass, Nitrate (mg/L)")
ax1.set_xlim(0, 18)
ax1.set_ylim(0, 1200)

#adds CPC on right axis
ax2 = ax1.twinx() #twin function creates secondary y-axis that shares same x-axis but has different units and limits than primary y-axis
#plots raw CPC data
ax2.scatter(data300["Time (Days)"], data300["C-PC (mg/mL)"], color='green', label='C-PC (raw)', s=20)
#plots smoothed fitted line over raw data
ax2.plot(x300_dense, y300_dense_cpc, color='green', label='C-PC (fit)')

#labels secondary y-axis and adds scale
ax2.set_ylabel("C-PC (mg/L)")
ax2.set_ylim(0, 25)

#combines legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

#adds title
plt.title("Biomass, Nitrate, and C-PC Over Time (Raw + Fit)")
#adds gridlines
plt.grid(True)

plt.show()


# In[14]:


#adding 3rd order polynomial regression to CPC, nitrate, and biomass to replicate plot in paper
#using imputed data (above) and then generating smooth curves to plot

# Biomass
x600_bio = data600_imputed[["Time (Days)"]] # adds input x to regression
y600_bio = data600_imputed["Biomass (mg/mL)"]  #y target number one, biomass
poly600_bio = PolynomialFeatures(degree=2) #makes 3rd order polynomial 
x600_bio_poly = poly600_bio.fit_transform(x600_bio) #Actually creates the polynomial features from time values
model600_bio = LinearRegression().fit(x600_bio_poly, y600_bio) #trains the regression model to learn a curve that fits your biomass data over time

x600_vals=data600_imputed["Time (Days)"] #grabs full age column again to have range for smoothed values
x600_dense = np.linspace(x600_vals.min(), x600_vals.max(), 200).reshape(-1, 1) #makes 200 evenly spaced time points between first and last day and gives smooth line when predicting values between them

x600_dense_bio = poly600_bio.transform(x600_dense) #Turns smooth x-values into polynomial features
y600_dense_bio = model600_bio.predict(x600_dense_bio)#Uses the model to predict biomass values at all the smooth time points.

# Nitrate
nitrate_data600_imputed = data600_imputed.dropna(subset=["Nitrate (mg/mL)"])
x600_nit = nitrate_data600_imputed[["Time (Days)"]]
y600_nit = nitrate_data600_imputed["Nitrate (mg/mL)"]
poly600_nit = PolynomialFeatures(degree=2)
x600_nit_poly = poly600_nit.fit_transform(x600_nit)
model600_nit = LinearRegression().fit(x600_nit_poly, y600_nit)

x600_dense_nit = poly600_nit.transform(x600_dense)
y600_dense_nit = model600_nit.predict(x600_dense_nit)

# C-PC smoothing
cpc_data600_imputed = data600_imputed.dropna(subset=["C-PC (mg/mL)"])
x600_cpc = cpc_data600_imputed[["Time (Days)"]]
y600_cpc = cpc_data600_imputed["C-PC (mg/mL)"]
poly600_cpc = PolynomialFeatures(degree=2)
x600_cpc_poly = poly600_cpc.fit_transform(x600_cpc)
model600_cpc = LinearRegression().fit(x600_cpc_poly, y600_cpc)

x600_dense_cpc = poly600_cpc.transform(x600_dense)
y600_dense_cpc = model600_cpc.predict(x600_dense_cpc)


# In[15]:


# print coefficients for comparison to paper
print(model600_bio.intercept_)
print(model600_bio.coef_)


# In[16]:


fig, ax1 = plt.subplots(figsize=(6, 6))
x_vals = data600["Time (Days)"] #grabs raw age values from original data to plot raw data


#plotting raw data for biomass and nitrate on left y-axis
ax1.scatter(x_vals, data600["Biomass (mg/mL)"], color='blue', label='Biomass (raw)',s=20)
ax1.scatter(x_vals, data600["Nitrate (mg/mL)"], color='red', label='Nitrate (raw)',s=20)

#draws a smooth biomass curve using predicted values (y_dense_bio/y_dense_nitrate) over a smooth time range (x_dense) for nitrate and biomass
ax1.plot(x600_dense, y600_dense_bio, color='blue', label='Biomass (fit)')
ax1.plot(x600_dense, y600_dense_nit, color='red', label='Nitrate (fit)')

#limits and labels for primary y-axis and x-axis (limits based on paper to make look the same)
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Biomass, Nitrate (mg/L)")
ax1.set_xlim(0, 18)
ax1.set_ylim(0, 1200)

#adds CPC on right axis
ax2 = ax1.twinx() #twin function creates secondary y-axis that shares same x-axis but has different units and limits than primary y-axis
#plots raw CPC data
ax2.scatter(data600["Time (Days)"], data600["C-PC (mg/mL)"], color='green', label='C-PC (raw)', s=20)
#plots smoothed fitted line over raw data
ax2.plot(x600_dense, y600_dense_cpc, color='green', label='C-PC (fit)')

#labels secondary y-axis and adds scale
ax2.set_ylabel("C-PC (mg/L)")
ax2.set_ylim(0, 35)

#combines legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

#adds title
plt.title("Biomass, Nitrate, and C-PC Over Time (Raw + Fit)")
#adds gridlines
plt.grid(True)

plt.show()


# In[17]:


#adding 3rd order polynomial regression to CPC, nitrate, and biomass to replicate plot in paper
#using imputed data (above) and then generating smooth curves to plot

# Biomass
x900_bio = data900_imputed[["Time (Days)"]] # adds input x to regression
y900_bio = data900_imputed["Biomass (mg/mL)"]  #y target number one, biomass
poly900_bio = PolynomialFeatures(degree=2) #makes 3rd order polynomial 
x900_bio_poly = poly900_bio.fit_transform(x900_bio) #Actually creates the polynomial features from time values
model900_bio = LinearRegression().fit(x900_bio_poly, y900_bio) #trains the regression model to learn a curve that fits your biomass data over time

x900_vals=data900_imputed["Time (Days)"] #grabs full age column again to have range for smoothed values
x900_dense = np.linspace(x900_vals.min(), x900_vals.max(), 200).reshape(-1, 1) #makes 200 evenly spaced time points between first and last day and gives smooth line when predicting values between them

x900_dense_bio = poly900_bio.transform(x900_dense) #Turns smooth x-values into polynomial features
y900_dense_bio = model900_bio.predict(x900_dense_bio)#Uses the model to predict biomass values at all the smooth time points.

# Nitrate
nitrate_data900_imputed = data900_imputed.dropna(subset=["Nitrate (mg/mL)"])
x900_nit = nitrate_data900_imputed[["Time (Days)"]]
y900_nit = nitrate_data900_imputed["Nitrate (mg/mL)"]
poly900_nit = PolynomialFeatures(degree=2)
x900_nit_poly = poly900_nit.fit_transform(x900_nit)
model900_nit = LinearRegression().fit(x900_nit_poly, y900_nit)

x900_dense_nit = poly900_nit.transform(x900_dense)
y900_dense_nit = model900_nit.predict(x900_dense_nit)

# C-PC smoothing
cpc_data900_imputed = data900_imputed.dropna(subset=["C-PC (mg/mL)"])
x900_cpc = cpc_data900_imputed[["Time (Days)"]]
y900_cpc = cpc_data900_imputed["C-PC (mg/mL)"]
poly900_cpc = PolynomialFeatures(degree=2)
x900_cpc_poly = poly900_cpc.fit_transform(x900_cpc)
model900_cpc = LinearRegression().fit(x900_cpc_poly, y900_cpc)

x900_dense_cpc = poly900_cpc.transform(x900_dense)
y900_dense_cpc = model900_cpc.predict(x900_dense_cpc)


# In[18]:


x900_bio_poly
model900_nit.coef_


# In[19]:


fig, ax1 = plt.subplots(figsize=(6, 6))
x_vals = data900["Time (Days)"] #grabs raw age values from original data to plot raw data


#plotting raw data for biomass and nitrate on left y-axis
ax1.scatter(x_vals, data900["Biomass (mg/mL)"], color='blue', label='Biomass (raw)',s=20)
ax1.scatter(x_vals, data900["Nitrate (mg/mL)"], color='red', label='Nitrate (raw)',s=20)

#draws a smooth biomass curve using predicted values (y_dense_bio/y_dense_nitrate) over a smooth time range (x_dense) for nitrate and biomass
ax1.plot(x900_dense, y900_dense_bio, color='blue', label='Biomass (fit)')
ax1.plot(x900_dense, y900_dense_nit, color='red', label='Nitrate (fit)')

#limits and labels for primary y-axis and x-axis (limits based on paper to make look the same)
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Biomass, Nitrate (mg/L)")
ax1.set_xlim(0, 18)
ax1.set_ylim(0, 1200)

#adds CPC on right axis
ax2 = ax1.twinx() #twin function creates secondary y-axis that shares same x-axis but has different units and limits than primary y-axis
#plots raw CPC data
ax2.scatter(data900["Time (Days)"], data900["C-PC (mg/mL)"], color='green', label='C-PC (raw)', s=20)
#plots smoothed fitted line over raw data
ax2.plot(x900_dense, y900_dense_cpc, color='green', label='C-PC (fit)')

#labels secondary y-axis and adds scale
ax2.set_ylabel("C-PC (mg/L)")
ax2.set_ylim(0, 45)

#combines legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

#adds title
plt.title("Biomass, Nitrate, and C-PC Over Time (Raw + Fit)")
#adds gridlines
plt.grid(True)

plt.show()


# In[20]:


#adding 3rd order polynomial regression to CPC, nitrate, and biomass to replicate plot in paper
#using imputed data (above) and then generating smooth curves to plot

# Biomass
x1200_bio = data1200_imputed[["Time (Days)"]] # adds input x to regression
y1200_bio = data1200_imputed["Biomass (mg/mL)"]  #y target number one, biomass
poly1200_bio = PolynomialFeatures(degree=2) #makes 3rd order polynomial 
x1200_bio_poly = poly1200_bio.fit_transform(x1200_bio) #Actually creates the polynomial features from time values
model1200_bio = LinearRegression().fit(x1200_bio_poly, y1200_bio) #trains the regression model to learn a curve that fits your biomass data over time

x1200_vals=data1200_imputed["Time (Days)"] #grabs full age column again to have range for smoothed values
x1200_dense = np.linspace(x1200_vals.min(), x1200_vals.max(), 200).reshape(-1, 1) #makes 200 evenly spaced time points between first and last day and gives smooth line when predicting values between them

x1200_dense_bio = poly1200_bio.transform(x1200_dense) #Turns smooth x-values into polynomial features
y1200_dense_bio = model1200_bio.predict(x1200_dense_bio)#Uses the model to predict biomass values at all the smooth time points.

# Nitrate
nitrate_data1200_imputed = data1200_imputed.dropna(subset=["Nitrate (mg/mL)"])
x1200_nit = nitrate_data1200_imputed[["Time (Days)"]]
y1200_nit = nitrate_data1200_imputed["Nitrate (mg/mL)"]
poly1200_nit = PolynomialFeatures(degree=2)
x1200_nit_poly = poly1200_nit.fit_transform(x1200_nit)
model1200_nit = LinearRegression().fit(x1200_nit_poly, y1200_nit)

x1200_dense_nit = poly1200_nit.transform(x1200_dense)
y1200_dense_nit = model1200_nit.predict(x1200_dense_nit)

# C-PC smoothing
cpc_data1200_imputed = data1200_imputed.dropna(subset=["C-PC (mg/mL)"])
x1200_cpc = cpc_data1200_imputed[["Time (Days)"]]
y1200_cpc = cpc_data1200_imputed["C-PC (mg/mL)"]
poly1200_cpc = PolynomialFeatures(degree=2)
x1200_cpc_poly = poly1200_cpc.fit_transform(x1200_cpc)
model1200_cpc = LinearRegression().fit(x1200_cpc_poly, y1200_cpc)

x1200_dense_cpc = poly1200_cpc.transform(x1200_dense)
y1200_dense_cpc = model1200_cpc.predict(x1200_dense_cpc)


# In[21]:


fig, ax1 = plt.subplots(figsize=(6, 6))
x_vals = data1200["Time (Days)"] #grabs raw age values from original data to plot raw data


#plotting raw data for biomass and nitrate on left y-axis
ax1.scatter(x_vals, data1200["Biomass (mg/mL)"], color='blue', label='Biomass (raw)',s=20)
ax1.scatter(x_vals, data1200["Nitrate (mg/mL)"], color='red', label='Nitrate (raw)',s=20)

#draws a smooth biomass curve using predicted values (y_dense_bio/y_dense_nitrate) over a smooth time range (x_dense) for nitrate and biomass
ax1.plot(x1200_dense, y1200_dense_bio, color='blue', label='Biomass (fit)')
ax1.plot(x1200_dense, y1200_dense_nit, color='red', label='Nitrate (fit)')

#limits and labels for primary y-axis and x-axis (limits based on paper to make look the same)
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Biomass, Nitrate (mg/L)")
ax1.set_xlim(0, 60)
ax1.set_ylim(0, 4000)

#adds CPC on right axis
ax2 = ax1.twinx() #twin function creates secondary y-axis that shares same x-axis but has different units and limits than primary y-axis
#plots raw CPC data
ax2.scatter(data1200["Time (Days)"], data1200["C-PC (mg/mL)"], color='green', label='C-PC (raw)', s=20)
#plots smoothed fitted line over raw data
ax2.plot(x1200_dense, y1200_dense_cpc, color='green', label='C-PC (fit)')

#labels secondary y-axis and adds scale
ax2.set_ylabel("C-PC (mg/L)")
ax2.set_ylim(0, 1500)

#combines legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

#adds title
plt.title("Biomass, Nitrate, and C-PC Over Time (Raw + Fit)")
#adds gridlines
plt.grid(True)

plt.show()


# #### Format Polynomial Predicted Values into DataFrame for each Dataset

# In[22]:


time300 = np.arange(18) # create time array with values 0-17, as done in original paper
time300_features = poly300_bio.fit_transform(time300.reshape(-1,1)) #Actually creates the polynomial features from time values, which are the same for each model

biomass300_pred = model300_bio.predict(time300_features) #predicted biomass values for time steps
nit300_pred = model300_nit.predict(time300_features) #predicted nitrogen values for time steps
cpc300_pred = model300_cpc.predict(time300_features) #predicted cpc values for time steps

fitted300_df = pd.DataFrame({'Time (Days)': time300, 
                             'Biomass (mg/mL)': biomass300_pred, 
                             'Nitrate (mg/mL)': nit300_pred, 
                             'C-PC (mg/mL)':cpc300_pred})
fitted300_df.head()


# In[23]:


time600 = np.arange(18) # create time array with values 0-17, as done in original paper
time600_features = poly600_bio.fit_transform(time600.reshape(-1,1)) #Actually creates the polynomial features from time values, which are the same for each model

biomass600_pred = model600_bio.predict(time600_features) #predicted biomass values for time steps
nit600_pred = model600_nit.predict(time600_features) #predicted nitrogen values for time steps
cpc600_pred = model600_cpc.predict(time600_features) #predicted cpc values for time steps

fitted600_df = pd.DataFrame({'Time (Days)': time600, 
                             'Biomass (mg/mL)': biomass600_pred, 
                             'Nitrate (mg/mL)': nit600_pred, 
                             'C-PC (mg/mL)':cpc600_pred})
fitted600_df.head()


# In[24]:


time900 = np.arange(18) # create time array with values 0-17, as done in original paper
time900_features = poly900_bio.fit_transform(time900.reshape(-1,1)) #Actually creates the polynomial features from time values, which are the same for each model

biomass900_pred = model900_bio.predict(time900_features) #predicted biomass values for time steps
nit900_pred = model900_nit.predict(time900_features) #predicted nitrogen values for time steps
cpc900_pred = model900_cpc.predict(time900_features) #predicted cpc values for time steps

fitted900_df = pd.DataFrame({'Time (Days)': time900, 
                             'Biomass (mg/mL)': biomass900_pred, 
                             'Nitrate (mg/mL)': nit900_pred, 
                             'C-PC (mg/mL)':cpc900_pred})
fitted900_df.head()


# In[25]:


time1200 = np.arange(56) # create time array with values 0-55, as done in original paper
time1200_features = poly1200_bio.fit_transform(time1200.reshape(-1,1)) #Actually creates the polynomial features from time values, which are the same for each model

biomass1200_pred = model1200_bio.predict(time1200_features) #predicted biomass values for time steps
nit1200_pred = model1200_nit.predict(time1200_features) #predicted nitrogen values for time steps
cpc1200_pred = model1200_cpc.predict(time1200_features) #predicted cpc values for time steps

fitted1200_df = pd.DataFrame({'Time (Days)': time1200, 
                             'Biomass (mg/mL)': biomass1200_pred, 
                             'Nitrate (mg/mL)': nit1200_pred, 
                             'C-PC (mg/mL)':cpc1200_pred})
fitted1200_df.head()


# ---
# ## ANN

# #### Format Training Data

# In[26]:


# Need to define output data as the delta between the current and next time step
def create_training_data(df):
    X = []
    Y = []
    for i in range(len(df) - 1):
        t, x, N, C_PC = df.iloc[i][['Time (Days)', 'Biomass (mg/mL)', 'Nitrate (mg/mL)', 'C-PC (mg/mL)']]
        t_next, x_next, N_next, C_PC_next = df.iloc[i+1][['Time (Days)', 'Biomass (mg/mL)', 'Nitrate (mg/mL)', 'C-PC (mg/mL)']]

        input_vec = [t, x, N, C_PC]
        output_vec = [t_next - t, x_next - x, N_next - N, C_PC_next - C_PC]

        X.append(input_vec)
        Y.append(output_vec)

    return np.array(X), np.array(Y)

# Load and process all 3 experiments
X_all, Y_all = [], []
for df in [fitted300_df, fitted900_df, fitted1200_df]:
    X, Y = create_training_data(df)
    X_all.append(X)
    Y_all.append(Y)

# Stack all together
X_train = np.vstack(X_all)
Y_train = np.vstack(Y_all)


# In[27]:


# Create validation set from 600 mg/L data set

X_validation, Y_validation = create_training_data(fitted600_df)


# In[28]:


# Scale data to a mean of zero and a mean of 0 and standard deviation of 1

# create instances of scalers
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# fit to training data and scale training data
X_train_scaled = scaler_X.fit_transform(X_train)
Y_train_scaled = scaler_Y.fit_transform(Y_train)

# scale validation data
X_validation_scaled = scaler_X.transform(X_validation)
Y_validation_scaled = scaler_Y.transform(Y_validation)


# #### Importing packages and building model

# In[29]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# In[ ]:


# Modified version with 600 mg/L experiment used as validation set, other three experiments used as training data
    # (as opposed to using a 15% validation split from the three training experiments)

# === Define the model ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),   # Input layer (4) → Hidden 1
    Dense(64, activation='relu'),                      # Hidden 2
    Dropout(0.15),                                     # Dropout (keep_prob=0.85)
    Dense(4)                                           # Output layer (Δt, Δx, ΔN, ΔC-PC)
])

# === Compile model ===
model.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.keras.losses.MeanSquaredError())

# === Train the model ===
history = model.fit(X_train_scaled, Y_train_scaled,
                    epochs=10000,               # Feel free to reduce to ~1000 to test first
                    batch_size=32,
                    validation_data=(X_validation_scaled, Y_validation_scaled),      # or use an explicit X_val/Y_val if you prefer
                    verbose=1)                  # Set to 1 if you want to watch training live

# === Plot training + validation loss ===
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('ANN Training and Validation Loss')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("Loss_v_Epoch.png")

# === Save model ===
model.save('plectonema_ann_model.h5')
print("Model saved as 'plectonema_ann_model.h5'")




