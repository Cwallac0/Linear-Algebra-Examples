# -*- coding: utf-8 -*-
"""
#Created on Mon Dec 23 18:45:13 2019
#
# @author: CWallace

# Import Pandas
import pandas as pd
import numpy as np
from scipy import linalg
import math
from sklearn.linear_model import LinearRegression

# Import data from CSV
data = pd.read_csv(r"Your\directory\data.csv")

# Step 1: Align Data =============================================
Y = data.iloc[:, 0]
X = data.iloc[:, 1:4]

# Add the ones needed for matrix multiplication 
ones = 1
X.insert(0, 'ones', ones)

# Step 2: Create X'X Matrix ======================================
XX = np.matmul(np.transpose(X), X)

# Step 3: X'Y Vector =============================================
XY = np.matmul(np.transpose(X), Y)

# Step 4: Inverse of X'X 
XX_inv = linalg.inv(XX)

# Step 5: Coefficients: b ========================================
b = np.matmul(XX_inv, XY) 

# Step 6: P Matrix Var & Covariance and Standard Errors ==========
P = np.matmul(X, np.matmul(XX_inv, np.transpose(X)))

# Check if each column and row sums to 1
np.sum(P, axis = 1) # Columns
np.sum(P, axis = 0) # Rows

# Step 7: ANOVA ==================================================
YY = np.matmul(np.transpose(Y), Y)
YJY = np.sum(Y)**2 / len(Y)

# Residual SS
bXY = np.matmul(np.transpose(b), np.matmul(np.transpose(X),Y))

df_reg   = 3 # Number of predictors
df_res   = len(Y) - df_reg - 1
df_total = df_reg + df_res

ss_reg   = bXY - YJY
ss_res   = YY - bXY
ss_total = ss_reg + ss_res 

ms_reg = ss_reg / df_reg
ms_res = ss_res / df_res

F_stat = ms_reg / ms_res

# Step 8: Predicted Values & Residuals ===========================
b_var = ms_res * XX_inv

stand_err = map(math.sqrt, np.diag(b_var))
print(list(stand_err))

pred_Y = np.matmul(P, Y)
resids = Y - pred_Y

# LinearRegression function ======================================
model = LinearRegression().fit(X,Y)

# Coefficients
model.coef_
model.intercept_

# Model prediction
model.predict(X)

# Model residuals
model.resids = Y - model.predict(X) 

