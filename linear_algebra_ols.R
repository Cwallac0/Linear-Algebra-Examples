# This is a manual example of OLS using linear algebra
# Geared to the person interested in the matrix multiplication underlying OLS
# This is toy example; not concerned with time series issues

# Import toy data
data <- EuStockMarkets[1:19, ]

# Step 1: Align Data==============================================
Y <- data[, 1]
X <- data[, -1]

# Add the ones needed for matrix multiplication
X <- cbind(ones = rep(1, length(Y)), X)

# Step 2: Create X'X Matrix=======================================
XX <- t(X) %*% X 

# Step 3: X'Y Vector==============================================
XY <- t(X) %*% Y

# Step 4: Inverse of X'X==========================================
XX_inv <- solve(XX)

# Step 5: Coefficients: b=========================================
b <- XX_inv %*% XY

# Step 6: P Matrix Var & Covariance and Standard Errors===========
P <- X %*%(XX_inv %*% t(X))

# Check if each column and row sums to 1
colSums(P)
rowSums(P)

# Step 7: ANOVA===================================================
YY  <- t(Y) %*% Y 
YJY <- sum(Y)^2 / length(Y)

# Residual SS
bXY <- t(b) %*% t(X) %*% Y

df_reg    <- 3 # Number of predictors
df_res    <- length(Y) - df_reg - 1
df_total  <- df_reg + df_res

ss_reg    <- bXY - YJY
ss_res    <- YY - bXY    
ss_total  <- ss_reg + ss_res

ms_reg    <-  ss_reg/ df_reg 
ms_res    <-  ss_res / df_res

F_stat    <- ms_reg / ms_res 

# Step 8: Predicted Values & Residuals
# Use drop to make ms_res a scalar: variance of b
b_var <- drop(ms_res) * XX_inv

stand_err <- sqrt(diag(b_var))

pred_Y <- P %*% Y
resids <- Y - pred_Y

# LM function=====================================================
# Use the original data where Y = DAX; X = SMI, CAC, & FTSE
model <- lm(DAX ~ SMI  +., data = as.data.frame(data))

# Model Summary
summary(model)

# Model prediction
predict(model)

# Model residuals
resid(model)
