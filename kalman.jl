# Kalman filter and smoother

# Copyright (c) 2016 Jeffrey W. Miller
# This software is released under the MIT License.
# If you use this software in your research, please cite:
# 
#   Jeffrey W. Miller (2016). Lecture Notes on Advanced Stochastic Modeling. Duke University, Durham, NC.

# This code implements a standard Kalman filter (forward algorithm) and
# Rauch–Tung–Striebel (RTS) smoother (backward algorithm). The model is:
#   p(z_1) = N(z_1 | mu_0, V_0)   Initial distribution
#   p(z_t | z_{t-1}) = N(z_t | F z_{t-1}, Q)   Process model for hidden states
#   p(x_t | z_t) = N(x_t | H z_t, R)   Observation model
# 
# The inputs to the functions below are the parameters in the model above, where
#   mu_0 = vector of length d
#   V_0 = d x d positive definite matrix
#   F = d x d matrix
#   Q = d x d positive definite matrix
#   H = D x d matrix
#   R = D x D positive definite matrix
# 
# and
#   x = D x n matrix in which x[:,t] is the vector of observations at time t.
# 
# The outputs of the functions below are:
#   mu = d x n matrix in which mu[:,t] = E(Z_t | x_{1:t})
#   V = d x d x n array in which V[:,:,t] = Cov(Z_t | x_{1:t})
#   P = d x d x n array required for the smoother
#   mu_h = d x n matrix in which mu_h[:,t] = E(Z_t | x_{1:n})
#   V_h = d x d x n array in which V_h[:,:,t] = Cov(Z_t | x_{1:n})
#
# Notes:
# - This implementation does not use matrix "square roots", and consequently,
#   may be susceptible to numerical instabilities.
# - This implementation assumes time-homogeneous process and observation models
#   (i.e., F, Q, H, and R do not depend on t), however, it could be easily
#   modified to allow for time dependence if desired.

module Kalman

# Kalman filter (forward algorithm). See above for inputs and outputs.
function filter(x,mu_0,V_0,F,Q,H,R)
    D,n = size(x)
    d = length(mu_0)
    mu = zeros(d,n)
    V = zeros(d,d,n)
    P = zeros(d,d,n)

    # Initialization: t = 1
    K = V_0*H'*inv(H*V_0*H' + R)
    mu[:,1] = mu_0 + K*(x[:,1] - H*mu_0)
    V[:,:,1] = V_0 - K*H*V_0

    # Forward pass
    for t = 2:n
        P[:,:,t-1] = F*V[:,:,t-1]*F' + Q
        Pl = P[:,:,t-1]
        K = Pl*H'*inv(H*Pl*H' + R)
        mu[:,t] = F*mu[:,t-1] + K*(x[:,t] - H*F*mu[:,t-1])
        V[:,:,t] = Pl - K*H*Pl
    end

    return mu,V,P
end

# RTS smoother (backward algorithm). See above for inputs and outputs.
function smoother(mu,V,P,F)
    d,n = size(mu)
    mu_h = zeros(d,n)
    V_h = zeros(d,d,n)

    # Initialization: t = n
    mu_h[:,n] = mu[:,n]
    V_h[:,:,n] = V[:,:,n]

    # Backward pass
    for t = n-1:-1:1
        C = V[:,:,t]*F'*inv(P[:,:,t])
        mu_h[:,t] = mu[:,t] + C*(mu_h[:,t+1] - F*mu[:,t])
        V_h[:,:,t] = V[:,:,t] + C*(V_h[:,:,t+1] - P[:,:,t])*C'
    end

    return mu_h,V_h
end


end # module

