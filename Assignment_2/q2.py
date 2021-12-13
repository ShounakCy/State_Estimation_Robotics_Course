import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt
import matplotlib


def plot_means(x_hat, error_hat):
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), dpi=100)

    for i, j in zip(range(x_hat.shape[0]), range(error_hat.shape[0])):

        ax[0].plot(x_hat[i,:], linewidth=0.5)
        ax[0].set_ylabel("Mean Estimated x")
        ax[0].set_xlabel("realizations")
        ax[0].set_title("Evolution of Estimated X")
    
        ax[1].plot(error_hat[j,:], linewidth=0.5)
        ax[1].set_ylabel("Mean Error")
        ax[1].set_xlabel("realizations")
        ax[1].set_title("Evolution of error")
            
    plt.grid() 

def plot_covariance(cov_diff_frob, cov_diff_trace):  

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), dpi=100)

    ax[0].plot(cov_diff_frob[1:-1], linewidth=0.5)
    ax[0].set_ylabel("Frobenious norm of covariance difference")
    ax[0].set_xlabel("realizations")
    ax[0].set_title("Evolution of the Frobenius norm of the difference of covariances")

    ax[1].plot(cov_diff_trace[1:-1], linewidth=0.5)
    ax[1].set_ylabel("Absolute trace of covariance difference")
    ax[1].set_xlabel("realizations")
    ax[1].set_title("Evolution of the trace of the difference of covariances:")
            
    plt.grid() 

def plot_variance(var_hat, var_diff):
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), dpi=100)

    ax[0].plot(var_hat[1:-1], linewidth=0.5)
    ax[0].set_ylabel("Mean of estimated variance")
    ax[0].set_xlabel("realizations")
    ax[0].set_title(" Evolution of the variance estimator")

    ax[1].plot(var_diff[1:-1], linewidth=0.5)
    ax[1].set_ylabel("variance differences")
    ax[1].set_xlabel("realizations")
    ax[1].set_title(" Evolution of the error of the noise variance")
            
    plt.grid() 

def means(M, K, X):
    x_hat_mean = np.zeros(shape=(M,K))
    for i in range(K):
        temp_x = np.take(X, range(i+1), axis=1)
        temp_mean = np.mean(temp_x, axis =1)
        x_hat_mean[:,i] = temp_mean
 
    #print(temp_x)
    return x_hat_mean

def cov_err(M, K, err, cov_given):
    err_hat_cov = []
    for i in range(K):
        temp_err = np.take(err, range(i+1), axis=1)
        temp_cov = np.cov(temp_err, bias =False)
        err_hat_cov.append(temp_cov)

    cov_diff = []
    cov_diff_frob = []
    cov_diff_trace = []
    for j in range(K):
        #covariance differences
        cov_d = cov_given - err_hat_cov[j]
        cov_diff.append(cov_d)
        # Frobenius norm differemce
        cov_diff_frob.append(np.linalg.norm(cov_d))
        # trace of the dierence of covariances
        cov_diff_trace.append(abs(cov_d.trace()))
 
    
    return cov_diff_frob, cov_diff_trace

def var_hat_n(M, K, X, var_given):
    var_hat = []
    for i in range(K):
        temp_x_var = np.take(X, range(i+1), axis=1)
        temp_var = np.mean(np.var(temp_x_var, axis =1))
        var_hat.append(temp_var)
    
    var_diff = []
    
    for j in range(K):
        #variance differences
        var_d = var_given - var_hat[j]
        var_diff.append(var_d)
        
 
    #print(temp_x)
    return var_hat, var_diff

if __name__ == '__main__':

    K = 10000
    M = 4
    N = 6
 
    H = np.random.random((N, M))
    H = H.astype(np.float32)
    H_T = H.T
    H_T_H = H_T @ H
    H_T_H_inv = np.linalg.inv(H_T_H)
    H_const = H_T_H_inv @ H_T

    x_true = np.random.random((M,1))
    var = 0.01

    #realization
    y = np.full((N,K), H @ x_true)

    #adding noise n, multivariate Gaussian
    for i in range(K):
        mean = np.zeros(N)
        covariance = np.diag(np.array([1] * N, dtype=np.float32)) * var
        noise = np.array([np.random.multivariate_normal(mean, covariance)])
        y[:, i] = y[:,i] + noise
    
    # a
    #estimate x and error
    x_est = H_const @ y
    error_x_est = x_true - x_est

    #mean across the relization
    x_hat = means(M, K, x_est)
    error_hat = means(M, K, error_x_est)
    plot_means(x_hat, error_hat)

    # b
    #given covariance
    cov_given = var * np.linalg.inv(H.T @ H)
    #covariance across the relization
    cov_diff_frob, cov_diff_trace = cov_err(M, K, error_x_est,cov_given)
    plot_covariance(cov_diff_frob, cov_diff_trace )

    # c
    # given sigma_sq = 0.01
    var_hat, var_diff = var_hat_n(M, K, x_est, var)
    plot_variance(var_hat, var_diff )

    plt.show()









    