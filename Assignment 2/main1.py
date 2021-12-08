import numpy as np
import scipy.io
import scipy.linalg
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')

def batch_est(delta=1):
    # load dataset variables
    dataset = scipy.io.loadmat("dataset1.mat")
    print(dataset)
    r = dataset["r"]
    x_true = dataset["x_true"]
    t = dataset["t"]
    v = dataset["v"]
    x_c = dataset["l"]
    r_var = dataset["r_var"]
    v_var = dataset["v_var"]
    y = x_c - r
    timestep = 0.1
    datapoints_no = x_true.shape[0]

    # create arrays/constants
    A = 1
    P_0 = r_var.item()
    Q = v_var.item()
    R = ((timestep**2)*r_var).item()
    C = 1

    P_prior_f = np.zeros(datapoints_no)
    P_post_f = np.zeros(datapoints_no)
    P_post = np.zeros(datapoints_no)
    x_prior_f = np.zeros(datapoints_no)
    x_post_f = np.zeros(datapoints_no)
    x_post = np.zeros(datapoints_no)
    K = np.zeros(datapoints_no)

    # iteration 0 calculations
    P_prior_f[0] = P_0
    x_prior_f[0] = y[0]
    K[0] = P_prior_f[0]*C*(C*P_prior_f[0]*C + R)**-1
    P_post_f[0] = (1-K[0]*C)*P_prior_f[0]
    x_post_f[0] = x_prior_f[0] + K[0]*(y[0] - C*x_prior_f[0])

    # forward pass
    for i in range(1, datapoints_no):
        # account for only receiving range measurements sometimes
        if i%delta != 0: C = 0
        else: C = 1

        P_prior_f[i] = A*P_post_f[i-1]*A+Q
        x_prior_f[i] = A*x_post_f[i-1]+v[i]*timestep

        K[i] = P_prior_f[i]*C*(C*P_prior_f[i]*C + R)**-1
        P_post_f[i] = (1-K[i]*C)*P_prior_f[i]
        x_post_f[i] = x_prior_f[i] + K[i]*(y[i] - C*x_prior_f[i])

    # backwards pass
    x_post[datapoints_no-1] = x_post_f[datapoints_no-1]
    P_post[datapoints_no-1] = P_post_f[datapoints_no-1]
    for i in range(datapoints_no-1, 0, -1):
        x_post[i-1] = x_post_f[i-1] + P_post_f[i-1]*A*(P_prior_f[i-1]**-1) * (x_post[i] - x_prior_f[i])
        P_post[i-1] = P_post_f[i-1] + (P_post_f[i-1]*A*P_prior_f[i]**-1)*(P_post[i] - P_prior_f[i])*(P_post_f[i-1]*A*P_prior_f[i]**-1)

    # error
    error = x_post - np.squeeze(x_true)
    var = P_post
    uncert = np.sqrt(var) * 3

    # plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), dpi=100)
    ax[0].plot(t, error, linewidth=0.5, label="Error [m]")
    ax[0].fill_between(np.squeeze(t), -uncert, +uncert, edgecolor='#CC4F1B', facecolor='#FF9848', alpha=0.5, linestyle=':', label='Uncertainty Envelope')
    ax[0].set_title(f"Estimation Error versus Time for Delta = {delta}")
    ax[0].set_ylabel("Estimation Error [m]")
    ax[0].set_xlabel("Time [s]")
    ax[0].legend(loc="upper right")
    ax[1].hist(error, rwidth=0.95, bins=20)
    # ax[1].axvline(x=np.max(uncert), color='r', linestyle='dashed', linewidth=2)
    # ax[1].axvline(x=-np.max(uncert), color='r', linestyle='dashed', linewidth=2)
    ax[1].set_title(f"Histogram of Estimation Error Values for Delta = {delta}")
    ax[1].set_ylabel("Count")
    ax[1].set_xlabel("Estimation Error [m]")

    plt.savefig(f"delta{delta}.png")
    # plt.show(block=False)
    print(f"Delta: {delta}, Avg Error: {np.average(np.abs(x_post) - np.abs(x_true))}")

def plot_mtx():
    # load dataset variables
    dataset = scipy.io.loadmat("dataset1.mat")
    r = dataset["r"]
    x_true = dataset["x_true"]
    t = dataset["t"]
    v = dataset["v"]
    x_c = dataset["l"]
    r_var = dataset["r_var"]
    v_var = dataset["v_var"]
    y = x_c - r
    timestep = 0.1
    datapoints_no = x_true.shape[0]
    K = 12709

    A = np.tril(np.ones((K, K), dtype=np.float32), 0)
    C = np.diag(np.array([1]*K, dtype=np.float32))
    Q = np.diag(np.array([1]*K, dtype=np.float32))
    R = np.diag(np.array([1]*K, dtype=np.float32))
    H = np.vstack([np.linalg.inv(A), C])
    H = H.astype(np.float32)
    W = scipy.linalg.block_diag(Q, R)
    W = W.astype(np.float32)
    print(W.dtype)
    W_inv = np.linalg.inv(W).astype(np.float32)
    print(W_inv.dtype)
    lhs = H.T @ W_inv @ H
    np.save("q4_array", lhs)
    fig, ax = plt.subplots(1, 1)
    ax.spy(lhs, markersize=5)
    fig.show()

def q4_plot():
    arr = np.load("q4_array.npy")
    fig, ax = plt.subplots(1, 1)
    ax.set_title("Sparsity Pattern of LHS")
    ax.spy(arr, markersize=5)
    fig.show()

if __name__ == "__main__":
    #Q4
    # plot_mtx()
    # q4_plot()

    # Q5
    for delta in [1, 10, 100, 1000]:
        batch_est(delta)


def set_up_linear_batch(u, y, A_n, var_sys, C_n, var_meas, gamma):
    """
    Create the matrices required for linear batch
    :param u: (ndarray) system inputs of all time steps
    :param y: (ndarray) measurements of all time steps
    :param A_n: motion model per timestep
    :param var_sys: motion model variance per time step
    :param C_n: measurement model per timestep
    :param var_meas: measurement variance per timestep
    :return:
    """
    print("Setting up helper matrices A, C, Q_inv and R_inv for linear batch")

    # create A^-1 matrix
    A_inv = np.identity(len(u))
    for i in range(len(A_inv) - 1):
        # under every main diagonal entry we set the negative motion model A_n
        A_inv[i + 1, i] = -A_n

    # create C matrix
    C = np.identity(len(y)) * C_n

    Q_inv = np.identity(len(u)) * 1. / var_sys
    R_inv = np.identity(len(y)) * 1. / var_meas

    print("Done!")

    y, C, R_inv = omit_meas_from_batch(y, C, R_inv, gamma)

    print("Setting up matrices z, H and inverted W for linear batch")

    z = np.vstack((u, y))
    H = np.vstack((A_inv, C))
    # directly create inverted W because it will save us lot of processing power
    W_inv = scipy.linalg.block_diag(Q_inv, R_inv)

    print("Done!")

    return z, H, W_inv