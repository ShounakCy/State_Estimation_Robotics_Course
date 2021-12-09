import numpy as np
import scipy.io
import scipy.linalg
from matplotlib import pyplot as plt
import matplotlib


def batch_est(d, y_d, measurement_var):
    K = 12709
    C_arr = np.diag(np.array([1] * K, dtype=np.float32))
    R_inv = np.diag(np.array([1] * K, dtype=np.float32)) * (1. / measurement_var)
    if d == 1:
        return C_arr, R_inv, y_d
    else:
        C_delta = np.zeros(shape=(K // d, K))
        R_delta = np.zeros(shape=(K // d, K // d))
        y_delta = np.zeros(shape=(K // d, 1))

        for i, j in zip(range(d, K, d), range(len(C_delta))):
            C_delta[j] = C_arr[i]
            R_delta[j][j] = R_inv[i][i]
            y_delta[j] = y_d[i]
            C_delta = np.vstack(C_delta)
            R_delta = np.vstack(R_delta)
            y_delta = np.vstack(y_delta)
            # print(y_delta)
        return C_delta, R_delta, y_delta


def q5(u_k, position_var, C_k, R_inv_k, y_kd):
    K = 12709
    A = np.tril(np.ones((K, K), dtype=np.float32), 0)  # not affected by gamma

    H = np.vstack([np.linalg.inv(A), C_k])
    H = np.delete(H, 0, 0)
    H = H.astype(np.float32)

    Q_inv = np.diag(np.array([1] * K, dtype=np.float32)) * (1. / position_var)  # not affected
    print(Q_inv)
    W_inv = scipy.linalg.block_diag(Q_inv, R_inv_k)
    # print(W_inv)
    W_inv = np.delete(W_inv, 0, 0)
    W_inv = np.delete(W_inv, 0, 1)
    W_inv = W_inv.astype(np.float32)

    # print(W_inv.dtype)

    # Q = np.diag(np.array([1] * K, dtype=np.float32)) * (pos_var)
    # R = np.diag(np.array([1] * K, dtype=np.float32)) * (meas_var)
    # W = scipy.linalg.block_diag(Q, R)
    # W_inv_old = np.linalg.inv(W).astype(np.float32)
    # print(W_inv_old.dtype)

    z = np.vstack([u_k, y_kd])
    z = np.delete(z, 0, 0)
    z = z.astype(np.float32)

    lhs = H.T @ W_inv @ H
    # np.save("q4_array", lhs)
    rhs = H.T @ W_inv @ z

    x_post = np.linalg.solve(lhs, rhs)
    print(x_post)
    np.save("x_post", x_post)


def plot(x_true, t, pos_var, delta):
    x_post = np.load("x_post.npy")
    error = x_post - x_true
    uncertainty = np.sqrt(pos_var) * 3
    abs_error = np.average(np.abs(x_post) - np.abs(x_true))
    # print("Avg Error :", abs_error)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), dpi=100)

    # ax[2].plot(x_true, linewidth=0.5, label="x_true [m]")
    # ax[2].plot(x_post, linewidth=0.5, label="x_est [m]")
    # ax[2].set_ylabel("X position [m]")
    # ax[2].set_xlabel("No. of Records")
    # ax[2].legend(loc="upper left")

    ax[0].fill_between(np.squeeze(t), -uncertainty, +uncertainty, edgecolor='#CC4F1B', facecolor='#FF9848', alpha=0.5,
                       linestyle=':', label='Uncertainty Envelope')
    ax[0].plot(t, error, linewidth=0.5, label="Error [m]")
    ax[0].set_ylabel("Estimation Error [m]")
    ax[0].set_xlabel("Time [s]")
    ax[0].legend(loc="upper right")
    ax[0].set_title(f"Estimation Error versus Time for Delta : {delta}")

    ax[1].hist(error, rwidth=0.95, bins=20)
    # ax[1].axvline(x=np.max(uncert), color='r', linestyle='dashed', linewidth=2)
    # ax[1].axvline(x=-np.max(uncert), color='r', linestyle='dashed', linewidth=2)
    ax[1].set_ylabel("Count")
    ax[1].set_xlabel("Estimation Error [m]")
    ax[1].set_title(f"Histogram of Estimation Error Values for Delta :{delta}")
    plt.savefig(f"delta{delta}.png")
    # plt.show()


if __name__ == '__main__':
    data = scipy.io.loadmat('C:/Users/Shounak/Desktop/dataset1.mat')

    # Timestep
    t_step = 0.1
    # x_c is the position of the cylinder’s center
    x_c = data["l"]
    # range, r_k
    r_k = data["r"]
    # observation model
    y_k = x_c - r_k
    # true position, x_k, of the robot
    x_true = data["x_true"]
    # x_true = np.delete(x_true, 0, 0)
    print(x_true)
    # data timestamps [s]
    t = data["t"]
    # speed, u_k
    v = data["v"]
    # control inputs
    u = v * t_step
    # variance of the range readings
    r_var = data["r_var"]
    # measurement noise variance
    meas_var = r_var.item()
    # variance of the speed readings
    v_var = data["v_var"]
    # process noise variance, position variance
    pos_var = (t_step ** 2) * v_var.item()

    # Total data points
    p = x_true.shape[0]
    for delta in [1, 10, 100, 1000]:
        C, R, y = batch_est(delta, y_k, meas_var)
        # print(C, R, y)
        q5(u, pos_var, C, R, y)
        plot(x_true, t, pos_var, delta)
