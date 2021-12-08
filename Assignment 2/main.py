import numpy as np
import scipy.io
import scipy.linalg

def q4(u, y_k, pos_var, meas_var):


    #r_var = var.r_var
    #v_var = var.v_var
    #print(r_var)
    K = 12709
    # A = np.tril(np.ones((K, K), dtype=np.float32), 0)
    # C = np.diag(np.array([1] * K, dtype=np.float32))
    # H = np.vstack([np.linalg.inv(A), C])
    # H = H.astype(np.float32)

    Q_inv = np.diag(np.array([1] * K, dtype=np.float32)) * (1/pos_var)
    R_inv = np.diag(np.array([1] * K, dtype=np.float32)) * (1/meas_var)
    W_inv = scipy.linalg.block_diag(Q_inv, R_inv)
    W_inv = W_inv.astype(np.float32)
    print(W_inv.dtype)

    Q = np.diag(np.array([1] * K, dtype=np.float32)) * (pos_var)
    R = np.diag(np.array([1] * K, dtype=np.float32)) * (meas_var)
    W = scipy.linalg.block_diag(Q, R)
    W_inv_old = np.linalg.inv(W).astype(np.float32)
    print(W_inv_old.dtype)

    z = np.vstack([u,y_k])

    print(z.dtype)
    #W_inv = np.linalg.inv(W).astype(np.float32)

    lhs = H.T @ W_inv @ H
    #np.save("q4_array", lhs)
    rhs = H.T @ W_inv @ z

    x = np.linalg.solve(lhs, rhs)
    print(x)

if __name__ == '__main__':
    data = scipy.io.loadmat("dataset1.mat")

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

    q4(u, y_k, pos_var, meas_var)































