import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from joblib import Parallel, delayed
import SimpleITK as itk
import os
import matplotlib.pyplot as plt

# === Eigen-decomposition for 1D Laplacian ===
def eig_1d_laplacian_matrix(n, dx):
    main_diag = -2.0 * np.ones(n)
    off_diag = np.ones(n - 1)
    L = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    eigvals, eigvecs = eigh(L / dx**2)
    return eigvecs, eigvals

# === Kronecker frequency transforms (no full Kronecker matrix) ===
def forward_transform(c, Qx, Qy, Qz):
    tmp = np.einsum('ia,abc->ibc', Qx.T, c)      # Apply Qx.T on axis 0
    tmp = np.einsum('jb,ibc->ijc', Qy.T, tmp)    # Apply Qy.T on axis 1
    c_hat = np.einsum('kc,ijc->ijk', Qz.T, tmp)  # Apply Qz.T on axis 2
    return c_hat

def inverse_transform(c_hat, Qx, Qy, Qz):
    tmp = np.einsum('ia,abc->ibc', Qx, c_hat)    # Apply Qx on axis 0
    tmp = np.einsum('jb,ibc->ijc', Qy, tmp)      # Apply Qy on axis 1
    c = np.einsum('kc,ijc->ijk', Qz, tmp)        # Apply Qz on axis 2
    return c

# === Fisher-KPP reaction ===
def reaction(c, rho, K=1.0):
    reac = rho * c * (1 - c / K)
    reac = np.nan_to_num(reac, nan=0.0, posinf=0.0, neginf=0.0)
    return reac

def step_forward_separable(c0, Qx, Qy, Qz, Lx, Ly, Lz, dt, rho):
    c_hat = forward_transform(c0, Qx, Qy, Qz)
    Nc_hat = forward_transform(reaction(c0, rho), Qx, Qy, Qz)

    result_hat = np.zeros_like(c_hat)
    for i in range(c_hat.shape[0]):
        for j in range(c_hat.shape[1]):
            for k in range(c_hat.shape[2]):
                lam = Lx[i] + Ly[j] + Lz[k]
                e_term = np.exp(-dt * lam)
                phi1 = (e_term - 1) / (dt * lam) if abs(lam) > 1e-8 else dt
                result_hat[i, j, k] = e_term * c_hat[i, j, k] + phi1 * Nc_hat[i, j, k]
    c =  inverse_transform(result_hat, Qx, Qy, Qz)
    c = np.clip(c, 0.0, 1.5)
    return c

def simulate_N_steps(c0, Qx, Qy, Qz, Lx, Ly, Lz, dt, rho, steps):
    c = c0.copy()
    for step in range(steps):
        c = step_forward_separable(c, Qx, Qy, Qz, Lx, Ly, Lz, dt, rho)
        print(f'Step {step} finished')
    return c

# === Loss & optimization ===
def joint_loss_separable(params, list_c0, list_c1, Qx, Qy, Qz, Lx, Ly, Lz, dt, steps=90, n_jobs=16):
    D, rho = params
    if D <= 0 or rho <= 0:
        return np.inf
    Lx_scaled, Ly_scaled, Lz_scaled = D * Lx, D * Ly, D * Lz

    def loss_one(c0, c1_true):
        c1_pred = simulate_N_steps(c0, Qx, Qy, Qz, Lx_scaled, Ly_scaled, Lz_scaled, dt, rho, steps)
        return np.linalg.norm(c1_pred - c1_true) ** 2

    loss_list = Parallel(n_jobs=n_jobs)(
        delayed(loss_one)(c0, c1) for c0, c1 in zip(list_c0, list_c1)
    )
    return np.sum(loss_list)

def fit_parameters_separable(list_c0_3d, list_c1_3d, dx=1.0, dt=1.0, steps=90):
    n = list_c0_3d.shape[1]
    Qx, Lx = eig_1d_laplacian_matrix(n, dx)
    Qy, Ly = eig_1d_laplacian_matrix(n, dx)
    Qz, Lz = eig_1d_laplacian_matrix(n, dx)

    initial = [0.001, 0.02]
    bounds = [(1e-5, 0.1), (1e-4, 0.1)]

    result = minimize(joint_loss_separable, initial,
                      args=(list_c0_3d, list_c1_3d, Qx, Qy, Qz, Lx, Ly, Lz, dt, steps),
                      bounds=bounds, method='L-BFGS-B')
    return result.x, result.fun, (Qx, Qy, Qz), (Lx, Ly, Lz)

list_c0_3d = np.load('/home/zengy2/isilon/Isaac/MRI_data/Simulation/data/list_c0_3d_resampled.npy')
list_c1_3d = np.load('/home/zengy2/isilon/Isaac/MRI_data/Simulation/data/list_c1_3d_resampled.npy')

training_index = int(len(list_c0_3d) * 0.8)

list_c0_3d = list_c0_3d[:training_index]
list_c1_3d = list_c1_3d[:training_index]

D, rho, loss, = None, None, None
params, loss, Qs, Ls = fit_parameters_separable(list_c0_3d, list_c1_3d)

D, rho = params
Qx, Qy, Qz = Qs
Lx, Ly, Lz = Ls

np.save("params.npy", np.array([D, rho, loss]))
np.save("Ls.npy", Ls)
np.save("Qs.npy", Qs)
