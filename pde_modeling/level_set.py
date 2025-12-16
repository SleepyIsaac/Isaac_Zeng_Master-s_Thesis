from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import distance_transform_edt, gaussian_filter

def _roll_diff_forward(a: np.ndarray, axis: int, h: float) -> np.ndarray:
    out = (np.roll(a, -1, axis=axis) - a) / h
    # Zero gradient at the last index
    slicer = [slice(None)] * a.ndim
    slicer[axis] = -1
    out[tuple(slicer)] = 0.0
    return out

def _roll_diff_backward(a: np.ndarray, axis: int, h: float) -> np.ndarray:
    out = (a - np.roll(a, 1, axis=axis)) / h
    slicer = [slice(None)] * a.ndim
    slicer[axis] = 0
    out[tuple(slicer)] = 0.0
    return out

def upwind_gradient(phi: np.ndarray, spacing: Tuple[float, ...], speed: np.ndarray) -> Tuple[np.ndarray, ...]:

    axes = list(range(phi.ndim))
    h = spacing

    grads = []
    for ax in axes:
        Dp = _roll_diff_forward(phi, ax, h[ax])
        Dm = _roll_diff_backward(phi, ax, h[ax])
        g = np.where(speed >= 0.0, Dm, Dp)
        grads.append(g)
    return tuple(grads)

def signed_distance_from_mask(mask: np.ndarray, spacing: Tuple[float, ...]) -> np.ndarray:
    if distance_transform_edt is None:
        raise RuntimeError("scipy.ndimage.distance_transform_edt is required.")
    mask = (mask > 0).astype(np.uint8)
    dist_out = distance_transform_edt(mask == 0, sampling=spacing)  # outside -> distance to mask
    dist_in = distance_transform_edt(mask == 1, sampling=spacing)   # inside  -> distance to outside
    sdf = dist_out - dist_in
    return sdf

def normal_field_from_brain(brain_mask: np.ndarray, spacing: Tuple[float, ...], smooth_sigma: Optional[float] = 0.0) -> Tuple[np.ndarray, ...]:
    phi_brain = signed_distance_from_mask(brain_mask, spacing)
    if smooth_sigma and smooth_sigma > 0 and gaussian_filter is not None:
        # Optional regularization to avoid noisy normals
        # smooth_sigma is in *voxels*. If want mm, divide by spacing.
        phi_brain = gaussian_filter(phi_brain, sigma=smooth_sigma)

    grads = []
    for ax, h in enumerate(spacing):
        grads.append(_roll_diff_forward(phi_brain, ax, h) + _roll_diff_backward(phi_brain, ax, h))
    gx, gy, gz = (grads + [np.zeros_like(phi_brain)] * (3 - len(grads)))[:3]

    norm = np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-12
    nx, ny, nz = gx / norm, gy / norm, gz / norm
    return nx, ny, nz

def opposition_field(brain_mask: np.ndarray, spacing: Tuple[float, ...], sigma_mm: float = 5.0, mode: str = "increasing") -> np.ndarray:
    if distance_transform_edt is None:
        raise RuntimeError("scipy.ndimage.distance_transform_edt is required.")

    brain = (brain_mask > 0).astype(np.uint8)
    # Distance to boundary on both sides (always >=0)
    d_in = distance_transform_edt(brain == 1, sampling=spacing)      # inside brain to outside
    d_out = distance_transform_edt(brain == 0, sampling=spacing)     # outside brain to inside
    d = np.where(brain == 1, d_in, d_out)

    if sigma_mm <= 0:
        raise ValueError("sigma_mm must be > 0")

    if mode == "increasing":
        P = 1.0 - np.exp(-d / float(sigma_mm))
    elif mode == "exp_negative":
        P = np.exp(-d / float(sigma_mm))
    else:
        raise ValueError("mode must be 'increasing' or 'exp_negative'")
    return P.astype(np.float32)

def anisotropic_norm(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray,
                     nx: np.ndarray, ny: np.ndarray, nz: np.ndarray,
                     a_parallel: float, a_perp: float) -> np.ndarray:
    g2 = gx * gx + gy * gy + gz * gz
    gdotn = gx * nx + gy * ny + gz * nz
    return np.sqrt(np.maximum(0.0, a_perp * g2 + (a_parallel - a_perp) * gdotn * gdotn) + 1e-12)

def reinitialize(phi: np.ndarray, spacing: Tuple[float, ...], iters: int = 25, dtau: float = 0.3) -> np.ndarray:
    eps = 1.0
    sgn = phi / np.sqrt(phi * phi + eps)
    for _ in range(iters):
        # Forward and backward diffs in all axes
        Dxp = _roll_diff_forward(phi, -1 if phi.ndim == 1 else 0, spacing[0])
        Dxm = _roll_diff_backward(phi, -1 if phi.ndim == 1 else 0, spacing[0])
        if phi.ndim >= 2:
            Dyp = _roll_diff_forward(phi, 1, spacing[1])
            Dym = _roll_diff_backward(phi, 1, spacing[1])
        else:
            Dyp = Dym = 0.0
        if phi.ndim == 3:
            Dzp = _roll_diff_forward(phi, 2, spacing[2])
            Dzm = _roll_diff_backward(phi, 2, spacing[2])
        else:
            Dzp = Dzm = 0.0

        # Godunov's scheme for |grad| in reinit eqn
        a_plus = np.sqrt(np.maximum(Dxm, 0) ** 2 + np.minimum(Dxp, 0) ** 2 +
                         (np.maximum(Dym, 0) ** 2 + np.minimum(Dyp, 0) ** 2 if phi.ndim >= 2 else 0) +
                         (np.maximum(Dzm, 0) ** 2 + np.minimum(Dzp, 0) ** 2 if phi.ndim == 3 else 0))
        a_minus = np.sqrt(np.maximum(Dxp, 0) ** 2 + np.minimum(Dxm, 0) ** 2 +
                          (np.maximum(Dyp, 0) ** 2 + np.minimum(Dym, 0) ** 2 if phi.ndim >= 2 else 0) +
                          (np.maximum(Dzp, 0) ** 2 + np.minimum(Dzm, 0) ** 2 if phi.ndim == 3 else 0))
        grad_norm = np.where(sgn >= 0, a_plus, a_minus)
        phi = phi - dtau * (np.sign(sgn) * (grad_norm - 1.0))
    return phi


def evolve_level_set(
        S0: np.ndarray,
        brain: np.ndarray,
        spacing: Tuple[float, ...],
        t_final: float,
        v0: float = 0.25,
        alpha: float = 0.9,
        sigma_mm: float = 6.0,
        a_parallel: float = 1.0,
        a_perp: float = 0.2,
        cfl: float = 0.9,
        reinit_every: Optional[int] = 15,
        P_mode: str = "increasing",
        smooth_normals_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    assert S0.shape == brain.shape, "S0 and brain must have same shape"
    spacing = tuple(float(x) for x in spacing)
    ndim = S0.ndim
    if ndim not in (2, 3):
        raise ValueError("Only 2D or 3D arrays are supported")

    # Initialize phi from tumor mask (negative inside)
    phi = signed_distance_from_mask(S0.astype(np.uint8), spacing)

    # Tissue opposition and normal field
    P = opposition_field(brain, spacing, sigma_mm=sigma_mm, mode=P_mode)
    nx, ny, nz = normal_field_from_brain(brain, spacing, smooth_sigma=smooth_normals_sigma)

    # Time stepping
    t = 0.0
    step = 0
    while t < t_final - 1e-12:
        # Local speed field
        speed = (v0 - alpha * P).astype(np.float32)

        # CFL time step bound (use the strongest metric eigenvalue = a_parallel)
        max_speed = float(np.max(np.abs(speed)))
        if max_speed < 1e-12:
            break
        dt = cfl * min(spacing) / (np.sqrt(a_parallel) * max_speed)
        if t + dt > t_final:
            dt = t_final - t

        # Upwind gradient
        gx, gy, gz = upwind_gradient(phi, spacing, speed)
        if ndim == 2:
            gz = 0.0

        # Anisotropic metric norm
        grad_norm = anisotropic_norm(gx, gy, gz, nx, ny, nz, a_parallel, a_perp)

        # HJ update: phi_t = -speed * ||grad||_A
        phi = phi - dt * speed * grad_norm

        # Optional reinitialization
        step += 1
        if reinit_every and (step % reinit_every == 0):
            phi = reinitialize(phi, spacing, iters=20, dtau=0.2)

        t += dt

    S1 = (phi <= 0).astype(np.uint8)
    return phi, S1


def load_nii(path: str) -> Tuple[np.ndarray, Tuple[float, ...], np.ndarray]:
    import nibabel as nib
    img = nib.load(path)
    data = np.asarray(img.get_fdata())
    # Convert to C‑order (Z, Y, X) with spacing from affine
    affine = img.affine
    zooms = tuple(float(z) for z in img.header.get_zooms()[: data.ndim])
    return data, zooms, affine


def save_nii(path: str, data: np.ndarray, affine: Optional[np.ndarray] = None):  # pragma: no cover
    import nibabel as nib
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, path)
