"""
Hamiltonian PINN loss for Keplerian motion
==========================================

This module defines the physics-informed loss used to train a SIREN
to reproduce Keplerian orbital dynamics.

Loss components
---------------
- Data loss: trajectory consistency
- Force loss: Newton's law at sparse collocation points
- Energy loss: Hamiltonian conservation
- Initial-condition loss

All physics is evaluated in physical (SI) units after denormalization.
"""

import torch
import numpy as np
import torch.nn.functional as F


def hamiltonian_mu(q, v, mu):
    """
    Specific orbital energy: H = 0.5*||v||^2 - mu/||q||
    q: (K,3), v: (K,3)  -> returns (K,)
    """
    kinetic = 0.5 * (v ** 2).sum(dim=1)          # (K,)
    r = q.norm(dim=1).clamp_min(1e-12)           # (K,)
    potential = -mu / r                           # (K,)
    return kinetic + potential


def loss_function(model, t, x, y, z, xdot, ydot, zdot, mu, q, phys_weight, step_size, 
                  return_pred=False,):
    # ---- Force consistent shapes: (N, 1) ----
    t    = t.view(-1, 1)
    x    = x.view(-1, 1);  y    = y.view(-1, 1);  z    = z.view(-1, 1)
    xdot = xdot.view(-1, 1); ydot = ydot.view(-1, 1); zdot = zdot.view(-1, 1)
    """
    Hamiltonian PINN loss for a 3D Keplerian orbit.
    """
    device = t.device

    # ------------------------
    # Normalization scales
    # Normalize by actual IC distance and speed so that all orbit types
    # (circular, elliptic, parabolic, hyperbolic) have O(1) normalized inputs.
    # Using q (periapsis) as scale fails for hyper/parabolic trajectories that
    # start far from periapsis: r0/q can reach 20–450, saturating the SIREN
    # input layer with pre-activations hundreds of times larger than π.
    # ------------------------
    r0     = float((x[0]**2 + y[0]**2 + z[0]**2).sum().sqrt())
    v_mag0 = float((xdot[0]**2 + ydot[0]**2 + zdot[0]**2).sum().sqrt())
    T_span = float(t.reshape(-1)[-1])
    time_scale = 1.0 / max(T_span, 1.0)

    scale_pos = torch.as_tensor(1.0 / r0,     dtype=torch.float64, device=device)
    vel_scale = torch.as_tensor(1.0 / v_mag0, dtype=torch.float64, device=device)


    # ------------------------
    # Full-trajectory input
    # ------------------------
    t_full = t.view(-1, 1) * time_scale

    x0 = (x[0:1] * scale_pos).expand(len(t), 1)
    y0 = (y[0:1] * scale_pos).expand(len(t), 1)
    z0 = (z[0:1] * scale_pos).expand(len(t), 1)

    vx0 = (xdot[0:1] * vel_scale).expand(len(t), 1)
    vy0 = (ydot[0:1] * vel_scale).expand(len(t), 1)
    vz0 = (zdot[0:1] * vel_scale).expand(len(t), 1)

    inp_full = torch.cat(
        [t_full, x0, y0, z0, vx0, vy0, vz0],
        dim=1
    )

    # ------------------------
    # Sparse collocation set
    # ------------------------
    idx = torch.tensor(step_size, dtype=torch.long, device=device)
    t_sp = t[idx] * time_scale

    inp_sparse = torch.cat(
        [
            t_sp,
            x0[idx], y0[idx], z0[idx],
            vx0[idx], vy0[idx], vz0[idx],
        ],
        dim=1
    )

    # ------------------------
    # Forward passes
    # ------------------------
    # K = number of collocation points
    idx = torch.as_tensor(step_size, dtype=torch.long, device=device)
    K = idx.numel()
    
    data_pred = model(inp_full)
    phys_pred = model(inp_sparse)

    # Unpack
    q_pred = data_pred[:, :3]
    v_pred = data_pred[:, 3:]

    q_phys = phys_pred[:, :3]
    v_phys = phys_pred[:, 3:]

    q_phys = q_phys.squeeze()
    v_phys = v_phys.squeeze()

    q_phys = q_phys.view(K,3)
    v_phys = v_phys.view(K,3)

    # ------------------------
    # Data loss
    # ------------------------
    q_target = torch.cat([x, y, z], dim=1) * scale_pos        # (N, 3)
    v_target = torch.cat([xdot, ydot, zdot], dim=1) * vel_scale  # (N, 3)

    data_loss = F.mse_loss(q_pred, q_target) + F.mse_loss(v_pred, v_target)

    # ------------------------
    # Force loss (shape-safe)
    # ------------------------
       
    # Ensure model outputs are 2D (K,6) and then (K,3)
    q_phys = q_phys.reshape(K, 3)
    v_phys = v_phys.reshape(K, 3)
    
    # If your network outputs SCALED states (as your data_loss suggests),
    # convert back to physical units for physics computations:
    # scale_pos is scalar (1/q), vel_scale is scalar (1/v0)
    q_m = q_phys / scale_pos          # (K,3) physical position
    # v_m = v_phys / vel_scale        # (K,3) physical velocity if needed later
    
    r = q_m.norm(dim=1, keepdim=True) # (K,1)
    accel_pred = -mu * q_m / (r**3 + 1e-6)  # (K,3)
    
    # Build true position at collocation points as (K,3) using cat (not stack)
    xk = x[idx].view(K, 1)
    yk = y[idx].view(K, 1)
    zk = z[idx].view(K, 1)
    q_true = torch.cat([xk, yk, zk], dim=1)  # (K,3)
    
    r_true = q_true.norm(dim=1, keepdim=True)  # (K,1)
    accel_true = -mu * q_true / (r_true**3 + 1e-6)  # (K,3)
    
    # Tripwires (remove after it runs once)
    assert accel_pred.shape == accel_true.shape, (accel_pred.shape, accel_true.shape)
    
    accel_scale = (accel_true ** 2).mean().detach() + 1e-20
    force_loss = (accel_pred - accel_true).pow(2).mean() / accel_scale

    # ------------------------
    # ------------------------
    # Energy loss (shape-safe)
    # ------------------------
    
    # Ensure scaled model outputs are (K,3)
    q_phys = q_phys.squeeze().contiguous().view(K, 3)
    v_phys = v_phys.squeeze().contiguous().view(K, 3)
    
    # Ensure scalar scales (prevents accidental (N,1) broadcasting)
    scale_pos = torch.as_tensor(scale_pos, dtype=q_phys.dtype, device=device).reshape(())
    vel_scale = torch.as_tensor(vel_scale, dtype=v_phys.dtype, device=device).reshape(())
    
    # Convert model outputs back to physical units
    q_m = q_phys / scale_pos                          # (K,3)
    v_m = v_phys / vel_scale                          # (K,3)
    
    # Build true physical states at collocation points as (K,3) using cat
    xk  = x[idx].view(K, 1)
    yk  = y[idx].view(K, 1)
    zk  = z[idx].view(K, 1)
    vxk = xdot[idx].view(K, 1)
    vyk = ydot[idx].view(K, 1)
    vzk = zdot[idx].view(K, 1)
    
    q_true = torch.cat([xk, yk, zk], dim=1)           # (K,3)
    v_true = torch.cat([vxk, vyk, vzk], dim=1)        # (K,3)
    
    # Reference energy: use the true initial condition at t=0 (recommended)
    q0_true = torch.cat([x[:1].view(1,1), y[:1].view(1,1), z[:1].view(1,1)], dim=1)       # (1,3)
    v0_true = torch.cat([xdot[:1].view(1,1), ydot[:1].view(1,1), zdot[:1].view(1,1)], dim=1)  # (1,3)
    
    H0 = hamiltonian_mu(q0_true, v0_true, mu).squeeze()   # scalar
    H  = hamiltonian_mu(q_m, v_m, mu)                     # (K,)

    # Tripwires (keep until it runs)
    assert H.dim() == 1 and H.shape[0] == K, H.shape
    assert H0.dim() == 0, H0.shape

    # Normalise by kinetic energy at t=0 — always positive, safe for parabolic
    # orbits where H0 = 0 (using H0 as denominator would overflow to 10^32+)
    ke0 = (0.5 * (v0_true ** 2).sum()).abs() + 1e-12
    energy_loss = ((H - H0) / ke0).pow(2).mean()

    # ------------------------
    # Initial condition loss
    # ------------------------
    ic_loss = (
        (q_pred[0, 0] - x[0, 0] * scale_pos) ** 2 +
        (q_pred[0, 1] - y[0, 0] * scale_pos) ** 2 +
        (q_pred[0, 2] - z[0, 0] * scale_pos) ** 2 +
        (v_pred[0, 0] - xdot[0, 0] * vel_scale) ** 2 +
        (v_pred[0, 1] - ydot[0, 0] * vel_scale) ** 2 +
        (v_pred[0, 2] - zdot[0, 0] * vel_scale) ** 2
    )

    physics_loss = force_loss + energy_loss

    total_loss = (
        100.0 * data_loss +
        phys_weight * physics_loss +
        10.0 * ic_loss
    )

    
    if return_pred:
        return total_loss, q_pred
    else:
        return total_loss