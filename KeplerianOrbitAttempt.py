"""
Hamiltonian SIREN architecture for Keplerian orbits
===================================================

This module defines a conditional SIREN (Sinusoidal Representation Network)
used for learning Keplerian orbital dynamics with Hamiltonian-structured
outputs.

Model map
---------

The neural network approximates:

    (t_norm, q0_norm, v0_norm)  ->  (q_norm(t), v_norm(t))

where:
- t_norm is a normalized time coordinate
- q0_norm, v0_norm are the normalized initial position/velocity (IC)
- q_norm(t), v_norm(t) are the normalized predicted position/velocity at time t

Key ideas
---------

- Predicting both position and velocity enables physics terms such as:
  - Force consistency (Newton's law) at sparse points
  - Energy conservation (Hamiltonian invariance) at sparse points
  - Initial-condition enforcement (q(0)=q0, v(0)=v0)

SIREN initialization
--------------------

Initialization follows Sitzmann et al. (2020):

- First layer:
      W ~ Uniform(-1/n_in, 1/n_in)
- Hidden layers:
      W ~ Uniform(-sqrt(6/n_in)/omega_0, sqrt(6/n_in)/omega_0)

Extending to N-body
-------------------

The architecture is body-count agnostic. For n bodies:

- in_features  = 1 + 6 * n_bodies
- out_features = 6 * n_bodies

Only the Hamiltonian / physics terms need updating accordingly.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    """
    Fully-connected layer with sine activation.

    The layer computes:

        y = sin(omega_0 * (W x + b))

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    bias : bool, default=True
        Whether to include a bias term.
    is_first : bool, default=False
        Whether this is the first SIREN layer (uses first-layer init).
    omega_0 : float, default=30.0
        Frequency scale factor.

    Notes
    -----
    The special first-layer initialization is used to keep pre-activations
    in a reasonable range for normalized inputs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.is_first = bool(is_first)
        self.omega_0 = float(omega_0)

        self.linear = nn.Linear(self.in_features, int(out_features), bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights following SIREN conventions."""
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.sin(self.omega_0 * self.linear(x))

    def forward_with_intermediate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both output and pre-activation.

        Returns
        -------
        y : torch.Tensor
            Activated output.
        pre : torch.Tensor
            Pre-activation tensor (omega_0 * (W x + b)), useful for diagnostics.
        """
        pre = self.omega_0 * self.linear(x)
        return torch.sin(pre), pre


class SIREN(nn.Module):
    """
    Conditional SIREN network for orbit learning.

    Parameters
    ----------
    in_features : int
        Input dimension (typically 7: t_norm + 6 IC components).
    hidden_features : int
        Width of hidden layers.
    hidden_layers : int
        Number of hidden layers.
    out_features : int
        Output dimension (typically 6: q_norm (3) + v_norm (3)).
    outermost_linear : bool, default=True
        If True, last layer is linear (no sine activation).
    first_omega_0 : float, default=30.0
        omega_0 for the first layer.
    hidden_omega_0 : float, default=30.0
        omega_0 for hidden sine layers.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = True,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        layers.append(
            SineLayer(
                in_features=in_features,
                out_features=hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )
        )

        for _ in range(int(hidden_layers)):
            layers.append(
                SineLayer(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6.0 / hidden_features) / hidden_omega_0
                final.weight.uniform_(-bound, bound)
            layers.append(final)
        else:
            layers.append(
                SineLayer(
                    in_features=hidden_features,
                    out_features=out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        coords : torch.Tensor, shape (N, in_features)
            Concatenated input (t_norm, q0_norm, v0_norm).

        Returns
        -------
        torch.Tensor, shape (N, out_features)
            Output where:
            - out[:, :3] is q_norm(t)
            - out[:, 3:] is v_norm(t)
        """
        return self.net(coords)

    def forward_with_intermediates(self, coords: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with intermediate pre-activations (for diagnostics/plots).

        Returns
        -------
        out : torch.Tensor
            Model output.
        intermediates : list[torch.Tensor]
            List of pre-activation tensors from each SineLayer.
        """
        intermediates: List[torch.Tensor] = []
        x = coords
        for layer in self.net:
            if isinstance(layer, SineLayer):
                x, pre = layer.forward_with_intermediate(x)
                intermediates.append(pre)
            else:
                x = layer(x)
        return x, intermediates
    

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
    

import torch
import numpy as np
import time
import os
import pickle
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
from scimba_torch.optimizers.scimba_optimizers import ScimbaAdam, ScimbaSSBFGS, ScimbaLBFGS, ScimbaSSBroyden
from keplerian_orbit_plotting import plot_result, save_gif_PIL, plot_loss, plot_activations, plot_weights



# ------------------------
# Hyperparameters
# ------------------------

trials = 100
outer_steps = 30
adam_steps = 1000
phys_weight = 1.0
plot_every  = 1
model_reload = True


# ------------------------
# Output directories
# ------------------------

time0 = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run_reason = (
    f"{trials}trials_{outer_steps}outersteps_{adam_steps}adamsteps_"
    f"{'reload' if model_reload else 'fresh'}" + time.strftime("%m_%d_%H_%M")
)
os.makedirs(f"plots/attempt/{run_reason}",             exist_ok=True)
os.makedirs(f"plots/attempt/{run_reason}/short",       exist_ok=True)
os.makedirs(f"plots/attempt/{run_reason}/weights",     exist_ok=True)
os.makedirs(f"plots/attempt/{run_reason}/activations", exist_ok=True)


# ------------------------
# Data collection
# ------------------------

cache = {"Used_files": []}

def data_collection():
    with open('data/keplerian_orbit_datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    while True:
        random_num = np.random.randint(0, len(datasets))
        if random_num not in cache["Used_files"]:
            cache["Used_files"].append(random_num)
            break
    data = datasets[random_num]
    t    = np.array(data.get("t"),  dtype=np.float64).reshape(-1, 1)
    x    = np.array(data.get("x"),  dtype=np.float64).reshape(-1, 1)
    y    = np.array(data.get("y"),  dtype=np.float64).reshape(-1, 1)
    z    = np.array(data.get("z"),  dtype=np.float64).reshape(-1, 1)
    xdot = np.array(data.get("vx"), dtype=np.float64).reshape(-1, 1)
    ydot = np.array(data.get("vy"), dtype=np.float64).reshape(-1, 1)
    zdot = np.array(data.get("vz"), dtype=np.float64).reshape(-1, 1)
    return [t, x, y, z, xdot, ydot, zdot,
            data.get("orbit_type"), data.get("e"),
            data.get("q"), data.get("p"), data.get("mu")]


# ------------------------
# Model + optimizer setup
# ------------------------

def keplerian_orbit_setup(device, t, x, y, z, xdot, ydot, zdot, learning_rate=1e-3):
    t_t    = torch.tensor(t,    dtype=torch.float64, device=device)
    x_t    = torch.tensor(x,    dtype=torch.float64, device=device)
    y_t    = torch.tensor(y,    dtype=torch.float64, device=device)
    z_t    = torch.tensor(z,    dtype=torch.float64, device=device)
    xdot_t = torch.tensor(xdot, dtype=torch.float64, device=device)
    ydot_t = torch.tensor(ydot, dtype=torch.float64, device=device)
    zdot_t = torch.tensor(zdot, dtype=torch.float64, device=device)

    model = SIREN(in_features=7, hidden_features=32,
                  hidden_layers=3, out_features=6, first_omega_0=15).to(device).double()

    ckpt_path = "keplerian_Orbit_PINN_attempt.pth"
    if os.path.exists(ckpt_path):
        print(f"[Resume] Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print("[Resume] Missing keys:", missing)
        if unexpected:
            print("[Resume] Unexpected keys:", unexpected)
        print("[Resume] Optimizer state reset for new IC.")
    else:
        print("[Resume] No checkpoint found; starting fresh.")

    Adam   = ScimbaAdam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    LBFGS = ScimbaLBFGS(model.parameters(), lr=1.0, history_size=10)

    return model, Adam, LBFGS, t_t, x_t, y_t, z_t, xdot_t, ydot_t, zdot_t


# ------------------------
# Training loop
# ------------------------

total_updates = (trials - 1) * outer_steps * adam_steps
overall_pbar  = tqdm(total=total_updates, desc="Overall progress", position=0)

for n in range(1, trials):
    files              = []
    short_files        = []
    act_files          = []
    weight_files       = []
    loss_list          = []
    activation_history = []
    weight_history     = []

    data = data_collection()
    t, x, y, z, xdot, ydot, zdot, orbit_type, ecc, q, p, mu = (
        data[0], data[1], data[2], data[3], data[4], data[5],
        data[6], data[7], data[8], data[9], data[10], data[11]
    )

    lr_decayed = 3e-4 / (1 + 0.02 * n)
    model, Adam, LBFGS, t_t, x_t, y_t, z_t, xdot_t, ydot_t, zdot_t = \
        keplerian_orbit_setup(device, t, x, y, z, xdot, ydot, zdot, learning_rate=lr_decayed)
    model.train()

    # Collocation indices (evenly spaced, fixed across all outer steps)
    trial_indice = np.linspace(0, len(t_t) - 1, 10, dtype=int)

    # Physical scales — must match normalization in loss_function.
    # Use actual IC distance/speed so all orbit types give O(1) inputs.
    r0_f      = float(np.sqrt(x[0, 0]**2 + y[0, 0]**2 + z[0, 0]**2))
    v_mag0_f  = float(np.sqrt(xdot[0, 0]**2 + ydot[0, 0]**2 + zdot[0, 0]**2))
    pos_scale = 1.0 / r0_f
    v0_scale  = v_mag0_f     # divisor for velocity: xdot0_norm = xdot[0] / v0_scale
    T_span    = float(t_t.reshape(-1)[-1])

    for i in range(outer_steps):

        # Decay lr across outer steps: full rate early, half rate by the last step
        lr_outer = lr_decayed / (1.0 + i / outer_steps)
        for param_group in Adam.param_groups:
            param_group["lr"] = lr_outer

        # ---- Adam pretraining -----------------------------------------------
        for m in range(adam_steps):
            def pinn_loss():
                Adam.zero_grad()
                loss = loss_function(
                    model, t_t, x_t, y_t, z_t, xdot_t, ydot_t, zdot_t,
                    mu, q, phys_weight, step_size=trial_indice,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                return loss

            loss = Adam.step(pinn_loss)
            if loss is not None and torch.isfinite(loss):
                overall_pbar.update(1)
                loss_list.append(float(loss))

        # ---- SSBFGS (uncomment to enable) -----------------------------------
        
        def pinn_loss_broyden():
            LBFGS.zero_grad(set_to_none=True)
            loss = loss_function(
                model, t_t, x_t, y_t, z_t, xdot_t, ydot_t, zdot_t,
                mu, q, 1e-2, step_size=trial_indice, return_pred=False,
            )
            loss.backward()
            return loss
        for _ in range(10):
            loss = LBFGS.step(pinn_loss_broyden)


        # ---- Predictions for plotting ---------------------------------------
        with torch.no_grad():
            _, q_pred_norm = loss_function(
                model, t_t, x_t, y_t, z_t, xdot_t, ydot_t, zdot_t,
                mu, q, phys_weight, step_size=trial_indice, return_pred=True,
            )
        pred_orbit = (q_pred_norm / pos_scale).cpu().numpy()   # (N, 3) meters

        # True state at collocation points for scatter overlay
        idx = trial_indice
        physics_data = np.column_stack([
            t_t[idx].cpu().numpy(),
            x_t[idx].cpu().numpy(),    y_t[idx].cpu().numpy(),    z_t[idx].cpu().numpy(),
            xdot_t[idx].cpu().numpy(), ydot_t[idx].cpu().numpy(), zdot_t[idx].cpu().numpy(),
        ])

        # ---- Plotting -------------------------------------------------------
        if i % plot_every == 0:
            model.eval()
            loss_val   = loss_list[-1] if loss_list else float("inf")
            t_np       = t_t.detach().cpu().numpy()
            data_orbit = np.column_stack([
                x_t.detach().cpu().numpy(),
                y_t.detach().cpu().numpy(),
                z_t.detach().cpu().numpy(),
            ])

            # Full-trajectory plot
            plot_result(t_np, data_orbit, pred_orbit, physics_data, i, loss_val)
            file = f"plots/attempt/{run_reason}/nn_%.8i.png" % (i + 1)
            plt.savefig(file, pad_inches=0.1, facecolor="white")
            files.append(file)
            plt.close()

            # Short (zoom) plot
            zoom_n     = min(2000, len(t_np))
            t_zoom_max = float(t_np[zoom_n - 1, 0])
            phys_mask  = physics_data[:, 0] <= t_zoom_max
            plot_result(t_np[:zoom_n], data_orbit[:zoom_n], pred_orbit[:zoom_n],
                        physics_data[phys_mask], i, loss_val)
            file = f"plots/attempt/{run_reason}/short/nn_%.8i.png" % (i + 1)
            plt.savefig(file, pad_inches=0.1, facecolor="white")
            short_files.append(file)
            plt.close()

            # Activation distributions
            with torch.no_grad():
                t_norm_act  = t_t / T_span
                x0_act      = (x_t[0:1]    * pos_scale).expand(len(t_t), 1)
                y0_act      = (y_t[0:1]    * pos_scale).expand(len(t_t), 1)
                z0_act      = (z_t[0:1]    * pos_scale).expand(len(t_t), 1)
                xdot0_act   = (xdot_t[0:1] / v0_scale).expand(len(t_t), 1)
                ydot0_act   = (ydot_t[0:1] / v0_scale).expand(len(t_t), 1)
                zdot0_act   = (zdot_t[0:1] / v0_scale).expand(len(t_t), 1)
                inp_act     = torch.cat([t_norm_act, x0_act, y0_act, z0_act,
                                         xdot0_act, ydot0_act, zdot0_act], dim=1)
                _, intermediates = model.forward_with_intermediates(inp_act)
                activation_history.append(
                    [inter.detach().cpu().numpy() for inter in intermediates])

            plot_activations(activation_history, i, n)
            file = f"plots/attempt/{run_reason}/activations/activations_%.8i.png" % (i + 1)
            plt.savefig(file, pad_inches=0.1, facecolor="white")
            act_files.append(file)
            plt.close()

            # Weight distributions
            with torch.no_grad():
                layer_weights = []
                for layer in model.net:
                    if hasattr(layer, 'linear'):
                        layer_weights.append(layer.linear.weight.detach().cpu().numpy())
                    elif hasattr(layer, 'weight'):
                        layer_weights.append(layer.weight.detach().cpu().numpy())
                weight_history.append(layer_weights)

            plot_weights(weight_history, i, n)
            file = f"plots/attempt/{run_reason}/weights/weights_%.8i.png" % (i + 1)
            plt.savefig(file, pad_inches=0.1, facecolor="white")
            weight_files.append(file)
            plt.close()

            model.train()

        tol = 1e-5
        if loss_list and abs(loss_list[-1]) < tol:
            print(f"Convergence at trial {n}, outer step {i}, loss {loss_list[-1]:.6e}")
            break

    # ---- End of trial: GIFs, loss curve, checkpoint ------------------------
    plot_loss(loss_list, n)
    plt.savefig(f"plots/attempt/{run_reason}/loss_curve_%.8i.png" % (n + 1),
                pad_inches=0.1, facecolor="white")
    plt.close()

    fps_gif = 700
    save_gif_PIL(f"plots/attempt/{run_reason}/pinn_trial{n}.gif",
                 files, fps_gif, loop=0)
    save_gif_PIL(f"plots/attempt/{run_reason}/short/pinn_trial{n}_short.gif",
                 short_files, fps_gif, loop=0)
    save_gif_PIL(f"plots/attempt/{run_reason}/activations/activations_trial{n}.gif",
                 act_files, fps_gif, loop=0)
    save_gif_PIL(f"plots/attempt/{run_reason}/weights/weights_trial{n}.gif",
                 weight_files, fps_gif, loop=0)

    torch.save({"model": model.state_dict(), "adam": Adam.state_dict()},
               "keplerian_Orbit_PINN_attempt.pth")

print(f"Training completed in {time.time() - time0:.2f}s")
torch.save({"model": model.state_dict(), "adam": Adam.state_dict()},
           "keplerian_Orbit_PINN_attempt.pth")
print("Model saved.")