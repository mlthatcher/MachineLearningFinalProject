   

import torch
import numpy as np
import time
import os
import pickle
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
from scimba_torch.optimizers.scimba_optimizers import ScimbaAdam, ScimbaSSBFGS, ScimbaLBFGS, ScimbaSSBroyden
from keplerian_orbit_plotting import plot_result, save_gif_PIL, plot_loss, plot_activations, plot_weights

from KeplerianOrbitStructure import SIREN
from KeplerianOrbitLoss import loss_function



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