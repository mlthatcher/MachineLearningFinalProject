#Plotting code for the results of the ODE solver, used in ODE_Solver_copy3.py
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np

#Used to compile and save the gifs of the training process.
def save_gif_PIL(filename, files, fps=20, loop=0):
    if not files:
        print("No frames to save for GIF.")
        return
    imgs = [Image.open(file) for file in files]
    imgs[0].save(filename, format='GIF', append_images = imgs[1:], save_all = True, duration = int(1000/fps), loop = loop)
    for im in imgs:
        im.close()


def plot_result(t, data_orbit, pred_orbit, physics_data, j, loss_val):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111, projection='3d')


    #Plot Earth
    radius = 6371000
    # 1. Create the meshgrid for spherical angles
    u = np.linspace(0, 2 * np.pi, 100) # Azimuthal angle
    v = np.linspace(0, np.pi, 100)     # Polar angle
    # 2. Convert Spherical to Cartesian coordinates
    # For a sphere with radius r=1 centered at (0,0,0)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    # 3. Plot the surface
    ax.plot_surface(x, y, z, color='b', alpha=0.5)

    # Ensure all tensors are detached and converted to numpy
    true_t = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
    true_orbit = data_orbit.detach().cpu().numpy() if isinstance(data_orbit, torch.Tensor) else data_orbit
    pred_orbit = pred_orbit.detach().cpu().numpy() if isinstance(pred_orbit, torch.Tensor) else pred_orbit
    physics_data = physics_data.detach().cpu().numpy() if isinstance(physics_data, torch.Tensor) else physics_data

    fig.suptitle(f"Training step: {j}\nloss: {loss_val:.6f}", fontsize="x-large", color="k")
    ax.set_title("Circular Orbit around Spherical Earth", fontsize="large")
    ax.set_xlabel("X", fontsize="large")
    ax.set_ylabel("Y", fontsize="large")
    ax.set_zlabel("Z", fontsize="large")
    ax.set_xlim([true_orbit[:, 0].min() - 0.05, true_orbit[:, 0].max() + 0.05])
    ax.set_ylim([true_orbit[:, 1].min() - 0.05, true_orbit[:, 1].max() + 0.05])
    ax.set_zlim([true_orbit[:, 2].min() - 0.05, true_orbit[:, 2].max() + 0.05])
    ax.axis('off')
    ax.plot(true_orbit[:, 0], true_orbit[:, 1], true_orbit[:, 2], color="grey", linewidth=2, alpha=0.8, label="Exact Solution")
    ax.plot(pred_orbit[:, 0], pred_orbit[:, 1], pred_orbit[:, 2], color="tab:blue", linewidth=4, alpha=0.8, label="Neural Network Prediction")
    ax.scatter(physics_data[:, 1], physics_data[:, 2], physics_data[:, 3], color="tab:orange", s=50, alpha=0.8, label="Physics Loss Data Points")
    ax.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    ax.set_aspect('equal')
    fig.tight_layout()

def plot_activations(activation_history, step, n):
    # activation_history: list (one entry per outer step so far) of lists
    # (one numpy array per SineLayer) of pre-activation values shape [N, hidden].
    # Plots overlaid histograms per layer, coloured light→dark as training progresses.
    n_layers = len(activation_history[0])
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 3))
    if n_layers == 1:
        axes = [axes]

    cmap = plt.cm.plasma
    n_steps = len(activation_history)
    layer_labels = ["Input layer"] + [f"Hidden layer {k}" for k in range(1, n_layers)]

    for layer_idx, ax in enumerate(axes):
        for s_idx, intermediates in enumerate(activation_history):
            frac = s_idx / max(n_steps - 1, 1)
            color = cmap(frac)
            vals = intermediates[layer_idx].flatten()
            ax.hist(vals, bins=60, density=True, alpha=0.4,
                    color=color, histtype="stepfilled", linewidth=0)

        ax.axvline(-np.pi, color="red", linestyle="--", linewidth=1, alpha=0.8)
        ax.axvline( np.pi, color="red", linestyle="--", linewidth=1, alpha=0.8,
                   label="±π")
        ax.set_title(layer_labels[layer_idx], fontsize=10)
        ax.set_xlabel("ω·Wx + b  (pre-activation)", fontsize=8)
        if layer_idx == 0:
            ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8, frameon=False)

    # Colourbar showing step progression
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=max(n_steps - 1, 1)))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[-1], label="Outer step", fraction=0.046, pad=0.04)
    fig.suptitle(f"Trial {n} | Outer step {step} — Pre-activation distributions",
                 fontsize=11)
    fig.tight_layout()


def plot_weights(weight_history, step, n):
    # weight_history: list (one entry per outer step so far) of lists
    # (one numpy array per layer) of weight values shape [out, in].
    # Plots overlaid histograms per layer, coloured light→dark as training progresses.
    n_layers = len(weight_history[0])
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 3))
    if n_layers == 1:
        axes = [axes]

    cmap = plt.cm.viridis
    n_steps = len(weight_history)
    layer_labels = ["Input layer"] + [f"Hidden layer {k}" for k in range(1, n_layers - 1)] + ["Output layer"]

    for layer_idx, ax in enumerate(axes):
        for s_idx, weights in enumerate(weight_history):
            frac = s_idx / max(n_steps - 1, 1)
            color = cmap(frac)
            vals = weights[layer_idx].flatten()
            ax.hist(vals, bins=60, density=True, alpha=0.2,
                    color=color, histtype="stepfilled", linewidth=1.0)

        ax.set_title(layer_labels[layer_idx], fontsize=10)
        ax.set_xlabel("Weight value", fontsize=8)
        ax.set_ylim([0, 80])
        ax.set_xlim([-0.2, 0.2])
        if layer_idx == 0:
            ax.set_ylabel("Density", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=max(n_steps - 1, 1)))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[-1], label="Outer step", fraction=0.046, pad=0.04)
    fig.suptitle(f"Trial {n} | Outer step {step} — Weight distributions",
                 fontsize=11)
    fig.tight_layout()

def plot_long_weights(long_weight_history_files):
    # long_weight_history_files: list of file paths to saved weight history numpy arrays.
    # Plots the evolution of weight distributions across all saved checkpoints.
    fig, axes = plt.subplots(1, len(long_weight_history_files[0]), figsize=(4 * len(long_weight_history_files[0]), 3))
    if len(long_weight_history_files[0]) == 1:
        axes = [axes]

    cmap = plt.cm.viridis
    n_steps = len(long_weight_history_files)
    layer_labels = ["Input layer"] + [f"Hidden layer {k}" for k in range(1, len(long_weight_history_files[0]) - 1)] + ["Output layer"]

    for layer_idx, ax in enumerate(axes):
        for s_idx, file in enumerate(long_weight_history_files):
            weights = np.load(file)[layer_idx]
            frac = s_idx / max(n_steps - 1, 1)
            color = cmap(frac)
            vals = weights.flatten()
            ax.hist(vals, bins=60, density=True, alpha=0.2,
                    color=color, histtype="stepfilled", linewidth=1.0)

        ax.set_title(layer_labels[layer_idx], fontsize=10)
        ax.set_xlabel("Weight value", fontsize=8)
        ax.set_ylim([0, 80])
        ax.set_xlim([-0.2, 0.2])
        if layer_idx == 0:
            ax.set_ylabel("Density", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=max(n_steps - 1, 1)))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[-1], label="Checkpoint", fraction=0.046, pad=0.04)
    fig.suptitle(f"Trial {n} — Weight distribution evolution across checkpoints",
                 fontsize=11)
    fig.tight_layout()


def plot_loss(loss_list, n):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_list, color="tab:blue", linewidth=2, alpha=0.8)
    plt.text(0.5, 1.12, f"Training step: {n}", fontsize="xx-large", color="k")
    plt.yscale("log")
    plt.xlabel("Adam Pretrain Step", fontsize="large")
    plt.ylabel("Loss", fontsize="large")
    plt.title(f"Trial {n} Loss Curve", fontsize="x-large")
    plt.grid(True, which="both", ls="--", lw=0.5)
