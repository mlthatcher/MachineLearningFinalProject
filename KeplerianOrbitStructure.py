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