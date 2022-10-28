# josiepy
# Copyright © 2019 Ruben Di Battista
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY Ruben Di Battista ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Ruben Di Battista BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation
# are those of the authors and should not be interpreted as representing
# official policies, either expressed or implied, of Ruben Di Battista.
import numpy as np

from typing import Union

from josie.dimension import MAX_DIMENSIONALITY
from josie.problem import Problem
from josie.math import Direction
from josie.mesh.cellset import CellSet, MeshCellSet

from .eos import TwoPhaseEOS
from .state import Q, MF5ConsFields, MF5GradFields


class MF5Problem(Problem):
    """A class representing an Euler system problem

    Attributes
    ---------
    eos
        An instance of :class:`~.EOS`, an equation of state for the fluid
    """

    def __init__(self, eos: TwoPhaseEOS):
        self.eos = eos

    def F(self, values: Q) -> np.ndarray:
        r""" This returns the tensor representing the flux for a two-phase model

        A general problem can be written in a compact way:

        .. math::

            \pdv{\vb{q}}{t} + \div{\vb{F\qty(\vb{q})}} + \vb{B}\qty(\vb{q})
            \cdot \gradient{\vb{q}} = \vb{s\qty(\vb{q})}

        This function needs to return :math:`\vb{F}\qty(\vb{q})`

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` that contains the cell data

        Returns
        ---------
        F
            An array of dimension :math:`Nx \times Ny \times 13 \times 2`, i.e.
            an array that of each :math:`x` cell and :math:`y` cell stores the
            :math:`13 \times 2` flux tensor

            The flux tensor is:

            .. math::

                \pdeConvective =
                \begin{bmatrix}
                    \rho u & \rho v \\
                    \rho u^2 + p & \rho uv \\
                    \rho vu & \rho v^ 2 + p \\
                    (\rho E + p)U & (\rho E + p)V \\
                    \omega U & \omega V \\
                    \rho \tilde{\Sigma} U & \rho \tilde{\Sigma} V \\
                    \rho H U & \rho H V \\
                    \rho y_1 U & \rho y_1 V \\
                    \alpha_1 U & \alpha_1 V \\
                    \alpha_1\rho_1 e_1 U & \alpha_1\rho_1 e_1 V\\
                    \rho y_2 U & \rho y_2 V \\
                    \alpha_2 U & \alpha_2 V \\
                    \alpha_2\rho_2 e_2 U & \alpha_2\rho_2 e_2 V\\
                \end{bmatrix}
        """

        num_cells_x, num_cells_y, num_dofs, _ = values.shape

        # Flux tensor
        F = np.zeros(
            (
                num_cells_x,
                num_cells_y,
                num_dofs,
                len(MF5ConsFields),
                MAX_DIMENSIONALITY,
            )
        )
        fields = Q.fields

        alpha1 = values[..., fields.alpha1]
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]
        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        arhoe1 = values[..., fields.arhoe1]
        arhoe2 = values[..., fields.arhoe2]

        U = values[..., fields.U]
        V = values[..., fields.V]
        P = values[..., fields.P]

        rhoUU = np.multiply(rhoU, U)
        rhoUV = np.multiply(rhoU, V)
        rhoVV = np.multiply(rhoV, V)
        rhoVU = np.multiply(rhoV, U)

        # X direction
        F[..., MF5ConsFields.alpha1, Direction.X] = np.multiply(alpha1, U)

        F[..., MF5ConsFields.rhoU, Direction.X] = rhoUU + P
        F[..., MF5ConsFields.rhoV, Direction.X] = rhoVU

        F[..., MF5ConsFields.arho1, Direction.X] = np.multiply(arho1, U)
        F[..., MF5ConsFields.arhoe1, Direction.X] = np.multiply(arhoe1, U)

        F[..., MF5ConsFields.arho2, Direction.X] = np.multiply(arho2, U)
        F[..., MF5ConsFields.arhoe2, Direction.X] = np.multiply(arhoe2, U)

        # Y direction
        F[..., MF5ConsFields.alpha1, Direction.Y] = np.multiply(alpha1, V)

        F[..., MF5ConsFields.rhoU, Direction.Y] = rhoUV
        F[..., MF5ConsFields.rhoV, Direction.Y] = rhoVV + P

        F[..., MF5ConsFields.arho1, Direction.Y] = np.multiply(arho1, V)
        F[..., MF5ConsFields.arhoe1, Direction.Y] = np.multiply(arhoe1, V)

        F[..., MF5ConsFields.arho2, Direction.Y] = np.multiply(arho2, V)
        F[..., MF5ConsFields.arhoe2, Direction.Y] = np.multiply(arhoe2, V)

        return F

    def B(self, values: Q):
        r""" This returns the tensor that pre-multiplies the non-conservative
        term of the problem.

        A general problem can be written in a compact way:

        .. math::

            \pdeFull


        This method needs in general to return
        :math:`\pdeNonConservativeMultiplier` but since most of the
        :math:`\pdeNonConservativeMultiplier` is zero,
        since we just have the terms that pre-multiply
        :math:`\gradient\cdot \bold{u}` we just return :math:`B_{p1r} =
        \tilde{B}_{pr} = \tilde{\vb{B}}\qty(\pdeState)` that is:

        .. math::

            \tilde{\vb{B}}\qty(\pdeState) =
            \begin{bmatrix}
            u_I & v_I \\
            0 & 0 \\
            -p_I & 0 \\
            0 & -p_I \\
            -p_I u_I & -p_I v_I \\
            0 & 0 \\
            p_I & 0 \\
            0 & p_I \\
            p_I u_I & p_I v_I \\
            \end{bmatrix}

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` that contains the cell data
        """

        num_cells_x, num_cells_y, num_dofs, num_fields = values.shape
        fields = Q.fields

        B = np.zeros(
            (
                num_cells_x,
                num_cells_y,
                num_dofs,
                len(fields),  # len(MF5ConsFields),
                len(MF5GradFields),
                MAX_DIMENSIONALITY,
            )
        )

        alpha1 = values[..., fields.alpha1]
        alpha2 = 1 - alpha1
        p1 = values[..., fields.p1]
        p2 = values[..., fields.p2]

        # Terms applied to \alpha_1
        B[..., fields.alpha1, MF5GradFields.U, Direction.X] = -alpha1
        B[..., fields.alpha1, MF5GradFields.V, Direction.Y] = -alpha1

        # Terms applied to \alpha_1*\rho_1*e_1
        B[..., fields.arhoe1, MF5GradFields.U, Direction.X] = -alpha1 * p1
        B[..., fields.arhoe1, MF5GradFields.V, Direction.Y] = -alpha1 * p1

        # Terms applied to \alpha_2*\rho_2*e_2
        B[..., fields.arhoe2, MF5GradFields.U, Direction.X] = -alpha2 * p2
        B[..., fields.arhoe2, MF5GradFields.V, Direction.Y] = -alpha2 * p2

        return B
