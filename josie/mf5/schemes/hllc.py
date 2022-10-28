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

from josie.mesh.cellset import NeighboursCellSet, MeshCellSet
from josie.scheme.nonconservative import NonConservativeScheme
from josie.scheme.convective import ConvectiveScheme

from josie.math import Direction
from josie.mf5.state import Q
from josie.twofluid.state import PhasePair
from josie.twofluid.fields import Phases


from .scheme import MF5Scheme


class HLLC(ConvectiveScheme, MF5Scheme):
    def F(self, cells: MeshCellSet, neighs: NeighboursCellSet) -> Q:

        """This method implements the HLLC scheme."""

        values: Q = cells.values.view(Q)
        Q_L, Q_R = values, neighs.values.view(Q)

        FS = np.zeros(values.shape).view(Q)
        F = np.zeros(values.get_conservative().shape)
        fields = values.fields
        # We start by computing the velocity of the contact discontinuity

        # Get necessary variables :
        rho_L = Q_L[..., [fields.rho]]
        p_L = Q_L[..., [fields.P]]
        UV_L = Q_L[..., [fields.U, fields.V]]
        Un_L = np.einsum("...kl,...l->...k", UV_L, neighs.normals)[
            ..., np.newaxis
        ]
        cF_L = Q_L[..., [fields.cF]]
        rho_R = Q_R[..., [fields.rho]]
        p_R = Q_R[..., [fields.P]]
        UV_R = Q_R[..., [fields.U, fields.V]]
        Un_R = np.einsum("...kl,...l->...k", UV_R, neighs.normals)[
            ..., np.newaxis
        ]
        cF_R = Q_R[..., [fields.cF]]
        # Get the velocity of the left and right shock waves
        sigma_L, sigma_R = self.compute_sigma(Un_L, Un_R, cF_L, cF_R)
        # compute this to simplify computations
        rhoDSigU_L = rho_L * (sigma_L - Un_L)
        rhoDSigU_R = rho_R * (sigma_R - Un_R)

        # Contact discontinuity speed
        S_star = np.divide(
            p_R - p_L + Un_L * rhoDSigU_L - Un_R * rhoDSigU_R,
            rhoDSigU_L - rhoDSigU_R,
        )

        # Init the intermediate states
        Q_star_L = np.zeros(Q_L.shape)
        Q_star_R = np.zeros(Q_R.shape)

        U_star_L = UV_L + np.einsum(
            "...kl,...l->...kl", (S_star - Un_L), neighs.normals
        )
        U_star_R = UV_R + np.einsum(
            "...kl,...l->...kl", (S_star - Un_R), neighs.normals
        )

        # Get arhos
        arhos_L = PhasePair(Q_L[..., [fields.arho1]], Q_L[..., [fields.arho2]])
        arhos_R = PhasePair(Q_R[..., [fields.arho1]], Q_R[..., [fields.arho2]])

        # Compute the left intermediate state (mixture)
        Q_star_L[..., [fields.arho1]] = arhos_L[Phases.PHASE1] / rho_L
        Q_star_L[..., [fields.arho2]] = arhos_L[Phases.PHASE2] / rho_L
        Q_star_L[..., fields.rhoU] = U_star_L[..., Direction.X]
        Q_star_L[..., fields.rhoV] = U_star_L[..., Direction.Y]

        # Compute the right intermediate state (mixture)
        Q_star_R[..., [fields.arho1]] = arhos_R[Phases.PHASE1] / rho_R
        Q_star_R[..., [fields.arho2]] = arhos_R[Phases.PHASE2] / rho_R
        Q_star_R[..., fields.rhoU] = U_star_R[..., Direction.X]
        Q_star_R[..., fields.rhoV] = U_star_R[..., Direction.Y]

        # Compute the common factor
        star_coef_L = rhoDSigU_L / (sigma_L - S_star)
        star_coef_R = rhoDSigU_R / (sigma_R - S_star)

        # Apply factor to get the right flux for the partial masses and
        # momentum
        Q_star_L *= star_coef_L
        Q_star_R *= star_coef_R

        # Get volume fractions
        alphas_L = PhasePair(
            Q_L[..., [fields.alpha1]], Q_L[..., [fields.alpha2]]
        )
        alphas_R = PhasePair(
            Q_R[..., [fields.alpha1]], Q_R[..., [fields.alpha2]]
        )

        # Compute the intermediate volumic fraction
        Q_star_L[..., [fields.alpha1]] = Q_L[..., [fields.alpha1]]
        Q_star_R[..., [fields.alpha1]] = Q_R[..., [fields.alpha1]]

        # Get arhos in star regions
        arhos_star_L = PhasePair(
            Q_star_L[..., [fields.arho1]], Q_star_L[..., [fields.arho2]]
        )
        arhos_star_R = PhasePair(
            Q_star_R[..., [fields.arho1]], Q_star_R[..., [fields.arho2]]
        )

        # Get pressures
        ps_L = PhasePair(Q_L[..., [fields.p1]], Q_L[..., [fields.p2]])
        ps_R = PhasePair(Q_R[..., [fields.p1]], Q_R[..., [fields.p2]])

        # Estimate the intermediate pressure
        # Then compute the internal energy
        for phase in Phases:
            phase_values = values.get_phase(phase)
            pfields = phase_values.fields

            # Get phase densities
            rho_L = arhos_L[phase] / alphas_L[phase]
            rho_star_L = arhos_star_L[phase] / alphas_L[phase]

            rho_R = arhos_R[phase] / alphas_R[phase]
            rho_star_R = arhos_star_R[phase] / alphas_R[phase]

            # Get OES (and assume that it is a stiffened gas law)
            eos = self.problem.eos[phase]
            gamma = eos.gamma
            p0 = eos.p0

            p_L = ps_L[phase]
            p_R = ps_R[phase]

            p_star_L = (p_L + p0) * np.divide(
                (gamma + 1) * rho_star_L - (gamma - 1) * rho_L,
                (gamma + 1) * rho_L - (gamma - 1) * rho_star_L,
            ) - p0
            p_star_R = (p_R + p0) * np.divide(
                (gamma + 1) * rho_star_R - (gamma - 1) * rho_R,
                (gamma + 1) * rho_R - (gamma - 1) * rho_star_R,
            ) - p0

            arhoe_star_L = alphas_L[phase] * eos.rhoe(rho_star_L, p_star_L)
            arhoe_star_R = alphas_R[phase] * eos.rhoe(rho_star_R, p_star_R)
            # Newton algorithms to estimate pressures in star regions
            # p_star_L = copy(p_L)
            # dp = p_L
            # while np.amax(np.abs(dp) / p_star_L, axis=(0, 1)) > 1e-6:
            #    dp = self.p_star_process(p_star_L, p_L, rho_star_L,
            #  rho_L, eos)
            #    p_star_L += dp

            # p_star_R = copy(p_R)
            # dp = p_R
            # while np.amax(np.abs(dp) / p_R, axis=(0, 1)) > 1e-6:
            #    dp = self.p_star_process(p_star_R, p_R, rho_star_R,
            # rho_R, eos)
            #    p_star_R += dp

            phase_Q_star_L = Q_star_L.view(Q).get_phase(phase)
            phase_Q_star_R = Q_star_R.view(Q).get_phase(phase)

            # Compute internal energies
            phase_Q_star_L[..., [pfields.arhoe]] = arhoe_star_L
            phase_Q_star_R[..., [pfields.arhoe]] = arhoe_star_R

            Q_star_L.view(Q).set_phase(phase, phase_Q_star_L)
            Q_star_R.view(Q).set_phase(phase, phase_Q_star_R)

        # Get conservative states
        Qc_L = Q_L.get_conservative()
        Qc_R = Q_R.get_conservative()
        Qc_star_L = Q_star_L.view(Q).get_conservative()
        Qc_star_R = Q_star_R.view(Q).get_conservative()

        # All four states are now known, fluxes are then computed
        # This is the flux tensor dot the normal
        F_L = np.einsum(
            "...jkl,...l->...jk", self.problem.F(cells), neighs.normals
        )
        F_R = np.einsum(
            "...jkl,...l->...jk", self.problem.F(neighs), neighs.normals
        )
        # Right supersonic flow
        np.copyto(F, F_L, where=(sigma_L >= 0))

        # Left supersonic flow
        np.copyto(F, F_R, where=(sigma_R < 0))

        # Subsonic flow - left state
        # Conservative quantities
        np.copyto(
            F,
            F_L + sigma_L * (Qc_star_L - Qc_L),
            where=(sigma_L < 0) * (0 <= S_star),
        )
        # Non-conservative quantities
        # np.copyto(
        #    F[..., MF5ConsFields.alpha1],
        #    Q_L[..., fields.alpha1] * S_star[..., 0],
        #    where=((sigma_L < 0) * (0 <= S_star))[..., 0],
        # )
        # np.copyto(
        #    F[..., MF5ConsFields.alpha2],
        #    Q_L[..., fields.alpha2] * S_star[..., 0],
        #    where=((sigma_L < 0) * (0 <= S_star))[..., 0],
        # )

        # np.copyto(
        #    F[..., MF5ConsFields.arhoe1],
        #    Q_star_L[..., fields.arhoe1] * S_star[..., 0],
        #    where=((sigma_L < 0) * (0 <= S_star))[..., 0],
        # )
        # np.copyto(
        #    F[..., MF5ConsFields.arhoe2],
        #    Q_star_L[..., fields.arhoe2] * S_star[..., 0],
        #    where=((sigma_L < 0) * (0 <= S_star))[..., 0],
        # )

        # Subsonic flow - right state
        # Conservative quantities
        np.copyto(
            F,
            F_R + sigma_R * (Qc_star_R - Qc_R),
            where=(S_star < 0) * (0 <= sigma_R),
        )
        # Non-conservative quantities
        # np.copyto(
        #    F[..., MF5ConsFields.alpha1],
        #    Q_R[..., fields.alpha1] * S_star[..., 0],
        #    where=((S_star < 0) * (0 <= sigma_R))[..., 0],
        # )
        # np.copyto(
        #    F[..., MF5ConsFields.alpha2],
        #    Q_R[..., fields.alpha2] * S_star[..., 0],
        #    where=((S_star < 0) * (0 <= sigma_R))[..., 0],
        # )

        # np.copyto(
        #    F[..., MF5ConsFields.arhoe1],
        #    Q_star_R[..., fields.arhoe1] * S_star[..., 0],
        #    where=((S_star < 0) * (0 <= sigma_R))[..., 0],
        # )
        # np.copyto(
        #    F[..., MF5ConsFields.arhoe2],
        #    Q_star_R[..., fields.arhoe2] * S_star[..., 0],
        #    where=((S_star < 0) * (0 <= sigma_R))[..., 0],
        # )

        FS.set_conservative(neighs.surfaces[..., np.newaxis, np.newaxis] * F)

        return FS


class HLLCNonCons(NonConservativeScheme, MF5Scheme):
    r"""
    Check also :class:`~MF5.problem.MF5Problem.B`.
    """

    def G(self, cells: MeshCellSet, neighs: NeighboursCellSet) -> np.ndarray:

        values: Q = cells.values.view(Q)
        Q_L, Q_R = values, neighs.values.view(Q)

        fields = values.fields

        # Get density
        rho_L = Q_L[..., [fields.rho]]
        rho_R = Q_R[..., [fields.rho]]

        # Get volumic fraction
        alphas_L = PhasePair(
            Q_L[..., [fields.alpha1]],
            Q_L[..., [fields.alpha2]],
        )
        alphas_R = PhasePair(
            Q_R[..., [fields.alpha1]],
            Q_R[..., [fields.alpha2]],
        )

        # Get pressure
        ps_L = PhasePair(Q_L[..., [fields.p1]], Q_L[..., [fields.p2]])
        ps_R = PhasePair(Q_R[..., [fields.p1]], Q_R[..., [fields.p2]])

        p_L = sum(alphas_L[phase] * ps_L[phase] for phase in Phases)
        p_R = sum(alphas_R[phase] * ps_R[phase] for phase in Phases)

        # Get velocity
        UV_L = Q_L[..., [fields.U, fields.V]]
        UV_R = Q_R[..., [fields.U, fields.V]]

        # Compute the normal velocity components
        U_L = np.einsum("...kl,...l->...k", UV_L, neighs.normals)[
            ..., np.newaxis
        ]
        U_R = np.einsum("...kl,...l->...k", UV_R, neighs.normals)[
            ..., np.newaxis
        ]

        # # Speed of sound
        # cs_L = PhasePair(Q_L[..., [fields.c1]], Q_L[..., [fields.c2]])
        # cs_R = PhasePair(Q_R[..., [fields.c1]], Q_R[..., [fields.c2]])
        # # Let's retrieve the values of the sigma on every cell
        # sigma_L, sigma_R = self.compute_sigma(U_L, U_R, cs_L, cs_R)
        c_L = Q_L[..., [fields.cF]]
        c_R = Q_R[..., [fields.cF]]
        sigma_L, sigma_R = self.compute_sigma(U_L, U_R, c_L, c_R)

        # Compute the approximate contact discontinuity speed
        S_star = np.divide(
            p_R
            - p_L
            + rho_L * U_L * (sigma_L - U_L)
            - rho_R * U_R * (sigma_R - U_R),
            rho_L * (sigma_L - U_L) - rho_R * (sigma_R - U_R),
        )

        UV_face = np.zeros(UV_L.shape)
        UV_star_L = UV_L + np.einsum(
            "...kl,...l->...kl", (S_star - U_L), neighs.normals
        )
        UV_star_R = UV_R + np.einsum(
            "...kl,...l->...kl", (S_star - U_R), neighs.normals
        )

        # Right supersonic flow
        np.copyto(
            UV_face,
            UV_L,
            where=(sigma_L >= 0),
        )

        # Left supersonic flow
        np.copyto(
            UV_face,
            UV_R,
            where=(sigma_R < 0),
        )

        # Subsonic flow - left state
        np.copyto(
            UV_face,
            UV_star_L,
            where=(sigma_L < 0) * (0 <= S_star),
        )

        # Subsonic flow - right state
        np.copyto(
            UV_face,
            UV_star_R,
            where=(S_star < 0) * (0 <= sigma_R),
        )

        UVn_face = np.einsum("...kl,...m", UV_face, neighs.normals)

        G = neighs.surfaces[..., np.newaxis, np.newaxis, np.newaxis] * UVn_face

        return G
