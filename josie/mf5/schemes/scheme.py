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
from typing import Tuple

from josie.mesh.cellset import MeshCellSet
from josie.euler.schemes.rusanov import Rusanov as EulerRusanov
from josie.scheme.convective import ConvectiveScheme
from josie.scheme.nonconservative import NonConservativeScheme

from josie.mf5.eos import TwoPhaseEOS
from josie.mf5.eos import EOSExt as EOS
from josie.mf5.problem import MF5Problem
from josie.mf5.state import Q, MF5PhaseFields
from josie.twofluid.state import PhasePair
from josie.twofluid.fields import Phases


class MF5Scheme(ConvectiveScheme, NonConservativeScheme):
    """A base class for a twophase scheme MF5"""

    problem: MF5Problem
    isSG: bool

    def __init__(self, eos: TwoPhaseEOS, isSG: bool):
        super().__init__(MF5Problem(eos))
        self.isSG = isSG

    def CFL(
        self,
        cells: MeshCellSet,
        CFL_value,
    ) -> float:

        dt = 1e9
        dx = np.min(cells.volumes[..., np.newaxis] / cells.surfaces)
        for phase in Phases:
            phase_values = cells.values.view(Q).get_phase(phase)
            fields = phase_values.fields

            # Get the velocity components
            UV_slice = slice(Q.fields.U, Q.fields.V)
            UV = phase_values[..., UV_slice]

            U = np.linalg.norm(UV, axis=-1)
            c = phase_values[..., fields.c]

            sigma = np.max(EulerRusanov.compute_sigma(U, U, c, c))

            dt = np.min((dt, CFL_value * dx / sigma))

        return dt

    @staticmethod
    def compute_sigma(
        U_L: np.ndarray, U_R: np.ndarray, cF_L: np.ndarray, cF_R: np.ndarray
    ) -> Tuple:
        r"""Returns the value of the :math:`\sigma_L` and :math:`\sigma_R`
        (i.e. the wave velocity "Davis" estimates) for a two-phase scheme.

        .. math::

            \sigma = \max_{L, R}{\qty(\norm{\vb{u}} + c, \norm{\vb{u}} - c)}


        Parameters
        ----------
        U_L
            The value of scalar velocity for each cell. Array dimensions
            :math:`N_x \times N_y \times 1`

        U_R
            The value of scalar velocity for each cell neighbour. Array
            dimensions :math:`N_x \times N_y \times 1`

        cs_L
            The values of sound velocity for each cell and phase

        cs_R
            The value of sound velocity for each cell neighbour and phase

        Returns
        -------
        sigma_L
            A :math:`Nx \times Ny \times 1` containing the value of
            the left sigma per each cell
        sigma_R
            A :math:`Nx \times Ny \times 1` containing the value of
            the right sigma per each cell
        """

        sigma_L = np.concatenate(
            (
                U_L - cF_L,
                U_R - cF_R,
            ),
            axis=-1,
        ).min(axis=-1, keepdims=True)

        sigma_R = np.concatenate(
            (
                U_L + cF_L,
                U_R + cF_R,
            ),
            axis=-1,
        ).max(axis=-1, keepdims=True)

        return sigma_L, sigma_R

    def SG_relax_process(self, values: Q):
        # For the case of two stiffened gas (SG) laws, the instantenous
        # relaxation
        # process can be computed exactly
        # P_star will be the postive root of a polynomial X**2 + b*X + c
        # and the densities also have an exact expression
        fields = Q.fields
        eosPair = PhasePair(
            self.problem.eos[Phases.PHASE1], self.problem.eos[Phases.PHASE2]
        )

        alpha1 = values[..., fields.alpha1]
        p1 = values[..., fields.p1]
        gamma1 = eosPair[Phases.PHASE1].gamma
        p01 = eosPair[Phases.PHASE1].p0

        alpha2 = 1 - alpha1
        p2 = values[..., fields.p2]
        gamma2 = eosPair[Phases.PHASE2].gamma
        p02 = eosPair[Phases.PHASE2].p0

        denominator = alpha1 * gamma2 + alpha2 * gamma1
        numerator_b = alpha1 * gamma2 * (p02 - p1) + alpha2 * gamma1 * (
            p01 - p2
        )
        numerator_c = -p1 * alpha1 * gamma2 * p02 - p2 * alpha2 * gamma1 * p01

        b = np.divide(numerator_b, denominator)
        c = np.divide(numerator_c, denominator)

        p_star = -0.5 * b + np.sqrt(0.25 * b * b - c)
        # next compute the densities
        aF2_star = 0
        for phase in Phases:
            pvalues = values.view(Q).get_phase(phase)
            pfields = pvalues.fields
            gamma = eosPair[phase].gamma
            p0 = eosPair[phase].p0
            rho = pvalues[..., pfields.arho] / pvalues[..., pfields.alpha]
            p = pvalues[..., pfields.p]
            rho_star = rho * np.divide(
                gamma * (p_star + p0), p + gamma * p0 + p_star * (gamma - 1)
            )
            rhoe_star = eosPair[phase].rhoe(rho_star, p_star)
            T_star = eosPair[phase].T(rho_star / rho_star, p_star)
            c_star = eosPair[phase].sound_velocity(rho_star, p_star)
            alpha_star = pvalues[..., pfields.arho] / rho_star
            pvalues[..., pfields.alpha] = alpha_star
            pvalues[..., pfields.arhoe] = alpha_star * rhoe_star
            pvalues[..., pfields.p] = p_star
            pvalues[..., pfields.T] = T_star
            pvalues[..., pfields.c] = c_star

            values.view(Q).set_phase(phase, pvalues)
            aF2_star += pvalues[..., pfields.arho] * c_star**2

        values[..., fields.P] = p_star
        values[..., fields.cF] = np.sqrt(
            np.divide(aF2_star, values[..., fields.rho])
        )

    def relax_process(self, values: Q, tol=1e-6, MaxIter=100, local=False):
        fields = Q.fields

        eosPair = PhasePair(
            self.problem.eos[Phases.PHASE1], self.problem.eos[Phases.PHASE2]
        )

        # Get fields over which Newton iterates
        alpha1 = values[..., fields.alpha1]
        alpha2 = 1 - alpha1
        e1 = values[..., fields.arhoe1] / values[..., fields.arho1]
        e2 = values[..., fields.arhoe2] / values[..., fields.arho2]

        # Get values necessary to compute increments
        arho1 = values[..., fields.arho1]
        rho1 = arho1 / alpha1
        arho2 = values[..., fields.arho2]
        rho2 = arho2 / alpha2
        p1 = values[..., fields.p1]
        p2 = values[..., fields.p2]
        P = values[..., fields.P]
        # print(
        #    "Entering relaxation process, nanmin p1 = ",
        #    np.nanmin(p1),
        # )
        N_iter = 0
        while (
            np.nanmax(np.abs(p1 - p2) / p1, axis=(0, 1)) > tol
            and N_iter < MaxIter
        ):
            N_iter += 1

            # Get EOS derivatives
            dp_drho_1 = eosPair[Phases.PHASE1].dp_drho(rho1, e1)
            dp_drho_2 = eosPair[Phases.PHASE2].dp_drho(rho2, e2)

            dp_de_1 = eosPair[Phases.PHASE1].dp_de(rho1, e1)
            dp_de_2 = eosPair[Phases.PHASE2].dp_de(rho2, e2)

            # Compute variations
            de_da_1 = -P / (alpha1 * rho1)
            de_da_2 = -P / (alpha2 * rho2)

            drho_da_1 = -rho1 / alpha1
            drho_da_2 = -rho2 / alpha2

            dp_da_1 = dp_drho_1 * drho_da_1 + dp_de_1 * de_da_1
            dp_da_2 = dp_drho_2 * drho_da_2 + dp_de_2 * de_da_2

            # Compute increments
            dalpha1 = (p2 - p1) / (dp_da_1 + dp_da_2)
            dalpha2 = -dalpha1

            de1 = de_da_1 * dalpha1
            de2 = de_da_2 * dalpha2

            # Update quantities
            alpha1 += dalpha1
            alpha2 = 1.0 - alpha1
            e1 += de1
            e2 += de2

            # Update quantities necessary for the variations (arho_k stays fixed)
            rho1 = arho1 / alpha1
            rho2 = arho2 / alpha2
            p1 = eosPair[Phases.PHASE1].p(rho1, e1)
            p2 = eosPair[Phases.PHASE2].p(rho2, e2)
            P = alpha1 * p1 + alpha2 * p2

        # After iteration process, update state vector
        values[..., fields.alpha1] = alpha1
        values[..., fields.alpha2] = alpha2
        values[..., fields.arhoe1] = values[..., fields.arho1] * e1
        values[..., fields.arhoe2] = values[..., fields.arho2] * e2
        values[..., fields.p1] = p1
        values[..., fields.p2] = p2
        values[..., fields.P] = P
        # Also update auxiliary fields
        T1 = eosPair[Phases.PHASE1].T(e1, rho1)
        T2 = eosPair[Phases.PHASE2].T(e2, rho2)
        c1 = eosPair[Phases.PHASE1].sound_velocity(rho1, p1)
        c2 = eosPair[Phases.PHASE2].sound_velocity(rho2, p2)
        values[..., fields.T1] = T1
        values[..., fields.T2] = T2
        values[..., fields.c1] = c1
        values[..., fields.c2] = c2
        values[..., fields.cF] = np.sqrt(
            np.divide(arho1 * c1**2 + arho2 * c2**2, arho1 + arho2)
        )
        print(
            "Ended relaxation with nanmax |p1-p2|/p1 = ",
            np.nanmax(
                np.abs(values[..., fields.p1] - values[..., fields.p2])
                / values[..., fields.p1]
            ),
            f" after {N_iter} iterations.",
        )

    def SG_pressure_correction_process(self, values: Q):
        fields = Q.fields

        eosPair = PhasePair(
            self.problem.eos[Phases.PHASE1], self.problem.eos[Phases.PHASE2]
        )
        # Get fields required to compute the new pressure
        rhoE = values[..., fields.rhoE]
        rhoU = values[..., fields.rhoU]
        U = values[..., fields.U]
        alpha1 = values[..., fields.alpha1]
        gamma1 = eosPair[Phases.PHASE1].gamma
        Pinf1 = eosPair[Phases.PHASE1].p0
        alpha2 = values[..., fields.alpha2]
        gamma2 = eosPair[Phases.PHASE2].gamma
        Pinf2 = eosPair[Phases.PHASE2].p0

        num = (rhoE - 0.5 * rhoU * U) - (
            alpha1 * gamma1 * Pinf1 / (gamma1 - 1)
            + alpha2 * gamma2 * Pinf2 / (gamma2 - 1)
        )
        den = alpha1 / (gamma1 - 1) + alpha2 / (gamma2 - 1)
        p_corr = np.divide(num, den)
        # Update all thermodynamic variables
        aF2 = 0
        for phase in Phases:
            phase_values = values.view(Q).get_phase(phase)
            pfields = MF5PhaseFields
            rho_k = (
                phase_values[..., pfields.arho]
                / phase_values[..., pfields.alpha]
            )
            e_k = eosPair[phase].rhoe(rho_k, p_corr) / rho_k
            T_k = eosPair[phase].T(e_k, p_corr)
            c_k = eosPair[phase].sound_velocity(rho_k, p_corr)
            aF2 += phase_values[..., pfields.arho] * c_k**2
            phase_values[..., pfields.arhoe] = (
                phase_values[..., pfields.arho] * e_k
            )
            phase_values[..., pfields.p] = p_corr
            phase_values[..., pfields.T] = T_k
            phase_values[..., pfields.c] = c_k
            values.view(Q).set_phase(phase, phase_values)

        values[..., fields.cF] = np.sqrt(aF2 / values[..., fields.rho])
        values[..., fields.P] = p_corr

    @staticmethod
    def corr_process(p, rhoe_0, arhos, rhos, eosPair) -> np.ndarray:
        """
        Pressure correction after the relaxation process.
        Computes a new pressure based on the conservation of the total energy.
        """
        denum = np.zeros(p.shape)
        dp = rhoe_0

        for phase in Phases:
            dp = (
                dp
                - arhos[phase]
                * eosPair[phase].rhoe(rhos[phase], p)
                / rhos[phase]
            )
            denum += arhos[phase] * eosPair[phase].de_dp(rhos[phase], p)

        return dp / denum

    @staticmethod
    def p_star_process(p_star, p, rho_star, rho, eos: EOS) -> np.ndarray:
        """
        Allows to compute the pressure in the star regions of the HLLC

        HLLC : 4 constant states q_L, q_L*, q_R*, q_R
        """
        dp = (
            rho * eos.rhoe(rho_star, p_star)
            - rho_star * eos.rhoe(rho, p)
            + 0.5 * (p_star + p) * (rho - rho_star)
        )

        dp /= rho * rho_star * eos.de_dp(rho_star, p_star) + 0.5 * (
            rho - rho_star
        )

        return -dp

    def update_auxiliary_variables(self, values: Q):
        """
        The conservative values of Q are assumed to have been updated
        Here, all auxiliary variables are recomputed (updated)
        using the conservative variables
        """

        fields = Q.fields

        # Pair the EOS
        eosPair = PhasePair(
            self.problem.eos[Phases.PHASE1], self.problem.eos[Phases.PHASE2]
        )
        # Get the conservatives variables
        alpha1 = values[..., fields.alpha1]
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]
        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        arhoe1 = values[..., fields.arhoe1]
        arhoe2 = values[..., fields.arhoe2]

        # Compute auxiliary variables
        rho = arho1 + arho2
        rhoe = arhoe1 + arhoe2
        U = rhoU / rho
        V = rhoV / rho

        alphas = PhasePair(alpha1, 1 - alpha1)
        rhos = PhasePair(arho1 / alpha1, arho2 / (1 - alpha1))
        es = PhasePair(arhoe1 / arho1, arhoe2 / arho2)

        values[..., fields.rho] = rho
        values[..., fields.U] = U
        values[..., fields.V] = V
        values[..., fields.rhoE] = 0.5 * (rhoU * U + rhoV * V) + rhoe
        values[..., fields.rhoe] = rhoe
        values[..., fields.alpha2] = 1.0 - alpha1

        aF2 = 0
        P = 0
        for phase in Phases:
            phase_values = values.view(Q).get_phase(phase)
            pfields = MF5PhaseFields

            alpha_k = alphas[phase]
            rho_k = rhos[phase]
            e_k = es[phase]
            p_k = eosPair[phase].p(rho_k, e_k)
            c_k = eosPair[phase].sound_velocity(rho_k, p_k)

            aF2 += alpha_k * rho_k * c_k**2
            P += alpha_k * p_k

            T_k = eosPair[phase].T(e_k, p_k)

            phase_values[..., pfields.p] = p_k
            phase_values[..., pfields.T] = T_k
            phase_values[..., pfields.c] = c_k

            values.view(Q).set_phase(phase, phase_values)

        values[..., fields.cF] = np.sqrt(aF2 / rho)
        values[..., fields.P] = P

    def post_step(self, values: Q):
        """During the step we update the conservative values. After the
        step we update the non-conservative variables. This method updates
        the values of the non-conservative (auxiliary) variables using the
        :class:`~.EOS`
        """

        self.update_auxiliary_variables(values)
        self.relax_process(values, tol=1e-10, MaxIter=50, local=False)
        # self.SG_pressure_correction_process(values)
        # self.SG_relax_process(values)