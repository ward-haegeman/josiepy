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
from __future__ import annotations

from josie.twofluid.fields import Phases
from josie.twofluid.state import TwoFluidState, PhaseState

from josie.state import Fields
from josie.state import SubsetState


class MF5Fields(Fields):
    """Indexing enum for the state variables of the problem"""

    alpha1 = 0
    alpha2 = 1

    rho = 2
    rhoU = 3
    rhoV = 4
    rhoE = 5
    rhoe = 6
    U = 7
    V = 8
    cF = 9
    P = 10

    arho1 = 11
    arhoe1 = 12
    p1 = 13
    T1 = 14
    c1 = 15

    arho2 = 16
    arhoe2 = 17
    p2 = 18
    T2 = 19
    c2 = 20


class MF5ConsFields(Fields):
    """Indexing enum for the conservative state variables of the problem"""

    alpha1 = 0
    rhoU = 1
    rhoV = 2

    arho1 = 3
    arhoe1 = 4

    arho2 = 5
    arhoe2 = 6


class MF5PhaseFields(Fields):
    """Indexing fields for a substate associated to a phase"""

    alpha = 0
    arho = 1
    arhoe = 2
    p = 3
    T = 4
    c = 5


class MF5PhaseState(PhaseState):
    """State array for one single phase"""

    fields = MF5PhaseFields
    full_state_fields = MF5Fields


class MF5ConsState(SubsetState):
    """State array for conservative part of the state of one single phase"""

    fields = MF5ConsFields
    full_state_fields = MF5Fields


class MF5GradFields(Fields):
    r"""Indexes used to index the gradient pre-factor
    :math:`\pdeNonConservativeMultiplier`. Check :mod:`twophase.problem` for
    more information on how the multiplier is reduced in size to optimize
    the compuation"""

    U = 0
    V = 1


class Q(TwoFluidState):
    r"""We create one big state that contains the actual conservative
    variables that are used in the flux together with the "auxiliary" variables
    that are instead needed, for example, to compute the speed of sound.

    The state of system described in :cite:`baer_two-phase_1986` is actually
    two Euler states together with the state associated to the volume fraction
    :math:`\alpha`"""

    fields = MF5Fields
    cons_state = MF5ConsState
    phase_state = MF5PhaseState

    def get_conservative(self) -> MF5ConsState:
        return super().get_conservative().view(MF5ConsState)

    def get_phase(self, phase: Phases) -> MF5PhaseState:
        return super().get_phase(phase).view(MF5PhaseState)
