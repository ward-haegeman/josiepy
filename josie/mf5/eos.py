# josiepy
# Copyright © 2020 Ruben Di Battista
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

import abc
import numpy as np

from josie.euler.eos import EOS, PerfectGas, StiffenedGas
from josie.twofluid.state import PhasePair

from typing import Union

ArrayAndScalar = Union[np.ndarray, float]


class EOSExt(EOS):
    """An Abstract Base Class representing an EOS for an Euler System
    with additional derivatives"""

    @abc.abstractmethod
    def de_dp(self, rho: ArrayAndScalar, p: ArrayAndScalar) -> ArrayAndScalar:
        raise NotImplementedError

    @abc.abstractmethod
    def de_drho(
        self, rho: ArrayAndScalar, p: ArrayAndScalar
    ) -> ArrayAndScalar:
        raise NotImplementedError

    @abc.abstractmethod
    def dp_drho(
        self, rho: ArrayAndScalar, e: ArrayAndScalar
    ) -> ArrayAndScalar:
        raise NotImplementedError

    @abc.abstractmethod
    def dp_de(self, rho: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:
        raise NotImplementedError

    @abc.abstractmethod
    def T(self, e: ArrayAndScalar, p: ArrayAndScalar) -> ArrayAndScalar:
        raise NotImplementedError

    @abc.abstractmethod
    def rhoe(self, rho: ArrayAndScalar, p: ArrayAndScalar) -> ArrayAndScalar:
        raise NotImplementedError

    @abc.abstractmethod
    def p(self, rho: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:
        raise NotImplementedError

    @abc.abstractmethod
    def rho(self, p: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:
        raise NotImplementedError

    @abc.abstractmethod
    def sound_velocity(
        self, rho: ArrayAndScalar, p: ArrayAndScalar
    ) -> ArrayAndScalar:
        raise NotImplementedError


class PerfectGasExt(PerfectGas, EOSExt):
    r"""This class embeds methods to compute states (including temperature) for
    problem with Newton-Raphson algorithms using an EOS (Equation of State) for
    perfect gases.

    .. math::

        p = \rho \mathcal{R} T = \rho \left( \gamma - 1 \right)e
        c_v = \partial_T e = R / \left( \gamma - 1 \right)


    Attributes
    ----------
    gamma
        The adiabatic coefficient
    c_v
        The specific heat capacity at constant volume (J/kg/K)
    """

    def __init__(self, gamma: float = 1.4, cv: float = 716):
        self.gamma = gamma
        self.cv = cv

    def de_dp(self, rho: ArrayAndScalar, p: ArrayAndScalar) -> ArrayAndScalar:
        r"""This returns the derivative of the internal energy with respect to
        the pressure at fixed density

        .. math::

        \partial_p e = \left(\rho \left(\gamma - 1 \right)\right)^{-1}

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells

        p
            A :class:`ArrayAndScalar` containing the values of the pressure on
            the mesh cells

        Returns
        -------
        de_dp
            A :class`ArrayAndScalar` containing the values of the derivative of
            the internal energy with respect to the pressure at fixed density
        """

        return 1 / rho / (self.gamma - 1)

    def de_drho(
        self, rho: ArrayAndScalar, p: ArrayAndScalar
    ) -> ArrayAndScalar:
        r"""This returns the derivative of the internal energy with respect to
        the density at fixed pressure

        .. math::

        \partial_\rho e = -p \left(\rho^2 \left(\gamma - 1 \right)\right)^{-1}

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells

        p
            A :class:`ArrayAndScalar` containing the values of the pressure on
            the mesh cells

        Returns
        -------
        de_drho
            A :class`ArrayAndScalar` containing the values of the derivative of
            the internal energy with respect to the density at fixed pressure
        """

        return -p / rho / rho / (self.gamma - 1)

    def dp_de(self, rho: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:
        r"""This returns the derivative of the pressure with respect to the
        internal energy at fixed density

        .. math::

        \partial_e p = \rho \left(\gamma - 1 \right)

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells

        e
            A :class:`ArrayAndScalar` containing the values of the internal
            energy on the mesh cells

        Returns
        -------
        dp_de
            A :class`ArrayAndScalar` containing the values of the derivative of
            the pressure with respect to the internal energy at fixed density
        """

        return rho * (self.gamma - 1)

    def dp_drho(
        self, rho: ArrayAndScalar, e: ArrayAndScalar
    ) -> ArrayAndScalar:
        r"""This returns the derivative of the pressure with respect to the
        density at fixed internal energy

        .. math::

        \partial_\rho p = e \left(\gamma - 1 \right)

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells

        e
            A :class:`ArrayAndScalar` containing the values of the internal
            energy on the mesh cells

        Returns
        -------
        dp_drho
            A :class`ArrayAndScalar` containing the values of the derivative of
            the pressure with respect to the density at fixed internal energy
        """

        return e * (self.gamma - 1)

    def T(self, e: ArrayAndScalar, p: ArrayAndScalar) -> ArrayAndScalar:
        r"""This returns the temperature

        .. math::

        T = e / c_v

        Parameters
        ----------
        p
            A :class:`ArrayAndScalar` containing the values of the pressure on
            the mesh cells

        e
            A :class:`ArrayAndScalar` containing the values of the internal
            energy on the mesh cells

        Returns
        -------
        T
            A :class`ArrayAndScalar` containing the values of the temperature
        """

        return e / self.cv


class StiffenedGasExt(StiffenedGas, EOSExt):
    def __init__(self, gamma: float = 3, p0: float = 1e9, cv: float = 4000):
        self.gamma = gamma
        self.p0 = p0
        self.cv = cv

    def de_dp(self, rho: ArrayAndScalar, p: ArrayAndScalar) -> ArrayAndScalar:
        return 1 / rho / (self.gamma - 1)

    def de_drho(
        self, rho: ArrayAndScalar, p: ArrayAndScalar
    ) -> ArrayAndScalar:
        return -(p + self.gamma * self.p0) / rho / rho / (self.gamma - 1)

    def dp_de(self, rho: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:
        return (self.gamma - 1) * rho

    def dp_drho(
        self, rho: ArrayAndScalar, e: ArrayAndScalar
    ) -> ArrayAndScalar:
        return (self.gamma - 1) * e

    def T(self, e: ArrayAndScalar, p: ArrayAndScalar) -> ArrayAndScalar:
        return np.divide(
            e, self.cv * (p + self.gamma * self.p0) / (p + self.p0)
        )

    def rhoe(self, rho: ArrayAndScalar, p: ArrayAndScalar) -> ArrayAndScalar:

        return (p + self.gamma * self.p0) / (self.gamma - 1)

    def p(self, rho: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:

        return (self.gamma - 1) * np.multiply(rho, e) - self.p0 * self.gamma

    def rho(self, p: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:
        return (p + self.p0 * self.gamma) / (self.gamma - 1) / e

    def sound_velocity(
        self, rho: ArrayAndScalar, p: ArrayAndScalar
    ) -> ArrayAndScalar:
        return np.sqrt(self.gamma * np.divide((p + self.p0), rho))


class TwoPhaseEOS(PhasePair):
    """An Abstract Base Class representing en EOS for a twophase system.  In
    particular two :class:`.euler.eos.EOS` instances for each phase need to be
    provided.

    You can access the EOS for a specified phase using the
    :meth:`__getitem__`

    """

    def __init__(self, phase1: EOSExt, phase2: EOSExt):
        """
        Parameters
        ----------
        phase1
            An instance of :class:`.euler.eos.EOS` representing the EOS for the
            single phase #1
        phase2
            An instance of :class:`.euler.eos.EOS` representing the EOS for the
            single phase #2
        """

        super().__init__(phase1, phase2)
