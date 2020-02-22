import numpy as np
import pytest

from josie.bc import Dirichlet, Neumann, Direction, make_periodic
from josie.geom import Line
from josie.mesh import Mesh, SimpleCell
from josie.solver.euler import Rusanov, Q, EulerSolver, PerfectGas

riemann_states = [
    {
        "rhoL": 1.0,
        "uL": 0.0,
        "vL": 0,
        "pL": 1.0,
        "rhoR": 0.125,
        "uR": 0,
        "vR": 0,
        "pR": 0.1,
        "CFL": 0.9,
    },
    {
        "rhoL": 1.0,
        "uL": -2,
        "vL": 0,
        "pL": 0.4,
        "rhoR": 1.0,
        "uR": 2.0,
        "vR": 0,
        "pR": 0.4,
        "CFL": 0.9,
    },
    {
        "rhoL": 1.0,
        "uL": 0,
        "vL": 0,
        "pL": 1000,
        "rhoR": 1.0,
        "uR": 0,
        "vR": 0,
        "pR": 0.01,
        "CFL": 0.45,
    },
    {
        "rhoL": 5.99924,
        "uL": 19.5975,
        "vL": 0,
        "pL": 460.894,
        "rhoR": 5.9924,
        "uR": -6.19633,
        "vR": 0,
        "pR": 46.0950,
        "CFL": 0.5,
    },
    {
        "rhoL": 1.0,
        "uL": -19.59745,
        "vL": 0,
        "pL": 1000,
        "rhoR": 1.0,
        "uR": -19.59745,
        "vR": 0,
        "pR": 0.01,
        "CFL": 0.5,
    },
]


def neumann(first, second, direction):
    second.bc = Neumann(Q.zeros())
    first.bc = Neumann(Q.zeros())

    return first, second


def periodic(first, second, direction):
    return make_periodic(first, second, direction)


@pytest.mark.parametrize("riemann_problem", riemann_states)
@pytest.mark.parametrize("bc_fun", [periodic, neumann])
def test_toro_x(riemann_problem, bc_fun, plot):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = PerfectGas(gamma=1.4)

    # BC
    rhoL = riemann_problem["rhoL"]
    uL = riemann_problem["uL"]
    vL = riemann_problem["vL"]
    pL = riemann_problem["pL"]
    rhoeL = eos.rhoe(rhoL, pL)
    EL = rhoeL / rhoL + 0.5 * (uL ** 2 + vL ** 2)
    cL = eos.sound_velocity(rhoL, pL)

    rhoR = riemann_problem["rhoR"]
    uR = riemann_problem["uR"]
    vR = riemann_problem["vR"]
    pR = riemann_problem["pR"]
    rhoeR = eos.rhoe(rhoR, pR)
    ER = rhoeR / rhoR + 0.5 * (uR ** 2 + vR ** 2)
    cR = eos.sound_velocity(rhoR, pR)

    Q_left = Q(rhoL, rhoL * uL, rhoL * vL, rhoL * EL, rhoeL, uL, vL, pL, cL)
    Q_right = Q(rhoR, rhoR * uR, rhoR * vR, rhoR * ER, rhoeR, uR, vR, pR, cR)

    left.bc = Dirichlet(Q_left)
    right.bc = Dirichlet(Q_right)
    bottom, top = bc_fun(bottom, top, Direction.Y)

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(30, 30)
    mesh.generate()

    def init_fun(solver: EulerSolver):
        xc = solver.mesh.centroids[:, :, 0]

        idx_left = np.where(xc >= 0.5)
        idx_right = np.where(xc < 0.5)

        solver.values[idx_left[0], idx_left[1], :] = Q_right
        solver.values[idx_right[0], idx_right[1], :] = Q_left

    solver = EulerSolver(mesh, eos)
    solver.init(init_fun)

    final_time = 0.25
    t = 0
    CFL = riemann_problem["CFL"]
    rusanov = Rusanov()

    while t <= final_time:
        if plot:
            solver.animate(t)
            # solver.save(t, "toro.xmf")

        dt = rusanov.CFL(
            solver.values,
            solver.mesh.volumes,
            solver.mesh.normals,
            solver.mesh.surfaces,
            CFL,
        )
        assert ~np.isnan(dt)
        solver.step(dt, rusanov)

        t += dt
        print(f"Time: {t}, dt: {dt}")

    if plot:
        solver.show("U")


@pytest.mark.parametrize("riemann_problem", riemann_states)
@pytest.mark.parametrize("bc_fun", [periodic, neumann])
def test_toro_y(riemann_problem, bc_fun, plot):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = PerfectGas(gamma=1.4)

    # BC
    rhoL = riemann_problem["rhoL"]
    uL = riemann_problem["vL"]
    vL = riemann_problem["uL"]
    pL = riemann_problem["pL"]
    rhoeL = eos.rhoe(rhoL, pL)
    EL = rhoeL / rhoL + 0.5 * (uL ** 2 + vL ** 2)
    cL = eos.sound_velocity(rhoL, pL)

    rhoR = riemann_problem["rhoR"]
    uR = riemann_problem["vR"]
    vR = riemann_problem["uR"]
    pR = riemann_problem["pR"]
    rhoeR = eos.rhoe(rhoR, pR)
    ER = rhoeR / rhoR + 0.5 * (uR ** 2 + vR ** 2)
    cR = eos.sound_velocity(rhoR, pR)

    Q_left = Q(rhoL, rhoL * uL, rhoL * vL, rhoL * EL, rhoeL, uL, vL, pL, cL)
    Q_right = Q(rhoR, rhoR * uR, rhoR * vR, rhoR * ER, rhoeR, uR, vR, pR, cR)

    bottom.bc = Dirichlet(Q_left)
    top.bc = Dirichlet(Q_right)
    left, right = bc_fun(left, right, Direction.X)

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(30, 30)
    mesh.generate()

    def init_fun(solver: EulerSolver):
        yc = solver.mesh.centroids[:, :, 1]

        idx_top = np.where(yc >= 0.5)
        idx_btm = np.where(yc < 0.5)

        solver.values[idx_btm[0], idx_btm[1], :] = Q_left
        solver.values[idx_top[0], idx_top[1], :] = Q_right

    solver = EulerSolver(mesh, eos)
    solver.init(init_fun)
    solver.plot()

    final_time = 0.25
    t = 0
    CFL = riemann_problem["CFL"]
    rusanov = Rusanov()

    while t <= final_time:
        if plot:
            solver.animate(t)
            solver.save(t, "toro.xmf")

        dt = rusanov.CFL(
            solver.values,
            solver.mesh.volumes,
            solver.mesh.normals,
            solver.mesh.surfaces,
            CFL,
        )
        assert ~np.isnan(dt)
        solver.step(dt, rusanov)

        t += dt
        print(f"Time: {t}, dt: {dt}")

    if plot:
        solver.show("V")