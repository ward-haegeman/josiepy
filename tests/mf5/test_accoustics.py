import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec

from josie.mf5.eos import StiffenedGasExt as StiffenedGas
from josie.bc import make_periodic, Direction
from josie.general.schemes.time import RK2, ExplicitEuler
from josie.general.schemes.space.limiters import No_Limiter
from josie.general.schemes.space.muscl import MUSCL
from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet
from josie.mf5.eos import TwoPhaseEOS
from josie.mf5.schemes.hllc import HLLC, HLLCNonCons
from josie.mf5.solver import TwoPhaseSolver
from josie.mf5.state import Q
from josie.twofluid.state import PhasePair
from josie.twofluid.fields import Phases

# Time scheme :
time_scheme = ExplicitEuler
# Spatial discretization
#space_scheme = MUSCL_Hancock_no_limiter


class MF5Scheme(HLLC, HLLCNonCons, MUSCL, No_Limiter, time_scheme):
    pass


@dataclass
class Problem:
    alpha1: float
    rho1: float
    rho2: float
    Pref: float
    scheme: MF5Scheme
    xmin: float
    xmax: float
    final_time: float
    CFL: float


accousticProblem = Problem(
    alpha1=0.12,
    rho1=1,
    rho2=1000,
    Pref=1e5,
    scheme=MF5Scheme(
        eos=TwoPhaseEOS(
            phase1=StiffenedGas(gamma=1.401015758, p0=0, cv=715.809222),
            phase2=StiffenedGas(gamma=1.1, p0=1.238e8, cv=4130),
        ),
        isSG=True,
    ),
    xmin=0.0,
    xmax=1.0,
    final_time=1.0e-2,
    CFL=0.9,
)


def plot_func(data, time_annotation, lines, axes, fields_to_plot):
    t = data[0]
    time_annotation.set_text(f"t={t:.6f}s")
    x = data[1]
    values = data[2]

    # mixture Pressure
    P = values[..., Q.fields.P]
    line = lines[0]
    line.set_data(x, P)
    ax = axes[0]
    ax.relim()
    ax.autoscale_view()

    for i, field in enumerate(fields_to_plot, 1):
        line = lines[i]
        line.set_data(x, values[..., field])
        ax = axes[i]
        ax.relim()
        ax.autoscale_view()


def test_accoustics(plot, write):
    prob = accousticProblem
    left = Line([prob.xmin, 0], [prob.xmin, 1])
    bottom = Line([prob.xmin, 0], [prob.xmax, 0])
    right = Line([prob.xmax, 0], [prob.xmax, 1])
    top = Line([prob.xmin, 1], [prob.xmax, 1])

    left, right = make_periodic(left, right, Direction.X)
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(200, 1)
    mesh.generate()

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0, 0]
        fields = Q.fields

        state_array: Q = np.zeros_like(cells.values).view(Q)

        U = 0
        V = 0

        alphas = PhasePair(prob.alpha1, 1 - prob.alpha1)
        rhos = PhasePair(prob.rho1, prob.rho2)
        aF = 0

        p = (
            prob.Pref
            + 1
            * np.sinc(4 * np.pi * (np.array(xc) - 0.5)) ** 4
            * (np.abs(np.array(xc) - 0.5) < 0.25)
        ).reshape((np.array(xc).size, 1, 1))

        for phase in Phases:
            phase_alpha = alphas[phase] * np.ones_like(p)
            phase_eos = prob.scheme.problem.eos[phase]

            # Get phase data
            phase_rho = rhos[phase] * np.ones_like(phase_alpha)

            # Compute phase data from EOS
            phase_rhoe = phase_eos.rhoe(phase_rho, p)
            phase_c = phase_eos.sound_velocity(phase_rho, p)
            phase_aF2 = phase_eos.dp_drho(phase_rho, phase_rhoe / phase_rho)
            aF += phase_alpha * phase_rho * phase_aF2
            phase_T = phase_eos.T(phase_rhoe / phase_rho, p)

            # Set phase data
            state_array.set_phase(
                phase,
                np.stack(
                    [
                        phase_alpha,
                        phase_alpha * phase_rho,
                        phase_alpha * phase_rhoe,
                        p,
                        phase_T,
                        phase_c,
                    ],
                    axis=-1,
                ),
            )

        # Get the needed data from each phase

        arho1 = state_array[..., fields.arho1]
        arho2 = state_array[..., fields.arho2]
        arhoe1 = state_array[..., fields.arhoe1]
        arhoe2 = state_array[..., fields.arhoe2]

        c1 = state_array[..., fields.c1]
        c2 = state_array[..., fields.c2]

        # Compute all the fields
        rho = arho1 + arho2
        rhoU = rho * U
        rhoV = rho * V
        rhoe = arhoe1 + arhoe2
        rhoE = rhoe + 0.5 * (rhoU * U + rhoV * V)
        cF = np.sqrt(
            np.divide(arho1 * np.power(c1, 2) + arho2 * np.power(c2, 2), rho)
        )
        state_array[..., fields.rho] = rho
        state_array[..., fields.rhoU] = rhoU
        state_array[..., fields.rhoV] = rhoV
        state_array[..., fields.rhoE] = rhoE
        state_array[..., fields.rhoe] = rhoe
        state_array[..., fields.U] = U
        state_array[..., fields.V] = V
        state_array[..., fields.cF] = cF
        state_array[..., fields.P] = p
        cells.values = state_array

    scheme = prob.scheme
    solver = TwoPhaseSolver(mesh, scheme)
    solver.init(init_fun)

    final_time = prob.final_time
    t = 0.0
    CFL = prob.CFL

    # :: Plot stuff ::
    fields = Q.fields
    fields_to_plot = [
        fields.rho,
        fields.alpha1,
        fields.p1,
        fields.p2,
    ]
    num_fields = len(fields_to_plot) // 2

    time_series = []
    artists = []
    axes = []

    x = solver.mesh.cells.centroids[..., 0]
    x = x.reshape(x.size)

    fig = plt.figure()

    # First plot the mixture pressure
    num_fields += 1
    gs = GridSpec(num_fields, 2)
    ax: plt.Axes = fig.add_subplot(gs[0, :])
    P = solver.mesh.cells.values[..., fields.P].reshape(x.size)
    (line,) = ax.plot(x, P, label=r"$P$")
    ax.legend(loc="best")
    ax.set_ylim(1e5 - 0.5, 1e5 + 0.5)
    time_annotation = fig.text(
        0.5, 0.05, "t=0.00s", horizontalalignment="center"
    )
    artists.append(line)
    axes.append(ax)

    for i, field in enumerate(fields_to_plot, 2):
        # Indices in the plot grid
        idx, idy = np.unravel_index(i, (num_fields, 2))
        axi: plt.Axes = fig.add_subplot(gs[idx, idy])
        field_value = solver.mesh.cells.values[..., field].reshape(x.size)
        (line,) = axi.plot(x, field_value, label=field.name)
        axi.legend(loc="best")
        artists.append(line)
        if field == fields.p1 or field == fields.p2:
            axi.set_ylim(1e5 - 1, 1e5 + 1)
        axes.append(axi)

    # :: End Plot Stuff ::

    ind_max_L_init = np.argmax(solver.mesh.cells.values[:, 0, 0, fields.P])
    x_max_L_init = solver.mesh.cells.centroids[ind_max_L_init, 0, 0, 0]

    while t <= final_time:
        # :: Plot Stuff ::
        x = solver.mesh.cells.centroids[..., 0]
        x = x.reshape(x.size)

        # :: End Plot Stuff ::
        dt = scheme.CFL(
            solver.mesh.cells,
            CFL,
        )
        ind_max_L = np.argmax(solver.mesh.cells.values[:100, 0, 0, fields.P])
        x_max_L = solver.mesh.cells.centroids[ind_max_L, 0, 0, 0]

        # Instantaneous velocity
        # v_t = np.abs(x_max_L - x_max_L_init) / t

        if plot:
            time_series.append(
                (t, x, np.copy(solver.mesh.cells.values).view(Q))
            )

        assert ~np.isnan(dt)
        solver.step(dt)

        t += dt

        print(f"Time: {t}, dt: {dt}")

    # 4-/3-equation system velocity
    cF = solver.mesh.cells.values[ind_max_L, 0, 0, fields.cF]
    alpha1 = solver.mesh.cells.values[ind_max_L, 0, 0, fields.alpha1]
    alpha2 = solver.mesh.cells.values[ind_max_L, 0, 0, fields.alpha2]
    rho1 = solver.mesh.cells.values[ind_max_L, 0, 0, fields.arho1] / alpha1
    rho2 = solver.mesh.cells.values[ind_max_L, 0, 0, fields.arho2] / alpha2
    c1 = solver.mesh.cells.values[ind_max_L, 0, 0, fields.c1]
    c2 = solver.mesh.cells.values[ind_max_L, 0, 0, fields.c2]
    rho = alpha1 * rho1 + alpha2 * rho2

    cW = 1 / np.sqrt(
        rho * (alpha1 / (rho1 * c1**2) + alpha2 / (rho2 * c2**2))
    )
    print("Cfrozen = " + str(cF))
    print("Cwood = " + str(cW))
    # print("Measured velocity = " + str(v_t))

    # assert np.abs(v_t - cW) / cW < 0.2

    fig.tight_layout()
    fig.subplots_adjust(
        bottom=0.15,
        top=0.95,
        hspace=0.35,
    )
    ani = FuncAnimation(
        fig,
        plot_func,
        [
            (data[0], data[1], data[2])
            for i, data in enumerate(time_series)
            if i % 1 == 0
            # Set i % 20 to plot 1 frame on 20
        ],
        fargs=(time_annotation, artists, axes, fields_to_plot),
        repeat=False,
        interval=1,
    )

    if write:
        mywriter = FFMpegWriter()
        ani.save("twophase-accoustics.mp4", writer=mywriter)

    if plot:
        plt.show()

    plt.close()
