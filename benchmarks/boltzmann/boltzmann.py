import os
import time
from typing import Callable
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)  # Disable FutureWarning

import numpy as np
from scipy.linalg import cholesky_banded
import pykokkos as pk

try:
    from kernels import stack_profiler as sp
    from kernels import reduction
except:
    from .kernels import stack_profiler as sp
    from .kernels import reduction


class ArrayLib:
    """
    Performance portability layer for numpy/cupy
    """

    def __init__(self, space: str) -> None:
        if space == "OpenMP" or space == "DebugOpenMP":
            self.lib = np
            self.random = np.random
            pk.set_default_space(pk.OpenMP if space== "OpenMP" else pk.DebugOpenMP)
        elif space == "Cuda" or space == "DebugCuda":
            import cupy as cp

            self.lib = cp
            self.random = cp.random
            pk.set_default_space(pk.Cuda if space== "Cuda" else pk.DebugCuda)
        elif space == "HIP" or space == "DebugHIP":
            import cupy as cp

            self.lib = cp
            self.random = cp.random
            pk.set_default_space(pk.HIP if space== "HIP" else pk.DebugHIP)
        else:
            raise RuntimeError(f"Undefined execution space '{space}'")

        self.space = space

    def use_device(self, device_id: int) -> None:
        if self.space in ("OpenMP", "DebugOpenMP"):
            return

        self.lib.cuda.Device(device_id).use()

    def zeros(self, shape, dtype=None, use_device_order=False) -> np.ndarray:
        use_device_order=False
        if use_device_order:
            if self.space in ("OpenMP", "DebugOpenMP"):
                order = "C"
            else:
                order = "F"
        else:
            order = "C"

        if dtype is None:
            return self.lib.zeros(shape)

        return self.lib.zeros(shape, dtype)

    def ones(self, shape, dtype=None) -> np.ndarray:
        if dtype is None:
            return self.lib.ones(shape)

        return self.lib.ones(shape, dtype)

    def multiply(self, arr1, arr2) -> np.ndarray:
        return self.lib.multiply(arr1, arr2)

    def add(self, arr1, arr2) -> np.ndarray:
        return self.lib.add(arr1, arr2)

    def device_synchronize(self):
        if self.space in ("OpenMP", "DebugOpenMP"):
            return

        self.lib.cuda.runtime.deviceSynchronize()

    def asarray(self, array) -> np.ndarray:
        return self.lib.asarray(array)

    def arange(self, value):
        return self.lib.arange(value)

    def save(self, file, array):
        self.lib.save(file, array)

    def load(self, file) -> np.ndarray:
        return self.lib.load(file)

    def array_equal(self, arr1, arr2) -> bool:
        return self.lib.array_equal(arr1, arr2)


xp = "TBD"

NUM_GPUS = 1


class BoltzmannDSMC:
    def __init__(self, Nc, gridpts, ecov, join, space):
        global xp
        xp = ArrayLib(space)

        self.reduction: Callable
        self.EF_kernel: Callable
        self.electron_kernel: Callable
        self.heavies_kernel_fluid: Callable
        self.post_processing: Callable
        self._init_kernels(join)

        # used for indexing in some places for clarity
        self.xx: int = 0
        self.vx: int = 3
        self.vy: int = 4
        self.vz: int = 5
        self.wt: int = 0
        self.ai: int = 1
        self.ae: int = 2
        self.en: int = 3

        ## Heavy parameters
        self.nn = 3.22e16
        self.nn_Di = 2.07e18
        self.nn_mui = 4.65e19
        self.nn_Ds = 2.42e18
        self.D_i = self.nn_Di / self.nn
        self.D_s = self.nn_Ds / self.nn
        self.mu_i = self.nn_mui / self.nn
        self.epsilon = 5.5263e5  # permitivity
        self.m_e = 9.10938 * 10 ** (-31)
        self.j_ev = 6.241509 * (10**18)

        ## 1-D electric field
        self.freq = 13.56e6
        self.V0 = 100

        ## Base numerical stuff findme
        self.N: int = gridpts - 1
        self.big_N: int = 4 * gridpts
        self.L: float = 2.00
        self.tau: float = 1.0 / self.freq  # period
        self.dx: float = self.L / self.N

        ## Time
        self.dt_big: float = self.tau / 50000
        self.dt_ratio: int = 10
        self.dt_el: float = self.dt_big / self.dt_ratio
        self.curr_t: float = 0.0

        ## Particles
        self.Nc_list = []
        self.Na_list = []
        self.Nmax_list = []
        self.Nnew_list = []
        self.Nmax_fac = 3.0
        Nsplit = int(Nc / NUM_GPUS)  # taking input
        for ng in range(NUM_GPUS):
            self.Nc_list.append(Nsplit)
            self.Na_list.append(Nsplit)
            self.Nmax_list.append(int(self.Nmax_fac * Nsplit))
            self.Nnew_list.append(0)

        ## Initial conditions
        self.E_cov: float = ecov
        ic_dens: int = 1e9

        ## Setup work arrays
        self.E_ar = np.zeros(self.N)
        self.Ji_ar = np.zeros(self.N)
        self.Js_ar = np.zeros(self.N)
        self.ni_rhs = np.zeros(self.N + 1)
        self.ns_rhs = np.zeros(self.N + 1)
        self.V_ar = np.zeros(self.N - 1)
        self.V_rhs = np.zeros(self.N - 1)
        self.ne_ar = np.zeros(self.N + 1)
        self.ni_ar = np.ones(self.N + 1) * ic_dens
        self.ns_ar = np.zeros(self.N + 1)
        self.nrg_ar = np.zeros(self.N + 1)
        self.counter_g0_ar = np.zeros(self.N + 1)
        self.counter_g1_ar = np.zeros(self.N + 1)
        self.counter_g2_ar = np.zeros(self.N + 1)
        self.counter_g3_ar = np.zeros(self.N + 1)
        self.Te_ar = np.zeros(self.N + 1)

        # Defining uniform particle wt with base IC
        self.ww: float = ic_dens * (self.N / Nc)

        # For Dilip's sum algorithm
        self.data_out_list = []
        self.data_out_np_list = []
        self.data_out_list2 = []
        self.data_out_np_list2 = []
        self.temp_x_list = []
        for ng in range(NUM_GPUS):
            xp.use_device(ng)

            self.data_out_list.append(xp.zeros((self.big_N, 4), dtype=np.float64))
            self.data_out_np_list.append(np.zeros((self.big_N, 4), dtype=np.float64))
            self.data_out_list2.append(xp.zeros((self.big_N, 4), dtype=np.float64))
            self.data_out_np_list2.append(np.zeros((self.big_N, 4), dtype=np.float64))

            self.temp_x_list.append(xp.zeros(self.Nmax_list[ng]))

        # For MPI
        self.ni_src = np.zeros(self.N + 1)
        self.ns_src = np.zeros(self.N + 1)

        # GPU Stuff
        self.threads_per_block: int = 64
        self.num_blocks: int = 1024

        # EF stuff
        self.Vc_diag, self.Vc_lower_diag, self.V_tempy = self._chol_EF(self.N, self.dx)

        # INITIALIZE
        ## Setting up particles and particle bins
        kTm = 1.5 * 1.1697 * 10**11  # 1 eV
        cov = [
            [self.E_cov * kTm, 0, 0],
            [0, self.E_cov * kTm, 0],
            [0, 0, self.E_cov * kTm],
        ]

        self.big_data_ar_list = []
        self.big_tosum_ar_list = []
        self.big_curr_xbins_list = []
        self.big_forE_xbins_list = []
        self.big_collct_ar_list = []
        self.nni_list = []

        xp.random.seed(123456789)

        for ng in range(NUM_GPUS):
            xp.use_device(ng)

            Nc_temp = self.Nc_list[ng]
            Nmax_temp = self.Nmax_list[ng]

            print(f"Nc temp = {Nc_temp}")
            print(f"Nmax temp = {Nmax_temp}")
            tot_data_ar = xp.zeros((Nmax_temp, 6), use_device_order=True)
            tot_tosum_ar = xp.zeros((Nmax_temp, 4), use_device_order=True)

            # Particle Position + Velocity
            tot_data_ar[
                0:Nc_temp, self.vx : self.vz + 1
            ] = xp.random.multivariate_normal([0, 0, 0], cov, Nc_temp)
            tot_data_ar[0:Nc_temp, self.xx] = xp.arange(Nc_temp) * (
                self.L / (Nc_temp - 1)
            )
            tot_data_ar[0, self.xx] += 0.000001 * self.dx
            tot_data_ar[Nc_temp - 1, self.xx] -= 0.000001 * self.dx

            # Dummy entries for reduction nonsense
            tot_data_ar[Nc_temp:Nmax_temp, self.xx] = self.L * xp.random.rand(
                Nmax_temp - Nc_temp
            )

            # Weights
            tot_tosum_ar[0:Nc_temp, self.wt] = self.ww * xp.ones(Nc_temp)
            # Energy
            en_xx = xp.multiply(
                tot_data_ar[0:Nc_temp, self.vx], tot_data_ar[0:Nc_temp, self.vx]
            )
            en_yy = xp.multiply(
                tot_data_ar[0:Nc_temp, self.vy], tot_data_ar[0:Nc_temp, self.vy]
            )
            en_zz = xp.multiply(
                tot_data_ar[0:Nc_temp, self.vz], tot_data_ar[0:Nc_temp, self.vz]
            )
            tot_tosum_ar[0:Nc_temp, self.en] = (
                0.5 * self.m_e * self.j_ev * xp.add(en_xx, xp.add(en_yy, en_zz))
            )

            temp_big_collct_ar = xp.zeros((Nmax_temp, 4))
            self.big_collct_ar_list.append(temp_big_collct_ar)

            # Shuffling
            xp.random.seed(1)
            xp.random.shuffle(tot_data_ar[0:Nc_temp, :])
            xp.random.seed(1)
            xp.random.shuffle(tot_tosum_ar[0:Nc_temp, :])

            self.big_data_ar_list.append(tot_data_ar)
            self.big_tosum_ar_list.append(tot_tosum_ar)

            # Bins
            curr_xbins_temp = xp.zeros(Nmax_temp).astype(int)
            forE_xbins_temp = xp.zeros(Nmax_temp).astype(int)
            curr_xbins_temp[0:Nc_temp] = (
                (self.big_data_ar_list[ng][0:Nc_temp, self.xx] + 0.5 * self.dx)
                / self.dx
            ).astype(int)
            forE_xbins_temp[0:Nc_temp] = (
                self.big_data_ar_list[ng][0:Nc_temp, self.xx] / self.dx
            ).astype(int)
            self.big_curr_xbins_list.append(curr_xbins_temp)
            self.big_forE_xbins_list.append(forE_xbins_temp)

            # Updating particle count
            self.nni_list.append(0)

        ## Grid arrays
        self.counter_g0_ar_list = []
        self.counter_g1_ar_list = []
        self.counter_g2_ar_list = []
        self.counter_g3_ar_list = []
        for ng in range(NUM_GPUS):
            self.counter_g0_ar_list.append(np.zeros(self.N + 1))
            self.counter_g1_ar_list.append(np.zeros(self.N + 1))
            self.counter_g2_ar_list.append(np.zeros(self.N + 1))
            self.counter_g3_ar_list.append(np.zeros(self.N + 1))

        self.ne_ar_sum = np.zeros((self.N + 1, NUM_GPUS))
        self.ni_src_sum = np.zeros((self.N + 1, NUM_GPUS))
        self.ns_src_sum = np.zeros((self.N + 1, NUM_GPUS))
        self.nrg_ar_sum = np.zeros((self.N + 1, NUM_GPUS))

        self.ne_ar.fill(0)
        self.ni_src.fill(0)
        self.ns_src.fill(0)
        self.nrg_ar.fill(0)

        # Initialize grid arrays
        for ng in range(NUM_GPUS):
            xp.use_device(ng)

            (
                self.ne_ar_sum,
                self.ni_src_sum,
                self.ns_src_sum,
                self.nrg_ar_sum,
                self.counter_g0_ar_list[ng],
                self.counter_g1_ar_list[ng],
                self.counter_g2_ar_list[ng],
                self.counter_g3_ar_list[ng],
                self.Nnew_list[ng],
                self.Na_list[ng],
            ) = self.post_processing(
                ng,
                self.data_out_list[ng],
                self.data_out_np_list[ng],
                self.data_out_list2[ng],
                self.data_out_np_list2[ng],
                self.big_data_ar_list[ng],
                self.big_tosum_ar_list[ng],
                self.big_collct_ar_list[ng],
                self.Na_list[ng],
                self.Nc_list[ng],
                self.nni_list[ng],
                self.ne_ar_sum,
                self.ni_src_sum,
                self.ns_src_sum,
                self.nrg_ar_sum,
                self.counter_g0_ar_list[ng],
                self.counter_g1_ar_list[ng],
                self.counter_g2_ar_list[ng],
                self.counter_g3_ar_list[ng],
                self.big_N,
                self.N + 1,
                self.L,
                self.dx,
                self.ww,
                self.temp_x_list[ng],
            )

        self.ne_ar[:] = np.sum(self.ne_ar_sum, axis=1)
        self.ni_src[:] = np.sum(self.ni_src_sum, axis=1)
        self.ns_src[:] = np.sum(self.ns_src_sum, axis=1)
        self.nrg_ar[:] = np.sum(self.nrg_ar_sum, axis=1)
        self.Te_ar.fill(0)
        self.Te_ar[:] = np.divide(
            (2.0 / 3.0) * self.ww * self.nrg_ar,
            self.ne_ar,
            where=self.ne_ar > 0.9 * self.ww,
        )
        self.Te_ar[np.where(self.Te_ar > 20)] = 0.0

        # Initializing EF
        self.Vcarry_ar, self.E_ar = self.EF_kernel(
            self.E_ar,
            self.ne_ar,
            self.ni_ar,
            self.curr_t,
            self.V_ar,
            self.V_rhs,
            self.Vc_diag,
            self.Vc_lower_diag,
            self.V_tempy,
            self.N,
            self.dx,
            self.epsilon,
            self.V0,
            self.freq,
        )

        ## Random #s, recombination, atomics, null coll method
        self.forgpu_R_vec_list = []
        self.curr_count_list = []

        for ng in range(NUM_GPUS):
            xp.use_device(ng)

            forgpu_R_vec_temp = xp.zeros((self.Nmax_list[ng], 7))
            forgpu_R_vec_temp[:, :] = xp.random.rand(self.Nmax_list[ng], 7)
            self.forgpu_R_vec_list.append(forgpu_R_vec_temp)

            curr_count_temp = xp.zeros(1).astype(int)
            self.curr_count_list.append(curr_count_temp)

        ## For copying to GPU and device arrays for electron kernel
        self.gpu_E_ar_list = []
        self.gpu_ns_ar_list = []
        self.np_data_ar = np.zeros(5 * self.N + 4)
        self.cp_data_ar_list = []

        self.d_curr_count_list = []
        self.d_currxbins_list = []
        self.d_forExbins_list = []
        self.d_bigRvec_list = []
        self.d_data_ar_list = []
        self.d_tosum_ar_list = []
        self.d_collct_ar_list = []
        self.d_E_ar_list = []
        self.d_ns_ar_list = []

        for ng in range(NUM_GPUS):
            xp.use_device(ng)

            gpu_E_ar_temp = xp.zeros(self.N)
            gpu_ns_ar_temp = xp.zeros(self.N + 1)
            cp_data_ar_temp = xp.zeros(5 * self.N + 4)
            self.gpu_E_ar_list.append(gpu_E_ar_temp)
            self.gpu_ns_ar_list.append(gpu_ns_ar_temp)
            self.cp_data_ar_list.append(cp_data_ar_temp)

            d_curr_count_temp = self.curr_count_list[ng]
            d_currxbins_temp = self.big_curr_xbins_list[ng]
            d_forExbins_temp = self.big_forE_xbins_list[ng]
            d_bigRvec_temp = self.forgpu_R_vec_list[ng]
            d_data_ar_temp = self.big_data_ar_list[ng]
            d_tosum_ar_temp = self.big_tosum_ar_list[ng]
            d_collct_ar_temp = self.big_collct_ar_list[ng]
            d_E_ar_temp = self.gpu_E_ar_list[ng]
            d_ns_ar_temp = self.gpu_ns_ar_list[ng]

            self.d_curr_count_list.append(d_curr_count_temp)
            self.d_currxbins_list.append(d_currxbins_temp)
            self.d_forExbins_list.append(d_forExbins_temp)
            self.d_bigRvec_list.append(d_bigRvec_temp)
            self.d_data_ar_list.append(d_data_ar_temp)
            self.d_tosum_ar_list.append(d_tosum_ar_temp)
            self.d_collct_ar_list.append(d_collct_ar_temp)
            self.d_E_ar_list.append(d_E_ar_temp)
            self.d_ns_ar_list.append(d_ns_ar_temp)

        ## For recombination and for counting collisions
        self.did_g0_ar_list = []
        self.did_g1_ar_list = []
        self.did_g2_ar_list = []
        self.did_g3_ar_list = []
        self.d_g0_ar_list = []
        self.d_g1_ar_list = []
        self.d_g2_ar_list = []
        self.d_g3_ar_list = []
        self.coll_indicator_g0_ar_list = []
        self.coll_indicator_g1_ar_list = []
        self.coll_indicator_g2_ar_list = []
        self.coll_indicator_g3_ar_list = []
        self.d_coll_indicator_g0_ar_list = []
        self.d_coll_indicator_g1_ar_list = []
        self.d_coll_indicator_g2_ar_list = []
        self.d_coll_indicator_g3_ar_list = []
        for ng in range(NUM_GPUS):
            xp.use_device(ng)

            did_g0_ar_temp = xp.zeros(self.Nmax_list[ng])
            did_g1_ar_temp = xp.zeros(self.Nmax_list[ng])
            did_g2_ar_temp = xp.zeros(self.Nmax_list[ng])
            did_g3_ar_temp = xp.zeros(self.Nmax_list[ng])
            self.did_g0_ar_list.append(did_g0_ar_temp)
            self.did_g1_ar_list.append(did_g1_ar_temp)
            self.did_g2_ar_list.append(did_g2_ar_temp)
            self.did_g3_ar_list.append(did_g3_ar_temp)

            coll_indicator_g0_ar_temp = xp.zeros((4, self.Nmax_list[ng])).astype(int)
            # coll_indicator_g1_ar_temp = cp.zeros(self.Nmax_list[ng]).astype(int)
            # coll_indicator_g2_ar_temp = cp.zeros(self.Nmax_list[ng]).astype(int)
            # coll_indicator_g3_ar_temp = cp.zeros(self.Nmax_list[ng]).astype(int)
            self.coll_indicator_g0_ar_list.append(coll_indicator_g0_ar_temp)
            # self.coll_indicator_g1_ar_list.append(coll_indicator_g1_ar_temp)
            # self.coll_indicator_g2_ar_list.append(coll_indicator_g2_ar_temp)
            # self.coll_indicator_g3_ar_list.append(coll_indicator_g3_ar_temp)

            d_g0_ar_temp = self.did_g0_ar_list[ng]
            d_g1_ar_temp = self.did_g1_ar_list[ng]
            d_g2_ar_temp = self.did_g2_ar_list[ng]
            d_g3_ar_temp = self.did_g3_ar_list[ng]
            self.d_g0_ar_list.append(d_g0_ar_temp)
            self.d_g1_ar_list.append(d_g1_ar_temp)
            self.d_g2_ar_list.append(d_g2_ar_temp)
            self.d_g3_ar_list.append(d_g3_ar_temp)

            d_coll_indicator_g0_temp = self.coll_indicator_g0_ar_list[ng]

            # d_coll_indicator_g1_temp = cuda.to_device(
            #     self.coll_indicator_g1_ar_list[ng]
            # )
            # d_coll_indicator_g2_temp = cuda.to_device(
            #     self.coll_indicator_g2_ar_list[ng]
            # )
            # d_coll_indicator_g3_temp = cuda.to_device(
            #     self.coll_indicator_g3_ar_list[ng]
            # )
            self.d_coll_indicator_g0_ar_list.append(d_coll_indicator_g0_temp)
            # self.d_coll_indicator_g1_ar_list.append(d_coll_indicator_g1_temp)
            # self.d_coll_indicator_g2_ar_list.append(d_coll_indicator_g2_temp)
            # self.d_coll_indicator_g3_ar_list.append(d_coll_indicator_g3_temp)

        self._setup_kernels()

    ## MAIN TIMESTEPPING LOOP
    def run(self, num_steps, stable):
        print(f"STARTING RUN {num_steps=}")

        self.np_data_ar[0 : self.N] = self.E_ar[0 : self.N]
        self.np_data_ar[self.N : 2 * self.N + 1] = self.ne_ar[0 : self.N + 1]
        self.np_data_ar[2 * self.N + 1 : 3 * self.N + 2] = self.ni_ar[0 : self.N + 1]
        self.np_data_ar[3 * self.N + 2 : 4 * self.N + 3] = self.ns_ar[0 : self.N + 1]
        self.np_data_ar[4 * self.N + 3 : 5 * self.N + 4] = self.Te_ar[0 : self.N + 1]

        # Optional verbose timings to pinpoint where DebugCuda slows down.
        # Enable with:
        #   PKDB_BOLTZ_VERBOSE_TIMINGS=1
        # Optionally limit verbosity:
        #   PKDB_BOLTZ_VERBOSE_TIMINGS_BIGSTEPS=1
        #   PKDB_BOLTZ_VERBOSE_TIMINGS_NG=0
        verbose_timings: bool = bool(int(os.environ.get("PKDB_BOLTZ_VERBOSE_TIMINGS", "0")))
        verbose_bigsteps: int = int(os.environ.get("PKDB_BOLTZ_VERBOSE_TIMINGS_BIGSTEPS", "1"))
        verbose_ng: int = int(os.environ.get("PKDB_BOLTZ_VERBOSE_TIMINGS_NG", "0"))

        # Accumulators for total time across all steps
        if verbose_timings:
            total_copy = 0.0
            total_electron = 0.0
            total_post = 0.0
            total_fill = 0.0
            total_pde = 0.0
            total_curr_count_read = 0.0
            total_other = 0.0

        # Reset post_processing detailed timings
        if verbose_timings:
            from kernels.post_processing import reset_detail_timings
            reset_detail_timings()

        ta = time.time()
        for bigstep in range(num_steps):
            ## Task 0: GPU Steps
            for ng in range(NUM_GPUS):
                if verbose_timings and bigstep < verbose_bigsteps and ng == verbose_ng:
                    t0 = time.perf_counter()

                ## COPY TO GPU
                t_copy_start = time.perf_counter() if verbose_timings else None
                self.cp_data_ar_list[ng][:] = xp.asarray(self.np_data_ar[:])
                self.gpu_E_ar_list[ng][:] = self.cp_data_ar_list[ng][0 : self.N]
                self.gpu_ns_ar_list[ng][:] = self.cp_data_ar_list[ng][
                    3 * self.N + 2 : 4 * self.N + 3
                ]
                self.curr_count_list[ng].fill(0)

                self.forgpu_R_vec_list[ng][0 : self.Nc_list[ng], :] = xp.random.rand(
                    self.Nc_list[ng], 7
                )

                xp.device_synchronize()

                if verbose_timings:
                    t_copy_end = time.perf_counter()
                    t_copy = t_copy_end - t_copy_start
                    total_copy += t_copy
                    te0 = time.perf_counter()

                ## ELECTRON KERNEL
                sp.profile(
                    self.electron_kernel,
                    self.Nc_list[ng],
                    self.d_data_ar_list[ng],
                    self.d_tosum_ar_list[ng],
                    self.d_collct_ar_list[ng],
                    self.d_E_ar_list[ng],
                    self.d_ns_ar_list[ng],
                    self.d_bigRvec_list[ng],
                    self.d_currxbins_list[ng],
                    self.d_forExbins_list[ng],
                    1e6 * self.nn,
                    self.dt_el,
                    self.d_curr_count_list[ng],
                    self.num_blocks,
                    self.threads_per_block,
                    self.d_g0_ar_list[ng],
                    self.d_g1_ar_list[ng],
                    self.d_g2_ar_list[ng],
                    self.d_g3_ar_list[ng],
                    self.d_coll_indicator_g0_ar_list[ng],
                    # self.d_coll_indicator_g1_ar_list[ng],
                    # self.d_coll_indicator_g2_ar_list[ng],
                    # self.d_coll_indicator_g3_ar_list[ng],
                    stable,
                )

                if verbose_timings:
                    te1 = time.perf_counter()
                    t_electron = te1 - te0
                    total_electron += t_electron

                if "PK_BOLTZ_EARLY_EXIT" in os.environ:
                    import sys
                    sys.exit()

                ## POST PROCESSING
                if verbose_timings:
                    tcurr0 = time.perf_counter()
                self.nni_list[ng] = int(self.curr_count_list[ng][0])
                if verbose_timings:
                    tcurr1 = time.perf_counter()
                    total_curr_count_read += (tcurr1 - tcurr0)
                    tp0 = time.perf_counter()
                else:
                    tp0 = None
                (
                    self.ne_ar_sum,
                    self.ni_src_sum,
                    self.ns_src_sum,
                    self.nrg_ar_sum,
                    self.counter_g0_ar_list[ng],
                    self.counter_g1_ar_list[ng],
                    self.counter_g2_ar_list[ng],
                    self.counter_g3_ar_list[ng],
                    self.Nnew_list[ng],
                    self.Na_list[ng],
                ) = self.post_processing(
                    ng,
                    self.data_out_list[ng],
                    self.data_out_np_list[ng],
                    self.data_out_list2[ng],
                    self.data_out_np_list2[ng],
                    self.big_data_ar_list[ng],
                    self.big_tosum_ar_list[ng],
                    self.big_collct_ar_list[ng],
                    self.Na_list[ng],
                    self.Nc_list[ng],
                    self.nni_list[ng],
                    self.ne_ar_sum,
                    self.ni_src_sum,
                    self.ns_src_sum,
                    self.nrg_ar_sum,
                    self.counter_g0_ar_list[ng],
                    self.counter_g1_ar_list[ng],
                    self.counter_g2_ar_list[ng],
                    self.counter_g3_ar_list[ng],
                    self.big_N,
                    self.N + 1,
                    self.L,
                    self.dx,
                    self.ww,
                    self.temp_x_list[ng],
                )

                if verbose_timings:
                    tp1 = time.perf_counter()
                    t_post = tp1 - tp0
                    total_post += t_post

                ## FILLS AND BINNING
                if verbose_timings:
                    tf0 = time.perf_counter()
                self.big_data_ar_list[ng][
                    self.Nnew_list[ng] :, self.vx : self.vz + 1
                ].fill(0)
                self.big_tosum_ar_list[ng][self.Nnew_list[ng] :, :].fill(0)
                self.big_tosum_ar_list[ng][:, self.ai : self.ae + 1].fill(0)
                self.big_forE_xbins_list[ng][0 : self.Nnew_list[ng]] = (
                    self.big_data_ar_list[ng][0 : self.Nnew_list[ng], self.xx] / self.dx
                ).astype(int)
                self.big_curr_xbins_list[ng][0 : self.Nnew_list[ng]] = (
                    (
                        self.big_data_ar_list[ng][0 : self.Nnew_list[ng], self.xx]
                        + 0.5 * self.dx
                    )
                    / self.dx
                ).astype(int)
                # Update # particles
                self.Nc_list[ng] = self.Nnew_list[ng]
                self.curr_count_list[ng].fill(0)
                xp.device_synchronize()

                if verbose_timings:
                    tf1 = time.perf_counter()
                    t_fill = tf1 - tf0
                    total_fill += t_fill

                    if bigstep < verbose_bigsteps and ng == verbose_ng:
                        t1 = time.perf_counter()
                        print(
                            "[boltzmann timings]"
                            f" bigstep={bigstep} ng={ng} "
                            f"copy_to_gpu+sync={t_copy:.6f}s "
                            f"electron_kernel(host)={t_electron:.6f}s "
                            f"post_processing={t_post:.6f}s "
                            f"fill+binning+sync={t_fill:.6f}s "
                            f"total_ng_block={t1 - t0:.6f}s"
                        )

                # Making sure we calc these properly each step
                self.ne_ar[:] = np.sum(self.ne_ar_sum, axis=1)
                self.ni_src[:] = np.sum(self.ni_src_sum, axis=1)
                self.ns_src[:] = np.sum(self.ns_src_sum, axis=1)
                self.nrg_ar[:] = np.sum(self.nrg_ar_sum, axis=1)

                # Update time
                self.curr_t += self.dt_big

            # PDE Solves - Fluid + EF
            if verbose_timings:
                t_pde0 = time.perf_counter()
            self.ni_ar, self.ns_ar = self.heavies_kernel_fluid(
                self.E_ar,
                self.ni_ar,
                self.ni_rhs,
                self.Ji_ar,
                self.ns_ar,
                self.ns_rhs,
                self.Js_ar,
                self.dx,
                self.dt_big,
                self.mu_i,
                self.D_i,
                self.D_s,
                self.N,
            )
            self.Vcarry_ar, self.E_ar = self.EF_kernel(
                self.E_ar,
                self.ne_ar,
                self.ni_ar,
                self.curr_t,
                self.V_ar,
                self.V_rhs,
                self.Vc_diag,
                self.Vc_lower_diag,
                self.V_tempy,
                self.N,
                self.dx,
                self.epsilon,
                self.V0,
                self.freq,
            )
            self.np_data_ar[0 : self.N] = self.E_ar[0 : self.N]
            self.np_data_ar[self.N : 2 * self.N + 1] = self.ne_ar[0 : self.N + 1]
            self.np_data_ar[2 * self.N + 1 : 3 * self.N + 2] = self.ni_ar[
                0 : self.N + 1
            ]
            self.np_data_ar[3 * self.N + 2 : 4 * self.N + 3] = self.ns_ar[
                0 : self.N + 1
            ]
            self.np_data_ar[4 * self.N + 3 : 5 * self.N + 4] = self.Te_ar[
                0 : self.N + 1
            ]
            if verbose_timings:
                t_pde1 = time.perf_counter()
                total_pde += (t_pde1 - t_pde0)

                if bigstep < verbose_bigsteps:
                    print(
                        "[boltzmann timings]"
                        f" bigstep={bigstep} pde_solve(fluid+ef)={t_pde1 - t_pde0:.6f}s"
                    )

        pk.flush()
        tb = time.time()
        total_measured = tb - ta

        if verbose_timings:
            # Get detailed post_processing breakdown
            from kernels.post_processing import get_detail_timings
            detail = get_detail_timings()

            total_accounted = total_copy + total_electron + total_post + total_fill + total_pde + total_curr_count_read
            total_other = total_measured - total_accounted

            print("\n" + "="*70)
            print(f"[TIMING SUMMARY across {num_steps} timesteps]")
            print("="*70)
            print(f"  copy_to_gpu+sync:      {total_copy:10.3f}s  ({100*total_copy/total_measured:5.1f}%)")
            print(f"  electron_kernel:       {total_electron:10.3f}s  ({100*total_electron/total_measured:5.1f}%)")
            print(f"  post_processing:       {total_post:10.3f}s  ({100*total_post/total_measured:5.1f}%)")
            if detail["call_count"] > 0:
                rk = detail.get("reduction_kernels", 0.0)
                print(f"    └─ reduction+sync:   {rk:10.3f}s  ({100*rk/total_measured:5.1f}%)")
                print(f"    └─ cp.asnumpy(1st):  {detail['asnumpy1']:10.3f}s  ({100*detail['asnumpy1']/total_measured:5.1f}%)")
                print(f"    └─ cp.asnumpy(2nd):  {detail['asnumpy2']:10.3f}s  ({100*detail['asnumpy2']/total_measured:5.1f}%)")
                other_post = total_post - rk - detail["asnumpy1"] - detail["asnumpy2"]
                print(f"    └─ other:            {other_post:10.3f}s  ({100*other_post/total_measured:5.1f}%)")
            print(f"  fill+binning+sync:     {total_fill:10.3f}s  ({100*total_fill/total_measured:5.1f}%)")
            print(f"  pde_solve:             {total_pde:10.3f}s  ({100*total_pde/total_measured:5.1f}%)")
            print(f"  curr_count_read+sync:  {total_curr_count_read:10.3f}s  ({100*total_curr_count_read/total_measured:5.1f}%)")
            print(
                f"  other/unaccounted:     {total_other:10.3f}s  ({100*total_other/total_measured:5.1f}%)  "
                f"(wall - sum of buckets; np.sum grid merge, pk.flush, Python)"
            )
            print(f"  ─────────────────────────────────────────")
            print(f"  TOTAL:                 {total_measured:10.3f}s  (100.0%)")
            print("="*70 + "\n")

        print(f"total time = {tb - ta}\n")

    def store_results(self, steps: int):
        """
        Store the results of d_data_ar
        """

        for ng in range(NUM_GPUS):
            array = self.d_data_ar_list[ng].xp_array
            Nc = self.Nc_list[ng]

            file_name: str = f"results/{xp.space}_{Nc}_{ng}_{steps}_data_ar.npy"
            xp.save(file_name, array)

    def compare_results(self, steps: int):
        """
        Compare the generated results with the stored results
        """

        for ng in range(NUM_GPUS):
            array = self.d_data_ar_list[ng].xp_array
            Nc = self.Nc_list[ng]

            file_name: str = f"results/{xp.space}_{Nc}_{ng}_{steps}_data_ar.npy"
            with open(file_name, "rb") as f:
                array_correct = xp.load(f)

            if xp.array_equal(array[:, 0], array_correct[:, 0]):
                print(f"GPU {ng} ar 0 is correct")
            else:
                print(f"GPU {ng} ar 0 is wrong")

            if xp.array_equal(array[:, 3:6], array_correct[:, 3:6]):
                print(f"GPU {ng} ar 3:6 is correct")
            else:
                print(f"GPU {ng} ar 3:6 is wrong")

    def _chol_EF(self, N: int, dx: float):
        """
        Set up Cholesky arrays
        """

        Ac = np.zeros((2, N - 1))
        Ac[0, 1 : N - 1] = -1.0 / (dx**2)
        Ac[1, 0 : N - 1] = 2.0 / (dx**2)
        c = cholesky_banded(Ac)
        Vc_lower_diag = np.zeros(N - 1)
        Vc_lower_diag[1 : N - 1] = np.copy(c[0, 1 : N - 1])
        Vc_diag = np.zeros(N - 1)
        Vc_diag[0 : N - 1] = np.copy(c[1, 0 : N - 1])

        V_tempy = np.zeros(N + 1)
        return (Vc_diag, Vc_lower_diag, V_tempy)

    def _init_kernels(self, join) -> None:
        """
        Initialize the kernels stored as member variables

        :param pyk: whether to use pykokkos versions of kernels
        """

        reduction.init_reduction()

        try:
            if join:
                print("RUNNING JOINED")
                from kernels.electron_kernel_joined import electron_kernel
            else:
                print("RUNNING ORIGINAL")
                from kernels.electron_kernel import electron_kernel
            from kernels.EF_kernel import EF_kernel
            from kernels.heavies_kernel import heavies_kernel_fluid
            from kernels.post_processing import post_processing

        except Exception as E:
            if join:
                print("RUNNING JOINED")
                from .kernels.electron_kernel_joined import electron_kernel
            else:
                print("RUNNING ORIGINAL")
                from .kernels.electron_kernel import electron_kernel
            from .kernels.EF_kernel import EF_kernel
            from .kernels.heavies_kernel import heavies_kernel_fluid
            from .kernels.post_processing import post_processing

        self.EF_kernel = EF_kernel
        self.electron_kernel = electron_kernel
        self.heavies_kernel_fluid = heavies_kernel_fluid
        self.post_processing = post_processing

    def _setup_kernels(self) -> None:
        """
        Setup the arrays and functor objects as needed by the kernels
        """

        for ng in range(NUM_GPUS):
            self.d_curr_count_list[ng] = pk.array(
                xp.asarray(self.d_curr_count_list[ng])
            )
            self.d_forExbins_list[ng] = pk.array(xp.asarray(self.d_forExbins_list[ng]))
            self.d_bigRvec_list[ng] = pk.array(xp.asarray(self.d_bigRvec_list[ng]))
            self.d_data_ar_list[ng] = pk.array(xp.asarray(self.d_data_ar_list[ng]))
            self.d_tosum_ar_list[ng] = pk.array(xp.asarray(self.d_tosum_ar_list[ng]))
            self.d_E_ar_list[ng] = pk.array(xp.asarray(self.gpu_E_ar_list[ng]))
            self.d_ns_ar_list[ng] = pk.array(xp.asarray(self.gpu_ns_ar_list[ng]))
            self.d_currxbins_list[ng] = pk.array(xp.asarray(self.d_currxbins_list[ng]))
            self.d_g0_ar_list[ng] = pk.array(xp.asarray(self.d_g0_ar_list[ng]))
            self.d_g1_ar_list[ng] = pk.array(xp.asarray(self.d_g1_ar_list[ng]))
            self.d_g2_ar_list[ng] = pk.array(xp.asarray(self.d_g2_ar_list[ng]))
            self.d_g3_ar_list[ng] = pk.array(xp.asarray(self.d_g3_ar_list[ng]))
            self.d_coll_indicator_g0_ar_list[ng] = pk.array(
                xp.asarray(self.d_coll_indicator_g0_ar_list[ng])
            )
            # self.d_coll_indicator_g1_ar_list[ng] = pk.array(
            #     cp.asarray(self.d_coll_indicator_g1_ar_list[ng])
            # )
            # self.d_coll_indicator_g2_ar_list[ng] = pk.array(
            #     cp.asarray(self.d_coll_indicator_g2_ar_list[ng])
            # )
            # self.d_coll_indicator_g3_ar_list[ng] = pk.array(
            #     cp.asarray(self.d_coll_indicator_g3_ar_list[ng])
            # )
            self.d_collct_ar_list[ng] = pk.array(xp.asarray(self.d_collct_ar_list[ng]))
